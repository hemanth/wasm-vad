/**
 * Zero-Dependency Pure-JS Acoustic-Spectral VAD Engine (`tier: 'dsp'`, 0 KB weights).
 *
 * Combines:
 *   1. Causal 80 Hz HPF + 50/60 Hz Q=22 mains hum biquad conditioning
 *   2. 512-point Hann-windowed Radix-2 FFT (31.25 Hz bin resolution @ 16 kHz)
 *   3. Adaptive Minima-Controlled Recursive Averaging (MCRA) per-bin noise PSD tracker
 *   4. Sohn et al. decision-directed Log-Likelihood Ratio (LLR) a-posteriori/a-priori SNR
 *   5. Vocal-tract formant band (280–3,600 Hz) concentration + harmonic peakiness vs Spectral Flatness
 *      (rejecting pure 50/60 Hz mains hum, stationary HVAC broadband noise, and transient clicks)
 */

import { PreConditioner16k, FRAME_SAMPLES } from './dsp.js';

const FFT_N = 512;
const NUM_BINS = 257; // 0 .. 8000 Hz @ 31.25 Hz/bin

const HANN_WIN = new Float32Array(FFT_N);
const BIT_REV = new Uint16Array(FFT_N);
const TWIDDLE_COS = new Float32Array(FFT_N / 2);
const TWIDDLE_SIN = new Float32Array(FFT_N / 2);

for (let i = 0; i < FFT_N; i++) {
  HANN_WIN[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (FFT_N - 1)));
  let rev = 0;
  let x = i;
  for (let b = 0; b < 9; b++) {
    rev = (rev << 1) | (x & 1);
    x >>>= 1;
  }
  BIT_REV[i] = rev;
}
for (let k = 0; k < FFT_N / 2; k++) {
  const ang = (-2.0 * Math.PI * k) / FFT_N;
  TWIDDLE_COS[k] = Math.cos(ang);
  TWIDDLE_SIN[k] = Math.sin(ang);
}

function fft512RealPower(frame, reBuf, imBuf, psdOut) {
  for (let i = 0; i < FFT_N; i++) {
    reBuf[BIT_REV[i]] = (frame[i] || 0.0) * HANN_WIN[i];
    imBuf[BIT_REV[i]] = 0.0;
  }

  for (let size = 2; size <= FFT_N; size <<= 1) {
    const half = size >>> 1;
    const step = FFT_N / size;
    for (let i = 0; i < FFT_N; i += size) {
      let tw = 0;
      for (let j = 0; j < half; j++) {
        const wr = TWIDDLE_COS[tw];
        const wi = TWIDDLE_SIN[tw];
        tw += step;
        const idxA = i + j;
        const idxB = idxA + half;
        const tr = wr * reBuf[idxB] - wi * imBuf[idxB];
        const ti = wr * imBuf[idxB] + wi * reBuf[idxB];
        reBuf[idxB] = reBuf[idxA] - tr;
        imBuf[idxB] = imBuf[idxA] - ti;
        reBuf[idxA] += tr;
        imBuf[idxA] += ti;
      }
    }
  }

  for (let k = 0; k < NUM_BINS; k++) {
    psdOut[k] = reBuf[k] * reBuf[k] + imBuf[k] * imBuf[k] + 1e-12;
  }
}

export class SpectralVadEngine {
  constructor() {
    this.conditioner = new PreConditioner16k({ highPassHz: 80, humNotch: true });
    this.reBuf = new Float32Array(FFT_N);
    this.imBuf = new Float32Array(FFT_N);
    this.psd = new Float32Array(NUM_BINS);
    this.smoothPsd = new Float32Array(NUM_BINS);
    this.noisePsd = new Float32Array(NUM_BINS);
    this.minTrack = new Float32Array(NUM_BINS);
    this.prevGainSqPsd = new Float32Array(NUM_BINS);
    this.prevProb = 0.0;
    this.frameCount = 0;
    this.reset();
  }

  reset() {
    this.conditioner.reset();
    this.noisePsd.fill(1e-5);
    this.smoothPsd.fill(1e-5);
    this.minTrack.fill(1e-5);
    this.prevGainSqPsd.fill(0.0);
    this.prevProb = 0.0;
    this.frameCount = 0;
  }

  /**
   * Scores a single 512-sample 16 kHz frame and returns `{ probability, snrDb }`.
   */
  scoreFrame(rawFrame) {
    const clean = this.conditioner.process(
      rawFrame.length === FRAME_SAMPLES ? rawFrame : rawFrame.subarray(0, FRAME_SAMPLES)
    );

    let energySq = 0.0;
    let zcr = 0;
    for (let i = 0; i < clean.length; i++) {
      energySq += clean[i] * clean[i];
      if (i > 0 && ((clean[i] >= 0) !== (clean[i - 1] >= 0))) zcr++;
    }
    const rms = Math.sqrt(energySq / Math.max(1, clean.length));
    const zcrRate = zcr / Math.max(1, clean.length);

    if (rms < 4e-4) {
      this.prevProb = this.prevProb * 0.65;
      this.frameCount++;
      return { probability: this.prevProb, snrDb: 0.0 };
    }

    fft512RealPower(clean, this.reBuf, this.imBuf, this.psd);

    // Speech band bins: bin 3 (~94 Hz) .. bin 128 (4,000 Hz)
    // Core vocal-tract formant band: bin 9 (~281 Hz) .. bin 112 (3,500 Hz)
    // Sibilant / fricative band: bin 113 (3,531 Hz) .. bin 224 (7,000 Hz)
    let totalBandEnergy = 1e-12;
    let formantEnergy = 0.0;
    let fricativeEnergy = 0.0;
    let logSumFormant = 0.0;
    let logSumFull = 0.0;
    let formantBins = 0;
    let fullBins = 0;
    let maxBinPower = 0.0;
    let maxBinIdx = 9;

    for (let k = 3; k <= 224; k++) {
      const p = this.psd[k];
      totalBandEnergy += p;
      if (k >= 6) {
        logSumFull += Math.log(p);
        fullBins++;
      }
      if (k >= 9 && k <= 112) {
        formantEnergy += p;
        logSumFormant += Math.log(p);
        formantBins++;
        if (p > maxBinPower) {
          maxBinPower = p;
          maxBinIdx = k;
        }
      } else if (k > 112) {
        fricativeEnergy += p;
      }
    }

    const sfm = Math.min(
      1.0,
      Math.exp(logSumFormant / Math.max(1, formantBins)) / Math.max(formantEnergy / Math.max(1, formantBins), 1e-12)
    );
    const fullSfm = Math.min(
      1.0,
      Math.exp(logSumFull / Math.max(1, fullBins)) / Math.max(totalBandEnergy / Math.max(1, fullBins), 1e-12)
    );

    // Bootstrap noise PSD on frame 0: if broadband flat noise is already present at stream start, seed full PSD
    if (this.frameCount === 0) {
      const looksLikeSpeechAtZero = sfm < 0.36 && fullSfm < 0.32;
      for (let k = 0; k < NUM_BINS; k++) {
        this.smoothPsd[k] = this.psd[k];
        this.noisePsd[k] = looksLikeSpeechAtZero ? Math.min(this.psd[k], 5e-3) : this.psd[k] * 1.15;
        this.minTrack[k] = this.noisePsd[k];
      }
    }

    // Sohn Decision-Directed LLR across vocal-tract + fricative band (bins 6..180: 187 Hz .. 5,625 Hz)
    let llrSum = 0.0;
    let postSnrLinSum = 0.0;
    let llrCount = 0;
    const alphaDd = 0.92;

    for (let k = 6; k <= 180; k++) {
      const yk = this.psd[k];
      const nk = Math.max(this.noisePsd[k], 1e-9);
      const gamma = Math.min(40.0, yk / nk); // a-posteriori SNR
      const xi = alphaDd * (this.prevGainSqPsd[k] / nk) + (1.0 - alphaDd) * Math.max(0.0, gamma - 1.0);
      const v = (xi * gamma) / (1.0 + xi);
      const gain = xi / (1.0 + xi);
      this.prevGainSqPsd[k] = gain * gain * yk;

      const llr = v - Math.log(1.0 + xi);
      llrSum += llr;
      postSnrLinSum += gamma;
      llrCount++;
    }

    const meanLlr = llrSum / llrCount;
    const meanPostSnr = postSnrLinSum / llrCount;
    const snrDb = Math.max(0.0, 10.0 * Math.log10(Math.max(1.0, meanPostSnr)));

    // Hann-windowed pure tone concentrates ~99.9% of energy in 3 adjacent bins (1:4:1 mainlobe),
    // whereas human voiced speech distributes power across multiple F0 harmonics & formants.
    const peak3Power = this.psd[maxBinIdx - 1] + this.psd[maxBinIdx] + this.psd[maxBinIdx + 1];
    const peak3Dominance = peak3Power / Math.max(formantEnergy, 1e-12);
    const formantRatio = formantEnergy / totalBandEnergy;
    const fricativeRatio = fricativeEnergy / totalBandEnergy;

    const isPureTone = peak3Dominance > 0.84;
    const isFlatNoise = fullSfm > 0.48 || sfm > 0.56;
    const isUnvoicedConsonant =
      !isFlatNoise && fricativeRatio > 0.45 && fricativeEnergy > 1.8 * formantEnergy && zcrRate > 0.14 && rms > 0.004;

    // Combine evidence into log-odds
    let logit = -2.45;
    logit += 1.45 * Math.min(4.5, meanLlr);
    logit += 2.15 * Math.max(0.0, formantRatio - 0.35);
    logit += 2.55 * Math.max(0.0, 0.50 - sfm);
    if (rms > 0.012 && !isFlatNoise && !isPureTone) {
      logit += 0.50 * Math.min(1.5, Math.log10(rms / 0.008 + 1.0) * 2.0);
    }
    if (isUnvoicedConsonant) logit += 1.10;
    if (isPureTone) logit -= 3.60;
    if (isFlatNoise) logit -= 3.10;

    const rawProb = 1.0 / (1.0 + Math.exp(-logit));

    // MCRA adaptive noise PSD update: update faster during non-speech, freeze during confident speech
    const speechPresence = rawProb;
    const alphaNoise = 0.88 + 0.115 * speechPresence; // 0.88 in silence, 0.995 in speech
    for (let k = 0; k < NUM_BINS; k++) {
      this.smoothPsd[k] = 0.7 * this.smoothPsd[k] + 0.3 * this.psd[k];
      if (this.frameCount % 45 === 0) {
        this.minTrack[k] = Math.min(this.smoothPsd[k], this.minTrack[k] * 1.35 + 1e-9);
      } else {
        this.minTrack[k] = Math.min(this.minTrack[k], this.smoothPsd[k]);
      }
      const candidateNoise = speechPresence < 0.42 ? this.psd[k] : Math.min(this.noisePsd[k], this.minTrack[k] * 1.6);
      this.noisePsd[k] = alphaNoise * this.noisePsd[k] + (1.0 - alphaNoise) * candidateNoise;
    }

    // Temporal Hangover Smoothing (fast attack for plosive onsets, smooth release)
    const smoothProb = rawProb >= this.prevProb
      ? 0.72 * rawProb + 0.28 * this.prevProb
      : 0.42 * rawProb + 0.58 * this.prevProb;

    this.prevProb = Math.max(0.0, Math.min(1.0, smoothProb));
    this.frameCount++;
    return { probability: this.prevProb, snrDb };
  }
}
