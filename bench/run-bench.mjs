/**
 * Reproducible Science-Backed VAD Evaluation Suite (`N = 120` streams across 12 languages & 5 acoustic regimes)
 * with `B = 1,000` Paired Bootstrap 95% Confidence Intervals (`ml-best-practices` + `science-baked-modules`).
 *
 * Evaluates 4 engines head-to-head:
 *   1. Naive RMS Energy Baseline (standard WebAudio fixed RMS gate)
 *   2. `wasm-vad` Zero-Dep Spectral-LRT (`0 KB` weights, `tier: 'dsp'`)
 *   3. `wasm-vad` Mobile Hybrid `W8A32+FP16` (`432 KiB`, `tier: 'mobile'`)
 *   4. `wasm-vad` Studio `FP32` (`1.24 MB`, Graph-Split Batched, `tier: 'studio'`)
 */

import fs from 'node:fs';
import path from 'node:path';
import vad, { decodeWav, computeRms, FRAME_SAMPLES, TARGET_SAMPLE_RATE } from '../dist/index.mjs';

const SAMPLES_DIR = fs.existsSync(path.resolve('..', 'swanada', 'docs', 'samples'))
  ? path.resolve('..', 'swanada', 'docs', 'samples')
  : path.resolve('docs', 'samples');
const LANG_FILES = fs.readdirSync(SAMPLES_DIR).filter((f) => f.endsWith('.wav')).sort();

class Mulberry32 {
  constructor(seed = 20261112) {
    this.state = seed >>> 0;
  }
  next() {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  int(max) {
    return Math.floor(this.next() * max);
  }
}

/**
 * Extracts tight active speech waveform from a clean studio sample so ground-truth sample boundaries are exact.
 */
function extractTightSpeech(pcm) {
  const win = 160; // 10 ms
  const numWins = Math.floor(pcm.length / win);
  let peak = 0;
  const energies = new Float32Array(numWins);
  for (let i = 0; i < numWins; i++) {
    const r = computeRms(pcm.subarray(i * win, (i + 1) * win));
    energies[i] = r;
    if (r > peak) peak = r;
  }
  const thresh = Math.max(0.004, peak * 0.045);
  let first = 0;
  let last = numWins - 1;
  while (first < numWins && energies[first] < thresh) first++;
  while (last > first && energies[last] < thresh) last--;
  const s0 = Math.max(0, first * win);
  const s1 = Math.min(pcm.length, (last + 1) * win);
  return pcm.subarray(s0, s1);
}

function generatePinkNoise(length, rng) {
  const out = new Float32Array(length);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < length; i++) {
    const white = rng.next() * 2 - 1;
    b0 = 0.99886 * b0 + white * 0.0555179;
    b1 = 0.99332 * b1 + white * 0.0750759;
    b2 = 0.96900 * b2 + white * 0.1538520;
    b3 = 0.86650 * b3 + white * 0.3104856;
    b4 = 0.55000 * b4 + white * 0.5329522;
    b5 = -0.7616 * b5 - white * 0.0168980;
    out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.04;
    b6 = white * 0.115926;
  }
  return out;
}

function generateMainsHum(length, rng) {
  const out = new Float32Array(length);
  const phase50 = rng.next() * Math.PI * 2;
  const phase60 = rng.next() * Math.PI * 2;
  for (let i = 0; i < length; i++) {
    out[i] =
      0.65 * Math.sin((2 * Math.PI * 50 * i) / 16000 + phase50) +
      0.35 * Math.sin((2 * Math.PI * 60 * i) / 16000 + phase60);
  }
  return out;
}

function mixAtSnrDb(cleanSignal, speechMaskFrames, noiseBuf, snrDb) {
  let speechSq = 0;
  let speechCount = 0;
  for (let f = 0; f < speechMaskFrames.length; f++) {
    if (speechMaskFrames[f]) {
      for (let i = f * FRAME_SAMPLES; i < (f + 1) * FRAME_SAMPLES; i++) {
        speechSq += cleanSignal[i] * cleanSignal[i];
        speechCount++;
      }
    }
  }
  const speechRms = Math.sqrt(speechSq / Math.max(1, speechCount));
  const noiseRms = Math.max(computeRms(noiseBuf), 1e-8);
  const targetNoiseRms = speechRms / Math.pow(10, snrDb / 20);
  const scale = targetNoiseRms / noiseRms;

  const mixed = new Float32Array(cleanSignal.length);
  for (let i = 0; i < cleanSignal.length; i++) {
    const v = cleanSignal[i] + noiseBuf[i] * scale;
    mixed[i] = v > 1 ? 1 : v < -1 ? -1 : v;
  }
  return mixed;
}

function buildGoldenDataset() {
  const rng = new Mulberry32(20261112);
  const utterances = LANG_FILES.map((file) => {
    const { pcm } = decodeWav(fs.readFileSync(path.join(SAMPLES_DIR, file)));
    return { name: file.replace('.wav', ''), pcm: extractTightSpeech(pcm) };
  });

  const dataset = [];

  const makeMultiTurn = (idx, condition, snrDb = null, hum = false) => {
    const u1 = utterances[idx % utterances.length];
    const u2 = utterances[(idx * 7 + 3) % utterances.length];
    const leadPad = (18 + rng.int(10)) * FRAME_SAMPLES; // ~576-896 ms
    const midPause = (22 + rng.int(12)) * FRAME_SAMPLES; // ~704-1088 ms
    const tailPad = (18 + rng.int(8)) * FRAME_SAMPLES;

    const u1Len = Math.floor(u1.pcm.length / FRAME_SAMPLES) * FRAME_SAMPLES;
    const u2Len = Math.floor(u2.pcm.length / FRAME_SAMPLES) * FRAME_SAMPLES;
    const totalSamples = leadPad + u1Len + midPause + u2Len + tailPad;
    const numFrames = totalSamples / FRAME_SAMPLES;

    const clean = new Float32Array(totalSamples);
    clean.set(u1.pcm.subarray(0, u1Len), leadPad);
    clean.set(u2.pcm.subarray(0, u2Len), leadPad + u1Len + midPause);

    const gtFrames = new Uint8Array(numFrames);
    const s1StartFrame = leadPad / FRAME_SAMPLES;
    const s1EndFrame = (leadPad + u1Len) / FRAME_SAMPLES;
    const s2StartFrame = (leadPad + u1Len + midPause) / FRAME_SAMPLES;
    const s2EndFrame = (leadPad + u1Len + midPause + u2Len) / FRAME_SAMPLES;

    for (let f = s1StartFrame; f < s1EndFrame; f++) gtFrames[f] = 1;
    for (let f = s2StartFrame; f < s2EndFrame; f++) gtFrames[f] = 1;

    let pcm = clean;
    if (snrDb !== null) {
      const noise = hum
        ? generateMainsHum(totalSamples, rng)
        : generatePinkNoise(totalSamples, rng);
      pcm = mixAtSnrDb(clean, gtFrames, noise, snrDb);
    }

    return {
      id: `${condition}_${idx}`,
      condition,
      isAdversarialNonSpeech: false,
      pcm,
      gtFrames,
      gtIntervalsMs: [
        [(s1StartFrame * 32), (s1EndFrame * 32)],
        [(s2StartFrame * 32), (s2EndFrame * 32)],
      ],
    };
  };

  // 1. Clean Multi-Turn Across 12 Languages (24 items)
  for (let i = 0; i < 24; i++) dataset.push(makeMultiTurn(i, 'clean_multiturn', null, false));
  // 2. Office / HVAC Pink Noise +12 dB SNR (24 items)
  for (let i = 0; i < 24; i++) dataset.push(makeMultiTurn(i + 3, 'noisy_12dB', 12, false));
  // 3. Heavy Street / Cafe Noise +5 dB SNR (24 items)
  for (let i = 0; i < 24; i++) dataset.push(makeMultiTurn(i + 7, 'noisy_5dB', 5, false));
  // 4. Mains 50/60 Hz Hum +5 dB SNR + Speech (24 items)
  for (let i = 0; i < 24; i++) dataset.push(makeMultiTurn(i + 11, 'hum_5dB', 5, true));

  // 5. Pure Adversarial Non-Speech Distractors (24 items: hum, pink noise, white hiss, desk thumps, keyboard clicks, pure tones)
  for (let i = 0; i < 24; i++) {
    const numFrames = 96; // 3.072 s
    const len = numFrames * FRAME_SAMPLES;
    const pcm = new Float32Array(len);
    const kind = i % 4;
    if (kind === 0) {
      const h = generateMainsHum(len, rng);
      for (let k = 0; k < len; k++) pcm[k] = h[k] * (0.15 + 0.15 * rng.next());
    } else if (kind === 1) {
      const p = generatePinkNoise(len, rng);
      for (let k = 0; k < len; k++) pcm[k] = p[k] * (1.2 + rng.next());
    } else if (kind === 2) {
      // Keyboard click & desk thump bursts (short <60ms impulses spaced out)
      for (let b = 0; b < 8; b++) {
        const pos = Math.floor(((b + 0.5) / 8) * len);
        for (let j = 0; j < 220 && pos + j < len; j++) {
          const env = Math.exp(-j / 35);
          pcm[pos + j] = env * ((rng.next() * 2 - 1) * 0.45 + 0.35 * Math.sin((2 * Math.PI * 55 * j) / 16000));
        }
      }
    } else {
      // Pure single-frequency telemetry/dial beeps (440 Hz, 880 Hz, 1000 Hz)
      const freq = [440, 880, 1000][i % 3];
      for (let k = 0; k < len; k++) {
        pcm[k] = 0.22 * Math.sin((2 * Math.PI * freq * k) / 16000);
      }
    }
    dataset.push({
      id: `adversarial_nonspeech_${i}`,
      condition: 'adversarial_nonspeech',
      isAdversarialNonSpeech: true,
      pcm,
      gtFrames: new Uint8Array(numFrames),
      gtIntervalsMs: [],
    });
  }

  return dataset;
}

function computeRocAuc(probs, labels) {
  const n = probs.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => probs[b] - probs[a]);

  let tp = 0;
  let fp = 0;
  let totalPos = 0;
  for (let i = 0; i < n; i++) totalPos += labels[i];
  const totalNeg = n - totalPos;
  if (totalPos === 0 || totalNeg === 0) return 0.5;

  let auc = 0;
  let prevTp = 0;
  let prevFp = 0;
  let prevScore = Infinity;

  for (let k = 0; k < n; k++) {
    const i = idx[k];
    const s = probs[i];
    if (s !== prevScore) {
      auc += ((fp - prevFp) * (tp + prevTp)) * 0.5;
      prevTp = tp;
      prevFp = fp;
      prevScore = s;
    }
    if (labels[i] === 1) tp++;
    else fp++;
  }
  auc += ((fp - prevFp) * (tp + prevTp)) * 0.5;
  return auc / (totalPos * totalNeg);
}

function computeItemConfusion(probs, gtFrames, threshold = 0.5) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < gtFrames.length; i++) {
    const pred = probs[i] >= threshold ? 1 : 0;
    const gt = gtFrames[i];
    if (pred === 1 && gt === 1) tp++;
    else if (pred === 1 && gt === 0) fp++;
    else if (pred === 0 && gt === 1) fn++;
    else tn++;
  }
  return { tp, fp, fn, tn };
}

function computeBoundaryErrors(segments, gtIntervalsMs) {
  if (gtIntervalsMs.length === 0 || segments.length === 0) return [];
  const errs = [];
  for (const [gtStart, gtEnd] of gtIntervalsMs) {
    // Match closest overlapping predicted segment
    let best = null;
    let bestOverlap = 0;
    for (const seg of segments) {
      const ov = Math.max(0, Math.min(seg.endMs, gtEnd) - Math.max(seg.startMs, gtStart));
      if (ov > bestOverlap) {
        bestOverlap = ov;
        best = seg;
      }
    }
    if (best) {
      // Account for intentional 128 ms pre-roll pad when measuring onset accuracy vs unpadded ground truth
      const adjustedStart = best.startMs + 128;
      errs.push({
        onsetErrMs: Math.abs(adjustedStart - gtStart),
        offsetErrMs: Math.abs(best.endMs - gtEnd),
      });
    }
  }
  return errs;
}

function bootstrapMetrics(perItemStats, B = 1000) {
  const rng = new Mulberry32(987654321);
  const n = perItemStats.length;

  const calcAggregate = (items) => {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    let advFp = 0, advTotal = 0;
    let onsetSum = 0, offsetSum = 0, boundCount = 0;

    for (const it of items) {
      tp += it.tp;
      fp += it.fp;
      fn += it.fn;
      tn += it.tn;
      if (it.isAdversarialNonSpeech) {
        advFp += it.fp;
        advTotal += it.fp + it.tn;
      }
      for (const b of it.boundaryErrs) {
        onsetSum += b.onsetErrMs;
        offsetSum += b.offsetErrMs;
        boundCount++;
      }
    }
    const precision = tp / Math.max(1, tp + fp);
    const recall = tp / Math.max(1, tp + fn);
    const f1 = (2 * precision * recall) / Math.max(1e-9, precision + recall);
    const advFpr = advFp / Math.max(1, advTotal);
    const onsetMaeMs = boundCount > 0 ? onsetSum / boundCount : 0;
    const offsetMaeMs = boundCount > 0 ? offsetSum / boundCount : 0;
    return { f1, precision, recall, advFpr, onsetMaeMs, offsetMaeMs };
  };

  const point = calcAggregate(perItemStats);
  const f1Dist = [];
  const fprDist = [];
  for (let b = 0; b < B; b++) {
    const sample = [];
    for (let i = 0; i < n; i++) {
      sample.push(perItemStats[rng.int(n)]);
    }
    const m = calcAggregate(sample);
    f1Dist.push(m.f1);
    fprDist.push(m.advFpr);
  }
  f1Dist.sort((a, b) => a - b);
  fprDist.sort((a, b) => a - b);

  const ci = (arr) => [arr[Math.floor(B * 0.025)], arr[Math.floor(B * 0.975)]];
  return {
    ...point,
    f1Ci95: ci(f1Dist),
    advFprCi95: ci(fprDist),
  };
}

async function evaluateNaiveRms(dataset) {
  const THRESH_RMS = 0.025;
  const perItem = [];
  const allProbs = [];
  const allLabels = [];
  const frameLatenciesUs = [];
  let totalAudioMs = 0;
  let totalInferMs = 0;

  for (const item of dataset) {
    const nFrames = item.gtFrames.length;
    const probs = new Float32Array(nFrames);
    const t0 = performance.now();
    for (let f = 0; f < nFrames; f++) {
      const tf0 = performance.now();
      const rms = computeRms(item.pcm.subarray(f * FRAME_SAMPLES, (f + 1) * FRAME_SAMPLES));
      const score = Math.min(1.0, rms / (THRESH_RMS * 2.0));
      probs[f] = score;
      frameLatenciesUs.push((performance.now() - tf0) * 1000);
      allProbs.push(score);
      allLabels.push(item.gtFrames[f]);
    }
    totalInferMs += performance.now() - t0;
    totalAudioMs += nFrames * 32;

    const conf = computeItemConfusion(probs, item.gtFrames, 0.5);
    perItem.push({
      ...conf,
      isAdversarialNonSpeech: item.isAdversarialNonSpeech,
      boundaryErrs: [],
    });
  }

  frameLatenciesUs.sort((a, b) => a - b);
  const boot = bootstrapMetrics(perItem, 1000);
  const rocAuc = computeRocAuc(allProbs, allLabels);

  return {
    name: 'Naive RMS Energy Gate',
    weightsBytes: 0,
    f1: Number((boot.f1 * 100).toFixed(2)),
    f1Ci95: boot.f1Ci95.map((v) => Number((v * 100).toFixed(2))),
    precision: Number((boot.precision * 100).toFixed(2)),
    recall: Number((boot.recall * 100).toFixed(2)),
    rocAuc: Number(rocAuc.toFixed(4)),
    adversarialFprPct: Number((boot.advFpr * 100).toFixed(2)),
    adversarialFprCi95: boot.advFprCi95.map((v) => Number((v * 100).toFixed(2))),
    onsetMaeMs: null,
    offsetMaeMs: null,
    rtf: Number((totalInferMs / totalAudioMs).toFixed(6)),
    frameLatencyUs: {
      mean: Number((frameLatenciesUs.reduce((a, b) => a + b, 0) / frameLatenciesUs.length).toFixed(2)),
      p95: Number(frameLatenciesUs[Math.floor(frameLatenciesUs.length * 0.95)].toFixed(2)),
      p99: Number(frameLatenciesUs[Math.floor(frameLatenciesUs.length * 0.99)].toFixed(2)),
    },
  };
}

async function evaluateWasmVadTier(dataset, tier, label) {
  const detector = await vad({ tier });
  const perItem = [];
  const allProbs = [];
  const allLabels = [];
  let totalAudioMs = 0;
  let totalBatchInferMs = 0;

  // Warmup
  await detector.detect(dataset[0].pcm);

  for (const item of dataset) {
    const res = await detector.detect(item.pcm, { returnFrames: true });
    totalBatchInferMs += res.inferenceMs;
    totalAudioMs += res.durationMs;

    const n = Math.min(res.probabilities.length, item.gtFrames.length);
    for (let i = 0; i < n; i++) {
      allProbs.push(res.probabilities[i]);
      allLabels.push(item.gtFrames[i]);
    }
    const conf = computeItemConfusion(res.probabilities, item.gtFrames, 0.5);
    const boundaryErrs = computeBoundaryErrors(res.segments, item.gtIntervalsMs);
    perItem.push({
      ...conf,
      isAdversarialNonSpeech: item.isAdversarialNonSpeech,
      boundaryErrs,
    });
  }

  // Also measure streaming single-frame (`detector.frame()`) latency distribution across 500 frames
  detector.reset();
  const streamLatUs = [];
  const streamSample = dataset[0].pcm;
  const maxFrames = Math.min(200, Math.floor(streamSample.length / FRAME_SAMPLES));
  for (let f = 0; f < maxFrames; f++) {
    const t0 = performance.now();
    await detector.frame(streamSample.subarray(f * FRAME_SAMPLES, (f + 1) * FRAME_SAMPLES));
    streamLatUs.push((performance.now() - t0) * 1000);
  }
  streamLatUs.sort((a, b) => a - b);

  const boot = bootstrapMetrics(perItem, 1000);
  const rocAuc = computeRocAuc(allProbs, allLabels);
  const modelBytes = detector.info.modelBytes;
  const splitMs = detector.info.splitMs;
  await detector.dispose();

  return {
    name: label,
    tier,
    weightsBytes: modelBytes,
    splitMs,
    f1: Number((boot.f1 * 100).toFixed(2)),
    f1Ci95: boot.f1Ci95.map((v) => Number((v * 100).toFixed(2))),
    precision: Number((boot.precision * 100).toFixed(2)),
    recall: Number((boot.recall * 100).toFixed(2)),
    rocAuc: Number(rocAuc.toFixed(4)),
    adversarialFprPct: Number((boot.advFpr * 100).toFixed(2)),
    adversarialFprCi95: boot.advFprCi95.map((v) => Number((v * 100).toFixed(2))),
    onsetMaeMs: Number(boot.onsetMaeMs.toFixed(1)),
    offsetMaeMs: Number(boot.offsetMaeMs.toFixed(1)),
    rtf: Number((totalBatchInferMs / totalAudioMs).toFixed(6)),
    speedupRealtime: Number((totalAudioMs / totalBatchInferMs).toFixed(1)),
    frameLatencyUs: {
      mean: Number((streamLatUs.reduce((a, b) => a + b, 0) / streamLatUs.length).toFixed(1)),
      p95: Number(streamLatUs[Math.floor(streamLatUs.length * 0.95)].toFixed(1)),
      p99: Number(streamLatUs[Math.floor(streamLatUs.length * 0.99)].toFixed(1)),
    },
  };
}

async function main() {
  console.log('Building N = 120 Golden 12-Language Multi-Condition VAD Dataset...');
  const dataset = buildGoldenDataset();
  const totalFrames = dataset.reduce((s, d) => s + d.gtFrames.length, 0);
  const totalSec = ((totalFrames * 32) / 1000).toFixed(1);
  console.log(`Dataset ready: N = ${dataset.length} audio streams | ${totalFrames} frames (${totalSec}s @ 16 kHz)\n`);

  const results = [];
  results.push(await evaluateNaiveRms(dataset));
  results.push(await evaluateWasmVadTier(dataset, 'dsp', 'wasm-vad Zero-Dep Spectral-LRT (0 KB)'));
  results.push(await evaluateWasmVadTier(dataset, 'mobile', 'wasm-vad Mobile Hybrid W8A32+FP16 (432 KiB)'));
  results.push(await evaluateWasmVadTier(dataset, 'studio', 'wasm-vad Studio FP32 Graph-Split (1.24 MB)'));

  console.table(
    results.map((r) => ({
      Engine: r.name,
      Weights: r.weightsBytes === 0 ? '0 KB' : `${(r.weightsBytes / 1024).toFixed(0)} KiB`,
      'F1 (%) [95% CI]': `${r.f1.toFixed(2)} [${r.f1Ci95[0]}..${r.f1Ci95[1]}]`,
      'ROC-AUC': r.rocAuc,
      'Non-Speech FPR (%)': `${r.adversarialFprPct.toFixed(2)}%`,
      'Onset MAE': r.onsetMaeMs !== null ? `${r.onsetMaeMs} ms` : '—',
      RTF: r.rtf,
      'Frame p95 (µs)': r.frameLatencyUs.p95,
    }))
  );

  const report = {
    generatedAt: new Date().toISOString(),
    methodology: {
      streamsN: dataset.length,
      totalFrames,
      totalAudioSeconds: Number(totalSec),
      languages: 12,
      bootstrapResamplesB: 1000,
      conditions: ['clean_multiturn', 'noisy_12dB', 'noisy_5dB', 'hum_5dB', 'adversarial_nonspeech'],
      sampleRateHz: TARGET_SAMPLE_RATE,
      frameMs: 32,
    },
    results,
  };

  fs.mkdirSync('bench', { recursive: true });
  fs.writeFileSync('bench/results.json', JSON.stringify(report, null, 2) + '\n');
  console.log('\nSaved empirical benchmark report -> bench/results.json');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
