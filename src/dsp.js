/**
 * Zero-dependency 16 kHz Audio DSP Utilities:
 * - RIFF WAV 8/16/24/32-bit PCM & IEEE Float32 decoder + 16-bit PCM WAV encoder
 * - 4th-order cascaded Butterworth anti-aliased resampler (fc = 7,400 Hz, -24 dB/oct) -> 16,000 Hz
 * - Causal 80 Hz 2nd-order Butterworth HPF + 50 Hz & 60 Hz Q=22 mains hum biquad notches
 * - Complementary raised-cosine (Hann) overlap-add crossfader & adversarial seam jump verifier
 */

export const TARGET_SAMPLE_RATE = 16000;
export const FRAME_SAMPLES = 512; // 32 ms @ 16 kHz
export const CONTEXT_SAMPLES = 64; // 4 ms receptive context @ 16 kHz
export const WINDOW_SAMPLES = 576; // 64 + 512

export function designHighPassBiquad(fc, sr, q = Math.SQRT1_2) {
  const omega = (2.0 * Math.PI * fc) / sr;
  const sinW = Math.sin(omega);
  const cosW = Math.cos(omega);
  const alpha = sinW / (2.0 * q);
  const a0 = 1.0 + alpha;
  return {
    b0: ((1.0 + cosW) * 0.5) / a0,
    b1: -(1.0 + cosW) / a0,
    b2: ((1.0 + cosW) * 0.5) / a0,
    a1: (-2.0 * cosW) / a0,
    a2: (1.0 - alpha) / a0,
  };
}

export function designLowPassBiquad(fc, sr, q) {
  const omega = (2.0 * Math.PI * fc) / sr;
  const sinW = Math.sin(omega);
  const cosW = Math.cos(omega);
  const alpha = sinW / (2.0 * q);
  const a0 = 1.0 + alpha;
  return {
    b0: ((1.0 - cosW) * 0.5) / a0,
    b1: (1.0 - cosW) / a0,
    b2: ((1.0 - cosW) * 0.5) / a0,
    a1: (-2.0 * cosW) / a0,
    a2: (1.0 - alpha) / a0,
  };
}

export function designNotchBiquad(f0, sr, q = 22.0) {
  const omega = (2.0 * Math.PI * f0) / sr;
  const sinW = Math.sin(omega);
  const cosW = Math.cos(omega);
  const alpha = sinW / (2.0 * q);
  const a0 = 1.0 + alpha;
  return {
    b0: 1.0 / a0,
    b1: (-2.0 * cosW) / a0,
    b2: 1.0 / a0,
    a1: (-2.0 * cosW) / a0,
    a2: (1.0 - alpha) / a0,
  };
}

export class BiquadState {
  constructor(coeffs) {
    this.b0 = coeffs.b0;
    this.b1 = coeffs.b1;
    this.b2 = coeffs.b2;
    this.a1 = coeffs.a1;
    this.a2 = coeffs.a2;
    this.x1 = 0.0;
    this.x2 = 0.0;
    this.y1 = 0.0;
    this.y2 = 0.0;
  }

  reset() {
    this.x1 = 0.0;
    this.x2 = 0.0;
    this.y1 = 0.0;
    this.y2 = 0.0;
  }

  processSample(x) {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1;
    this.x1 = x;
    this.y2 = this.y1;
    this.y1 = y;
    return y;
  }

  processBufferInPlace(buf) {
    for (let i = 0; i < buf.length; i++) {
      buf[i] = this.processSample(buf[i]);
    }
    return buf;
  }
}

export class PreConditioner16k {
  constructor({ highPassHz = 80, humNotch = true } = {}) {
    this.hpf = highPassHz > 0 ? new BiquadState(designHighPassBiquad(highPassHz, TARGET_SAMPLE_RATE)) : null;
    this.n50 = humNotch ? new BiquadState(designNotchBiquad(50.0, TARGET_SAMPLE_RATE, 22.0)) : null;
    this.n60 = humNotch ? new BiquadState(designNotchBiquad(60.0, TARGET_SAMPLE_RATE, 22.0)) : null;
  }

  reset() {
    if (this.hpf) this.hpf.reset();
    if (this.n50) this.n50.reset();
    if (this.n60) this.n60.reset();
  }

  process(input) {
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      let s = input[i];
      if (this.hpf) s = this.hpf.processSample(s);
      if (this.n50) s = this.n50.processSample(s);
      if (this.n60) s = this.n60.processSample(s);
      out[i] = s > 1.0 ? 1.0 : s < -1.0 ? -1.0 : s;
    }
    return out;
  }
}

/**
 * 4th-order cascaded Butterworth anti-aliased resampler (fc = 7,400 Hz when downsampling) to 16,000 Hz.
 */
export function resample16k(pcm, inputSampleRate = TARGET_SAMPLE_RATE) {
  if (!(pcm instanceof Float32Array)) {
    pcm = Float32Array.from(pcm);
  }
  if (!inputSampleRate || inputSampleRate === TARGET_SAMPLE_RATE) {
    return pcm;
  }

  let filtered = pcm;
  if (inputSampleRate > TARGET_SAMPLE_RATE) {
    const fc = Math.min(7400.0, 0.45 * TARGET_SAMPLE_RATE);
    const s1 = new BiquadState(designLowPassBiquad(fc, inputSampleRate, 0.541196100146197));
    const s2 = new BiquadState(designLowPassBiquad(fc, inputSampleRate, 1.3065629648763766));
    filtered = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) {
      filtered[i] = s2.processSample(s1.processSample(pcm[i]));
    }
  }

  const ratio = inputSampleRate / TARGET_SAMPLE_RATE;
  const outLength = Math.max(1, Math.round(filtered.length / ratio));
  const out = new Float32Array(outLength);
  const maxIdx = filtered.length - 1;

  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const s0 = filtered[Math.min(idx, maxIdx)];
    const s1 = filtered[Math.min(idx + 1, maxIdx)];
    out[i] = (1.0 - frac) * s0 + frac * s1;
  }
  return out;
}

/**
 * Decodes RIFF WAV (8/16/24/32-bit integer PCM or 32-bit IEEE float, mono or multi-channel) into mono Float32Array.
 */
export function decodeWav(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (bytes.byteLength < 44) {
    throw new Error('Invalid WAV buffer: shorter than 44 bytes');
  }
  const riff = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  const wave = String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]);
  if (riff !== 'RIFF' || wave !== 'WAVE') {
    throw new Error(`Unsupported audio header "${riff}/${wave}" (expected RIFF/WAVE)`);
  }

  let audioFormat = 1;
  let numChannels = 1;
  let sampleRate = 16000;
  let bitsPerSample = 16;
  let dataOffset = -1;
  let dataSize = 0;

  let offset = 12;
  while (offset + 8 <= bytes.byteLength) {
    const chunkId = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;

    if (chunkId === 'fmt ' && chunkSize >= 16) {
      audioFormat = view.getUint16(chunkStart, true);
      numChannels = view.getUint16(chunkStart + 2, true);
      sampleRate = view.getUint32(chunkStart + 4, true);
      bitsPerSample = view.getUint16(chunkStart + 14, true);
      if (audioFormat === 0xfffe && chunkSize >= 40) {
        audioFormat = view.getUint16(chunkStart + 24, true);
      }
    } else if (chunkId === 'data') {
      dataOffset = chunkStart;
      dataSize = Math.min(chunkSize, bytes.byteLength - chunkStart);
      break;
    }
    offset = chunkStart + chunkSize + (chunkSize & 1);
  }

  if (dataOffset < 0) {
    throw new Error('Invalid WAV buffer: missing data chunk');
  }

  const bytesPerSample = bitsPerSample >> 3;
  const frameStride = bytesPerSample * numChannels;
  const totalFrames = Math.floor(dataSize / frameStride);
  const pcm = new Float32Array(totalFrames);

  for (let i = 0; i < totalFrames; i++) {
    let acc = 0.0;
    const base = dataOffset + i * frameStride;
    for (let ch = 0; ch < numChannels; ch++) {
      const p = base + ch * bytesPerSample;
      if (audioFormat === 3 && bitsPerSample === 32) {
        acc += view.getFloat32(p, true);
      } else if (bitsPerSample === 16) {
        acc += view.getInt16(p, true) / 32768.0;
      } else if (bitsPerSample === 24) {
        const b0 = bytes[p];
        const b1 = bytes[p + 1];
        const b2 = bytes[p + 2];
        let v = b0 | (b1 << 8) | (b2 << 16);
        if (v & 0x800000) v |= ~0xffffff;
        acc += v / 8388608.0;
      } else if (bitsPerSample === 32) {
        acc += view.getInt32(p, true) / 2147483648.0;
      } else if (bitsPerSample === 8) {
        acc += (bytes[p] - 128) / 128.0;
      } else {
        throw new Error(`Unsupported WAV format=${audioFormat} bitsPerSample=${bitsPerSample}`);
      }
    }
    pcm[i] = acc / numChannels;
  }

  return { pcm, sampleRate, numChannels };
}

/**
 * Encodes mono Float32Array PCM into a standard 16-bit RIFF WAV ArrayBuffer.
 */
export function encodeWav(pcm, sampleRate = TARGET_SAMPLE_RATE) {
  const numSamples = pcm.length;
  const buf = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buf);

  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let off = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1.0, Math.min(1.0, pcm[i]));
    view.setInt16(off, s < 0 ? Math.round(s * 32768) : Math.round(s * 32767), true);
    off += 2;
  }
  return buf;
}

/**
 * Concatenates speech segments using complementary raised-cosine (Hann) ramps
 * (`fade_in[i] + fade_out[i] === 1.0`) and 64-sample outer edge tapers so zero boundary clicks occur.
 */
export function concatSegmentsHann(segments, overlapSamples = 128, edgeTaperSamples = 64) {
  const valid = segments.filter((s) => s && s.length > 0);
  if (valid.length === 0) return { pcm: new Float32Array(0), seamIndices: [] };

  if (valid.length === 1) {
    const out = new Float32Array(valid[0]);
    applyEdgeTaper(out, edgeTaperSamples);
    return { pcm: out, seamIndices: [] };
  }

  let totalLen = valid[0].length;
  for (let k = 1; k < valid.length; k++) {
    const ov = Math.min(overlapSamples, Math.floor(valid[k - 1].length / 3), Math.floor(valid[k].length / 3));
    totalLen += valid[k].length - ov;
  }

  const out = new Float32Array(totalLen);
  out.set(valid[0], 0);
  let writePos = valid[0].length;
  const seamIndices = [];

  for (let k = 1; k < valid.length; k++) {
    const curr = valid[k];
    const ov = Math.min(overlapSamples, Math.floor(writePos / 3), Math.floor(curr.length / 3));
    const seamStart = writePos - ov;
    seamIndices.push(seamStart);

    for (let i = 0; i < ov; i++) {
      const w = 0.5 * (1.0 - Math.cos((Math.PI * i) / ov));
      out[seamStart + i] = out[seamStart + i] * (1.0 - w) + curr[i] * w;
    }
    out.set(curr.subarray(ov), writePos);
    writePos += curr.length - ov;
  }

  applyEdgeTaper(out, edgeTaperSamples);
  return { pcm: out.subarray(0, writePos), seamIndices };
}

function applyEdgeTaper(buf, taperSamples) {
  const n = Math.min(taperSamples, Math.floor(buf.length / 4));
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1.0 - Math.cos((Math.PI * i) / n));
    buf[i] *= w;
    buf[buf.length - 1 - i] *= w;
  }
}

/**
 * Adversarial Seam Continuity Metric (Section 5 of /browser-streaming-onnx):
 * Compares max first-difference jump across seam regions against the 99.9th percentile interior sample jump.
 */
export function verifySeamContinuity(pcm, seamIndices, windowRadius = 4) {
  if (!pcm || pcm.length < 16 || !seamIndices || seamIndices.length === 0) {
    return { ok: true, maxSeamJump: 0, p999InteriorJump: 0, ratio: 0 };
  }

  const seamMask = new Uint8Array(pcm.length);
  for (const idx of seamIndices) {
    for (let j = Math.max(1, idx - windowRadius); j <= Math.min(pcm.length - 1, idx + windowRadius); j++) {
      seamMask[j] = 1;
    }
  }

  const interiorDiffs = [];
  let maxSeamJump = 0.0;
  for (let i = 1; i < pcm.length; i++) {
    const d = Math.abs(pcm[i] - pcm[i - 1]);
    if (seamMask[i]) {
      if (d > maxSeamJump) maxSeamJump = d;
    } else {
      interiorDiffs.push(d);
    }
  }

  if (interiorDiffs.length === 0) {
    return { ok: true, maxSeamJump, p999InteriorJump: maxSeamJump, ratio: 1.0 };
  }
  interiorDiffs.sort((a, b) => a - b);
  const p999Idx = Math.min(interiorDiffs.length - 1, Math.floor(interiorDiffs.length * 0.999));
  const p999 = Math.max(interiorDiffs[p999Idx], 1e-6);
  const ratio = maxSeamJump / p999;

  return {
    ok: ratio <= 1.5,
    maxSeamJump,
    p999InteriorJump: p999,
    ratio,
  };
}

export function computeRms(frame) {
  if (!frame || frame.length === 0) return 0.0;
  let sumSq = 0.0;
  for (let i = 0; i < frame.length; i++) {
    sumSq += frame[i] * frame[i];
  }
  return Math.sqrt(sumSq / frame.length);
}
