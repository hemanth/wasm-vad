/**
 * wasm-vad — Capability-routed streaming 16 kHz Voice Activity Detection for Browsers & Node.js.
 *
 * Distills lessons from `ninada` and `swanada`:
 *   1. Capability-routed checkpoints: `vad-studio.onnx` (1.24 MB FP32) on desktop/server vs
 *      `vad-mobile.onnx` (432 KiB Hybrid W8A32+FP16, 1T WASM) on mobile vs Zero-Dep Spectral-LRT (`0 KB`).
 *   2. Branch-free ONNX (0 `If` subgraphs, 0 `LSTM` ops, `Gather` reflect pad) + `<1 ms` client-side
 *      protobuf graph splitting (`[B, 576] -> [B, 128]` batched encoder + lightweight recurrent scan).
 *   3. Causal 80 Hz HPF + 50/60 Hz Q=22 hum conditioning, 4th-order Butterworth anti-aliased 16 kHz
 *      resampling, 128 ms pre-roll ring buffer, 320 ms Indic geminate stop hangover, and Hann seam crossfades.
 */

import { detectCapabilities } from './capabilities.js';
import {
  TARGET_SAMPLE_RATE,
  FRAME_SAMPLES,
  CONTEXT_SAMPLES,
  WINDOW_SAMPLES,
  PreConditioner16k,
  resample16k,
  decodeWav,
  encodeWav,
  concatSegmentsHann,
  verifySeamContinuity,
  computeRms,
} from './dsp.js';
import { splitOnnxGraph } from './onnx-splitter.js';
import { fetchAndCacheModel } from './cache.js';
import { SpectralVadEngine } from './spectral-vad.js';
import { VadHysteresisArbiter, buildContextWindows, DEFAULT_VAD_OPTIONS } from './hysteresis.js';
import { startAudioWorkletCapture } from './worklet.js';

const CDN_MODEL_BASE = 'https://cdn.jsdelivr.net/npm/wasm-vad@0.1.1/models/';

function isAudioInput(val) {
  if (!val) return false;
  if (typeof val === 'string') return true;
  if (val instanceof Float32Array || val instanceof Int16Array || val instanceof Uint8Array || val instanceof ArrayBuffer) {
    return true;
  }
  if (typeof val === 'object') {
    if (typeof val.getChannelData === 'function') return true; // Web AudioBuffer
    if (val.pcm instanceof Float32Array || val.pcm instanceof Int16Array) return true;
  }
  return false;
}

function int16ToFloat32(int16) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    out[i] = int16[i] / 32768.0;
  }
  return out;
}

function hasRiffHeader(bytes) {
  return (
    bytes.byteLength >= 12 &&
    bytes[0] === 0x52 && // R
    bytes[1] === 0x49 && // I
    bytes[2] === 0x46 && // F
    bytes[3] === 0x46    // F
  );
}

async function normalizeAudioTo16k(input, sampleRateHint = TARGET_SAMPLE_RATE) {
  const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node) && typeof window === 'undefined';

  if (typeof input === 'string') {
    let bytes;
    if (isNode && !/^https?:\/\//i.test(input)) {
      const fs = await import('node:fs/promises');
      bytes = new Uint8Array(await fs.readFile(input));
    } else {
      const res = await fetch(input);
      if (!res.ok) throw new Error(`Failed to fetch audio "${input}": ${res.status}`);
      bytes = new Uint8Array(await res.arrayBuffer());
    }
    const decoded = decodeWav(bytes);
    return resample16k(decoded.pcm, decoded.sampleRate);
  }

  if (input instanceof ArrayBuffer || input instanceof Uint8Array) {
    const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
    if (hasRiffHeader(bytes)) {
      const decoded = decodeWav(bytes);
      return resample16k(decoded.pcm, decoded.sampleRate);
    }
    const f32 = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
    return resample16k(f32, sampleRateHint);
  }

  if (input instanceof Float32Array) {
    return resample16k(input, sampleRateHint);
  }

  if (input instanceof Int16Array) {
    return resample16k(int16ToFloat32(input), sampleRateHint);
  }

  if (input && typeof input.getChannelData === 'function') {
    const numCh = input.numberOfChannels || 1;
    const len = input.length;
    const mono = new Float32Array(len);
    for (let c = 0; c < numCh; c++) {
      const ch = input.getChannelData(c);
      for (let i = 0; i < len; i++) mono[i] += ch[i] / numCh;
    }
    return resample16k(mono, input.sampleRate || TARGET_SAMPLE_RATE);
  }

  if (input && typeof input === 'object' && input.pcm) {
    const raw = input.pcm instanceof Int16Array ? int16ToFloat32(input.pcm) : input.pcm;
    return resample16k(raw, input.sampleRate || sampleRateHint);
  }

  throw new TypeError(
    'Unsupported audio input: expected WAV path/URL, ArrayBuffer, Uint8Array, Float32Array, Int16Array, AudioBuffer, or { pcm, sampleRate }'
  );
}

async function loadOrtRuntime(caps, userOrt, wasmPaths) {
  if (caps.backend === 'dsp') return { ort: null, caps };
  const isNodeEnv = typeof process !== 'undefined' && Boolean(process.versions?.node) && typeof window === 'undefined';

  if (userOrt) {
    if (isNodeEnv && userOrt?.env?.wasm) {
      userOrt.env.wasm.numThreads = 1;
      userOrt.env.wasm.simd = true;
    }
    return { ort: userOrt, caps };
  }
  if (typeof globalThis.ort !== 'undefined' && globalThis.ort?.InferenceSession) {
    return { ort: globalThis.ort, caps };
  }

  if (isNodeEnv) {
    if (caps.backend !== 'wasm') {
      try {
        const mod = await import('onnxruntime-node');
        return { ort: mod.default || mod, caps };
      } catch {
        // Fall through to bundled `onnxruntime-web` WASM SIMD engine
      }
    }
    try {
      const webMod = await import('onnxruntime-web');
      const ort = webMod.default || webMod;
      if (ort?.env?.wasm) {
        // Keep numThreads=1 in Node so Emscripten pthread workers never hang CLI / serverless exits
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;
      }
      const updatedCaps = {
        ...caps,
        backend: 'wasm',
        executionProviders: ['wasm'],
        numThreads: 1,
        sessionOptions: {
          executionProviders: ['wasm'],
          enableCpuMemArena: false,
          enableMemPattern: false,
          graphOptimizationLevel: 'all',
        },
      };
      return { ort, caps: updatedCaps };
    } catch {
      return { ort: null, caps };
    }
  } else {
    try {
      const webMod = await import('onnxruntime-web');
      const ort = webMod.default || webMod;
      if (ort?.env?.wasm) {
        ort.env.wasm.numThreads = caps.numThreads;
        ort.env.wasm.simd = true;
        if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
      }
      return { ort, caps };
    } catch {
      return { ort: null, caps };
    }
  }
}

async function loadModelBytes(caps, options) {
  if (options.modelBytes) {
    return options.modelBytes instanceof Uint8Array ? options.modelBytes : new Uint8Array(options.modelBytes);
  }

  const modelFile = caps.tier === 'mobile' ? 'vad-mobile.onnx' : 'vad-studio.onnx';
  const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node) && typeof window === 'undefined';

  if (isNode && (!options.modelUrl || !/^https?:\/\//i.test(options.modelUrl))) {
    const fs = await import('node:fs/promises');
    const path = await import('node:path');
    const urlMod = await import('node:url');

    if (options.modelUrl) {
      return new Uint8Array(await fs.readFile(options.modelUrl));
    }

    const candidates = [];
    if (typeof __dirname === 'string') {
      candidates.push(path.resolve(__dirname, '..', 'models', modelFile));
      candidates.push(path.resolve(__dirname, 'models', modelFile));
    }
    try {
      if (import.meta && import.meta.url) {
        const here = path.dirname(urlMod.fileURLToPath(import.meta.url));
        candidates.push(path.resolve(here, '..', 'models', modelFile));
        candidates.push(path.resolve(here, 'models', modelFile));
      }
    } catch {
      // CJS fallback handled above
    }
    candidates.push(path.resolve(process.cwd(), 'models', modelFile));

    for (const candidate of candidates) {
      try {
        return new Uint8Array(await fs.readFile(candidate));
      } catch {
        // Try next candidate
      }
    }
    throw new Error(`Could not locate bundled ONNX checkpoint "${modelFile}" in [${candidates.join(', ')}]`);
  }

  let url = options.modelUrl;
  if (!url) {
    try {
      if (import.meta && import.meta.url && /^https?:\/\//i.test(import.meta.url)) {
        url = new URL(`../models/${modelFile}`, import.meta.url).href;
      }
    } catch {
      // Ignore and use CDN fallback
    }
    if (!url) {
      url = `${CDN_MODEL_BASE}${modelFile}`;
    }
  }

  try {
    return await fetchAndCacheModel(url, {
      onProgress: options.onProgress,
      cache: options.cache !== false,
    });
  } catch (primaryErr) {
    if (!options.modelUrl && url !== `${CDN_MODEL_BASE}${modelFile}`) {
      return await fetchAndCacheModel(`${CDN_MODEL_BASE}${modelFile}`, {
        onProgress: options.onProgress,
        cache: options.cache !== false,
      });
    }
    throw primaryErr;
  }
}

class OnnxSplitBackend {
  constructor(ort, encoderSession, recurrentSession, splitStats) {
    this.ort = ort;
    this.encoderSession = encoderSession;
    this.recurrentSession = recurrentSession;
    this.splitStats = splitStats;
    this.conditioner = new PreConditioner16k({ highPassHz: 80, humNotch: true });
    this.spectralSnr = new SpectralVadEngine();
    this.context = new Float32Array(CONTEXT_SAMPLES);
    this.state = new Float32Array(2 * 1 * 128);
    this.windowBuf = new Float32Array(WINDOW_SAMPLES);
  }

  reset() {
    this.conditioner.reset();
    this.spectralSnr.reset();
    this.context.fill(0.0);
    this.state.fill(0.0);
  }

  async scoreSingleFrame(framePcm, preconditioned = false) {
    const clean = preconditioned ? framePcm : this.conditioner.process(framePcm);
    const { snrDb } = this.spectralSnr.scoreFrame(clean);

    this.windowBuf.set(this.context, 0);
    this.windowBuf.set(clean, CONTEXT_SAMPLES);
    this.context.set(clean.subarray(FRAME_SAMPLES - CONTEXT_SAMPLES));

    const encOut = await this.encoderSession.run({
      input: new this.ort.Tensor('float32', this.windowBuf, [1, WINDOW_SAMPLES]),
    });
    const feats = encOut.acoustic_features.data;
    const recOut = await this.recurrentSession.run({
      acoustic_features: new this.ort.Tensor('float32', feats, [1, 128]),
      state: new this.ort.Tensor('float32', this.state, [2, 1, 128]),
    });
    const prob = recOut.output.data[0];
    this.state.set(recOut.stateN.data);
    return { probability: prob, snrDb, cleanFrame: clean };
  }

  async scoreBatchPcm16k(pcm16k, precondition = true) {
    const cleanPcm = precondition ? new PreConditioner16k({ highPassHz: 80, humNotch: true }).process(pcm16k) : pcm16k;
    const numFrames = Math.floor(cleanPcm.length / FRAME_SAMPLES);
    if (numFrames === 0) {
      return { probabilities: new Float32Array(0), snrs: new Float32Array(0), cleanPcm };
    }

    const { windows } = buildContextWindows(cleanPcm);

    // Batched encoder execution in chunks of up to 2,048 frames (~65.5s per batch) to bound peak RAM
    const BATCH_CHUNK = 2048;
    const probabilities = new Float32Array(numFrames);
    const snrs = new Float32Array(numFrames);
    const snrTracker = new SpectralVadEngine();
    let rnnState = new Float32Array(2 * 1 * 128);

    for (let start = 0; start < numFrames; start += BATCH_CHUNK) {
      const count = Math.min(BATCH_CHUNK, numFrames - start);
      const winSlice = windows.subarray(start * WINDOW_SAMPLES, (start + count) * WINDOW_SAMPLES);
      const encOut = await this.encoderSession.run({
        input: new this.ort.Tensor('float32', winSlice, [count, WINDOW_SAMPLES]),
      });
      const feats = encOut.acoustic_features.data;

      for (let i = 0; i < count; i++) {
        const globalIdx = start + i;
        const frameSlice = cleanPcm.subarray(globalIdx * FRAME_SAMPLES, (globalIdx + 1) * FRAME_SAMPLES);
        snrs[globalIdx] = snrTracker.scoreFrame(frameSlice).snrDb;

        const featSlice = feats.subarray(i * 128, (i + 1) * 128);
        const recOut = await this.recurrentSession.run({
          acoustic_features: new this.ort.Tensor('float32', featSlice, [1, 128]),
          state: new this.ort.Tensor('float32', rnnState, [2, 1, 128]),
        });
        probabilities[globalIdx] = recOut.output.data[0];
        rnnState = recOut.stateN.data;
      }
    }

    return { probabilities, snrs, cleanPcm };
  }

  async dispose() {
    await Promise.allSettled([
      this.encoderSession?.release?.(),
      this.recurrentSession?.release?.(),
    ]);
  }
}

class DspBackend {
  constructor() {
    this.engine = new SpectralVadEngine();
  }

  reset() {
    this.engine.reset();
  }

  async scoreSingleFrame(framePcm) {
    const { probability, snrDb } = this.engine.scoreFrame(framePcm);
    return { probability, snrDb, cleanFrame: framePcm };
  }

  async scoreBatchPcm16k(pcm16k) {
    const numFrames = Math.floor(pcm16k.length / FRAME_SAMPLES);
    const probabilities = new Float32Array(numFrames);
    const snrs = new Float32Array(numFrames);
    const localEngine = new SpectralVadEngine();

    for (let i = 0; i < numFrames; i++) {
      const frame = pcm16k.subarray(i * FRAME_SAMPLES, (i + 1) * FRAME_SAMPLES);
      const res = localEngine.scoreFrame(frame);
      probabilities[i] = res.probability;
      snrs[i] = res.snrDb;
    }
    return { probabilities, snrs, cleanPcm: pcm16k };
  }

  async dispose() {}
}

function createDetectorInstance(engine, info, baseOptions = {}) {
  const listeners = {
    frame: new Set(),
    speechStart: new Set(),
    speechEnd: new Set(),
  };

  let streamArbiter = new VadHysteresisArbiter(baseOptions);
  let residualBuf = new Float32Array(FRAME_SAMPLES);
  let residualLen = 0;

  const emit = (eventName, payload) => {
    const set = listeners[eventName];
    if (!set) return;
    for (const fn of set) {
      fn(payload);
    }
  };

  const stepExactFrame = async (frame512, preconditioned = false) => {
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const { probability, snrDb } = await engine.scoreSingleFrame(frame512, preconditioned);
    const latencyMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
    const { event, frameInfo, segment } = streamArbiter.step(frame512, probability, snrDb);
    const enrichedFrame = { ...frameInfo, latencyMs: Number(latencyMs.toFixed(4)), event, segment };

    emit('frame', enrichedFrame);
    if (event === 'speechStart') {
      emit('speechStart', { startMs: frameInfo.startMs, probability, snrDb });
    } else if (event === 'speechEnd' && segment) {
      emit('speechEnd', segment);
    }
    return enrichedFrame;
  };

  const detector = {
    info,

    on(event, fn) {
      if (!listeners[event]) {
        throw new Error(`Unknown event "${event}" (expected 'frame', 'speechStart', or 'speechEnd')`);
      }
      listeners[event].add(fn);
      return () => listeners[event].delete(fn);
    },

    off(event, fn) {
      listeners[event]?.delete(fn);
    },

    reset() {
      engine.reset();
      streamArbiter.reset();
      residualLen = 0;
    },

    flush() {
      const seg = streamArbiter.flush();
      residualLen = 0;
      if (seg) emit('speechEnd', seg);
      return seg;
    },

    async frame(chunk, opts = {}) {
      let f32 = chunk instanceof Int16Array ? int16ToFloat32(chunk) : chunk;
      if (!(f32 instanceof Float32Array)) {
        f32 = Float32Array.from(f32);
      }
      if (opts.sampleRate && opts.sampleRate !== TARGET_SAMPLE_RATE) {
        f32 = resample16k(f32, opts.sampleRate);
      }

      // Fast path for exact 512-sample (32 ms @ 16 kHz) frames with empty residual buffer
      if (residualLen === 0 && f32.length === FRAME_SAMPLES) {
        return stepExactFrame(f32, Boolean(opts.preconditioned));
      }

      let offset = 0;
      let lastResult = null;
      while (offset < f32.length) {
        const space = FRAME_SAMPLES - residualLen;
        const take = Math.min(space, f32.length - offset);
        residualBuf.set(f32.subarray(offset, offset + take), residualLen);
        residualLen += take;
        offset += take;

        if (residualLen === FRAME_SAMPLES) {
          const ready = new Float32Array(residualBuf);
          residualLen = 0;
          lastResult = await stepExactFrame(ready, Boolean(opts.preconditioned));
        }
      }
      return (
        lastResult || {
          frameIndex: streamArbiter.frameIndex,
          startMs: streamArbiter.frameIndex * 32,
          endMs: streamArbiter.frameIndex * 32,
          probability: 0,
          speaking: streamArbiter.speaking,
          rms: computeRms(f32),
          snrDb: 0,
          latencyMs: 0,
          bufferedSamples: residualLen,
          event: null,
          segment: null,
        }
      );
    },

    async *stream(asyncSource, opts = {}) {
      const localQueue = [];
      const unsub = detector.on('speechEnd', (seg) => localQueue.push(seg));
      try {
        for await (const chunk of asyncSource) {
          await detector.frame(chunk, opts);
          while (localQueue.length > 0) {
            yield localQueue.shift();
          }
        }
        detector.flush();
        while (localQueue.length > 0) {
          yield localQueue.shift();
        }
      } finally {
        unsub();
      }
    },

    async detect(audioInput, opts = {}) {
      const mergedOpts = { ...baseOptions, ...opts };
      const pcm16k = await normalizeAudioTo16k(audioInput, mergedOpts.sampleRate || TARGET_SAMPLE_RATE);
      const durationMs = Math.round((pcm16k.length / TARGET_SAMPLE_RATE) * 1000);

      const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const { probabilities, snrs } = await engine.scoreBatchPcm16k(pcm16k, mergedOpts.precondition !== false);
      const inferenceMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;

      const arbiter = new VadHysteresisArbiter(mergedOpts);
      const segments = [];
      const frames = mergedOpts.returnFrames ? [] : null;

      for (let i = 0; i < probabilities.length; i++) {
        // Slice original un-notched 16k PCM for pristine natural voice timbre in extracted segments!
        const rawFrame = pcm16k.subarray(i * FRAME_SAMPLES, (i + 1) * FRAME_SAMPLES);
        const { frameInfo, segment } = arbiter.step(rawFrame, probabilities[i], snrs[i]);
        if (frames) frames.push(frameInfo);
        if (segment) segments.push(segment);
      }
      const tailSeg = arbiter.flush();
      if (tailSeg) segments.push(tailSeg);

      if (mergedOpts.returnFrames) {
        return {
          segments,
          frames,
          probabilities,
          durationMs,
          inferenceMs: Number(inferenceMs.toFixed(3)),
          rtf: durationMs > 0 ? Number((inferenceMs / durationMs).toFixed(6)) : 0,
        };
      }
      return segments;
    },

    async trim(audioInput, opts = {}) {
      const mergedOpts = { ...baseOptions, ...opts };
      const pcm16k = await normalizeAudioTo16k(audioInput, mergedOpts.sampleRate || TARGET_SAMPLE_RATE);
      const originalDurationMs = Math.round((pcm16k.length / TARGET_SAMPLE_RATE) * 1000);
      const segments = await detector.detect({ pcm: pcm16k, sampleRate: TARGET_SAMPLE_RATE }, mergedOpts);

      const overlapSamples = mergedOpts.overlapSamples ?? 128; // 8 ms complementary Hann crossfade
      const { pcm: trimmedPcm, seamIndices } = concatSegmentsHann(
        segments.map((s) => s.pcm),
        overlapSamples,
        64
      );
      const seamStats = verifySeamContinuity(trimmedPcm, seamIndices);
      const durationMs = Math.round((trimmedPcm.length / TARGET_SAMPLE_RATE) * 1000);

      let cachedWav = null;
      return {
        pcm: trimmedPcm,
        get wav() {
          if (!cachedWav) cachedWav = encodeWav(trimmedPcm, TARGET_SAMPLE_RATE);
          return cachedWav;
        },
        sampleRate: TARGET_SAMPLE_RATE,
        durationMs,
        originalDurationMs,
        speechRatio: originalDurationMs > 0 ? Number((durationMs / originalDurationMs).toFixed(4)) : 0,
        segments,
        seamStats,
      };
    },

    async listen(streamOrOpts = {}) {
      return startAudioWorkletCapture(detector, streamOrOpts);
    },

    async dispose() {
      listeners.frame.clear();
      listeners.speechStart.clear();
      listeners.speechEnd.clear();
      await engine.dispose();
    },
  };

  return detector;
}

/**
 * Primary Hemanth-style entry point (`import vad from 'wasm-vad'`).
 *
 * - `const detector = await vad(options?)` -> creates a reusable streaming/batch VAD instance
 * - `const segments = await vad('./speech.wav', options?)` -> smart one-shot utterance detection
 */
export default async function vad(inputOrOptions = {}, maybeOptions = {}) {
  const oneShot = isAudioInput(inputOrOptions);
  const options = oneShot ? maybeOptions || {} : inputOrOptions || {};

  let caps = detectCapabilities(options);
  const loaded = await loadOrtRuntime(caps, options.ort, options.wasmPaths);
  const ort = loaded.ort;
  caps = loaded.caps;

  if (!ort && caps.backend !== 'dsp') {
    if (options.strictOnnx) {
      throw new Error(
        'ONNX Runtime not found: install `onnxruntime-web` (browser) or `onnxruntime-node` (Node.js), or pass `{ tier: "dsp" }`'
      );
    }
    caps = detectCapabilities({ ...options, tier: 'dsp' });
  }

  let engine;
  let modelBytesLen = 0;
  let splitMs = 0;

  if (caps.backend === 'dsp') {
    engine = new DspBackend();
  } else {
    const onnxBytes = await loadModelBytes(caps, options);
    modelBytesLen = onnxBytes.byteLength;
    const split = splitOnnxGraph(onnxBytes);
    splitMs = Number(split.splitMs.toFixed(3));

    const sessOpts = caps.sessionOptions || {};
    let encoderSession;
    let recurrentSession;
    try {
      [encoderSession, recurrentSession] = await Promise.all([
        ort.InferenceSession.create(split.encoderBytes, sessOpts),
        ort.InferenceSession.create(split.recurrentBytes, sessOpts),
      ]);
    } catch (err) {
      if (caps.backend === 'webgpu') {
        // Graceful fallback to WASM SIMD if WebGPU adapter rejects session creation
        const fallbackOpts = { ...sessOpts, executionProviders: ['wasm'] };
        [encoderSession, recurrentSession] = await Promise.all([
          ort.InferenceSession.create(split.encoderBytes, fallbackOpts),
          ort.InferenceSession.create(split.recurrentBytes, fallbackOpts),
        ]);
        caps = { ...caps, backend: 'wasm', deviceClass: 'desktop-wasm', executionProviders: ['wasm'] };
      } else {
        throw err;
      }
    }
    engine = new OnnxSplitBackend(ort, encoderSession, recurrentSession, split.stats);
  }

  const info = Object.freeze({
    tier: caps.tier,
    backend: caps.backend,
    deviceClass: caps.deviceClass,
    numThreads: caps.numThreads,
    crossOriginIsolated: caps.crossOriginIsolated,
    isMobile: caps.isMobile,
    modelBytes: modelBytesLen,
    splitMs,
  });

  const detector = createDetectorInstance(engine, info, options);

  if (oneShot) {
    try {
      return await detector.detect(inputOrOptions, options);
    } finally {
      await detector.dispose();
    }
  }

  return detector;
}

export {
  vad,
  detectCapabilities,
  splitOnnxGraph,
  fetchAndCacheModel,
  SpectralVadEngine,
  VadHysteresisArbiter,
  DEFAULT_VAD_OPTIONS,
  TARGET_SAMPLE_RATE,
  FRAME_SAMPLES,
  CONTEXT_SAMPLES,
  WINDOW_SAMPLES,
  PreConditioner16k,
  resample16k,
  decodeWav,
  encodeWav,
  concatSegmentsHann,
  verifySeamContinuity,
  computeRms,
};
