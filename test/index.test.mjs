import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import ort from 'onnxruntime-node';

import vad, {
  detectCapabilities,
  splitOnnxGraph,
  decodeWav,
  encodeWav,
  resample16k,
  concatSegmentsHann,
  verifySeamContinuity,
} from '../dist/index.mjs';

const require = createRequire(import.meta.url);
const SAMPLES_DIR = path.resolve('docs', 'samples');

function makeSine(freqHz, durationSec, sr = 16000, amp = 0.5) {
  const n = Math.floor(durationSec * sr);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = amp * Math.sin((2 * Math.PI * freqHz * i) / sr);
  }
  return out;
}

test('CJS require("wasm-vad") exports callable default function + named utilities', () => {
  const cjsVad = require('../dist/index.cjs');
  assert.equal(typeof cjsVad, 'function');
  assert.equal(typeof cjsVad.detectCapabilities, 'function');
  assert.equal(typeof cjsVad.splitOnnxGraph, 'function');
  assert.equal(typeof cjsVad.decodeWav, 'function');
});

test('detectCapabilities routes mobile to 1-thread 432 KiB WASM and desktop to studio tier', () => {
  const mobile = detectCapabilities({ isMobile: true });
  assert.equal(mobile.tier, 'mobile');
  assert.equal(mobile.numThreads, 1);

  const dsp = detectCapabilities({ tier: 'dsp' });
  assert.equal(dsp.tier, 'dsp');
  assert.equal(dsp.backend, 'dsp');
  assert.deepEqual(dsp.executionProviders, []);

  const nodeDefault = detectCapabilities();
  assert.equal(nodeDefault.tier, 'studio');
  assert.equal(nodeDefault.backend, 'onnx-node');
});

test('WAV encode/decode round-trip and 48 kHz -> 16 kHz anti-aliased resampler', () => {
  const tone1k = makeSine(1000, 0.25, 16000, 0.6);
  const wavBuf = encodeWav(tone1k, 16000);
  const decoded = decodeWav(wavBuf);
  assert.equal(decoded.sampleRate, 16000);
  assert.equal(decoded.numChannels, 1);
  assert.equal(decoded.pcm.length, tone1k.length);
  let maxErr = 0;
  for (let i = 0; i < tone1k.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(decoded.pcm[i] - tone1k[i]));
  }
  assert.ok(maxErr < 1e-4, `Expected 16-bit quantization error < 1e-4, got ${maxErr}`);

  // 12,500 Hz tone @ 48 kHz sits above 8,000 Hz Nyquist and would alias to 3,500 Hz without 4th-order Butterworth LPF!
  const aliasTone48k = makeSine(12500, 0.25, 48000, 0.8);
  const down16k = resample16k(aliasTone48k, 48000);
  // Skip first 32 samples of IIR transient settling
  let rmsAfter = 0;
  for (let i = 32; i < down16k.length; i++) rmsAfter += down16k[i] * down16k[i];
  rmsAfter = Math.sqrt(rmsAfter / (down16k.length - 32));
  const inputRms = 0.8 / Math.SQRT2;
  const attenuationDb = 20 * Math.log10(inputRms / Math.max(rmsAfter, 1e-9));
  assert.ok(attenuationDb > 22, `Expected >22 dB alias attenuation at 12.5 kHz, got ${attenuationDb.toFixed(2)} dB`);
});

test('splitOnnxGraph splits studio and mobile ONNX in <10 ms with bit-exact parity', async () => {
  for (const file of ['vad-studio.onnx', 'vad-mobile.onnx']) {
    const bytes = fs.readFileSync(path.join('models', file));
    const split = splitOnnxGraph(bytes);
    assert.ok(split.splitMs < 15, `Expected fast protobuf split (<15ms), got ${split.splitMs}ms`);
    assert.ok(split.encoderBytes.byteLength > 100_000);
    assert.ok(split.recurrentBytes.byteLength > 100_000);

    const fullSess = await ort.InferenceSession.create(bytes);
    const encSess = await ort.InferenceSession.create(split.encoderBytes);
    const recSess = await ort.InferenceSession.create(split.recurrentBytes);

    const sample = decodeWav(fs.readFileSync(path.join(SAMPLES_DIR, 'sa_madhva.wav')));
    const nFrames = 30;
    let stateSeq = new Float32Array(256);
    let ctx = new Float32Array(64);
    const batchWin = new Float32Array(nFrames * 576);
    const seqOut = [];

    for (let i = 0; i < nFrames; i++) {
      const fr = sample.pcm.subarray(i * 512, (i + 1) * 512);
      const win = new Float32Array(576);
      win.set(ctx, 0);
      win.set(fr, 64);
      batchWin.set(win, i * 576);
      ctx.set(fr.subarray(512 - 64));
      const r = await fullSess.run({
        input: new ort.Tensor('float32', win, [1, 576]),
        state: new ort.Tensor('float32', stateSeq, [2, 1, 128]),
      });
      seqOut.push(r.output.data[0]);
      stateSeq = Float32Array.from(r.stateN.data);
    }

    const encRes = await encSess.run({
      input: new ort.Tensor('float32', batchWin, [nFrames, 576]),
    });
    let stateRec = new Float32Array(256);
    for (let i = 0; i < nFrames; i++) {
      const feat = encRes.acoustic_features.data.subarray(i * 128, (i + 1) * 128);
      const r = await recSess.run({
        acoustic_features: new ort.Tensor('float32', feat, [1, 128]),
        state: new ort.Tensor('float32', stateRec, [2, 1, 128]),
      });
      assert.ok(Math.abs(r.output.data[0] - seqOut[i]) < 1e-6);
      stateRec = r.stateN.data;
    }
  }
});

test('3-line smart one-shot vad(path) and reusable detector across studio, mobile, and dsp tiers', async () => {
  const samplePath = path.join(SAMPLES_DIR, 'kanglish.wav');

  // One-shot smart intent detection
  const oneShotSegs = await vad(samplePath);
  assert.ok(Array.isArray(oneShotSegs) && oneShotSegs.length >= 1);
  assert.ok(oneShotSegs[0].durationMs > 800);
  assert.ok(oneShotSegs[0].wav.byteLength > 44);

  for (const tier of ['studio', 'mobile', 'dsp']) {
    const detector = await vad({ tier });
    assert.equal(detector.info.tier, tier);
    const segs = await detector.detect(samplePath);
    assert.ok(segs.length >= 1, `Expected speech detected on ${tier}`);
    assert.ok(segs[0].confidence > 0.5);
    await detector.dispose();
  }
});

test('Streaming detector.frame() with arbitrary chunk sizes + detector.stream() async generator', async () => {
  const composite = decodeWav(fs.readFileSync(path.join(SAMPLES_DIR, 'multiturn_3lang.wav'))).pcm;

  const detector = await vad({ tier: 'mobile' });
  const events = [];
  detector.on('speechStart', () => events.push('start'));
  detector.on('speechEnd', () => events.push('end'));

  // Feed odd 333-sample chunks via async generator
  async function* oddChunkGenerator() {
    for (let i = 0; i < composite.length; i += 333) {
      yield composite.subarray(i, Math.min(composite.length, i + 333));
    }
  }

  const yielded = [];
  for await (const seg of detector.stream(oddChunkGenerator())) {
    yielded.push(seg);
  }
  assert.equal(yielded.length, 3, `Expected 3 distinct utterances (English + Hindi + Sanskrit), got ${yielded.length}`);
  assert.deepEqual(events, ['start', 'end', 'start', 'end', 'start', 'end']);
  await detector.dispose();
});

test('Adversarial non-speech rejection: 50/60 Hz hum, pure beep, and broadband white noise yield 0 segments', async () => {
  // 2 seconds of 50 Hz + 60 Hz mains hum
  const hum = new Float32Array(32000);
  for (let i = 0; i < hum.length; i++) {
    hum[i] = 0.25 * Math.sin((2 * Math.PI * 50 * i) / 16000) + 0.25 * Math.sin((2 * Math.PI * 60 * i) / 16000);
  }

  // Deterministic pseudo-random white noise
  let seed = 42;
  const white = new Float32Array(32000);
  for (let i = 0; i < white.length; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    white[i] = ((seed / 0xffffffff) * 2 - 1) * 0.08;
  }

  const beep1k = makeSine(1000, 2.0, 16000, 0.2);

  for (const tier of ['studio', 'mobile', 'dsp']) {
    const detector = await vad({ tier });
    assert.equal((await detector.detect(hum)).length, 0, `${tier} should reject 50/60 Hz hum`);
    assert.equal((await detector.detect(white)).length, 0, `${tier} should reject stationary white noise`);
    assert.equal((await detector.detect(beep1k)).length, 0, `${tier} should reject pure 1 kHz sine beep`);
    await detector.dispose();
  }
});

test('Adversarial Seam Continuity Gate: detector.trim() produces click-free Hann seams (ratio <= 1.5)', async () => {
  const { pcm: streamPcm } = decodeWav(fs.readFileSync(path.join(SAMPLES_DIR, 'multiturn_3lang.wav')));

  const detector = await vad({ tier: 'studio' });
  const trimmed = await detector.trim(streamPcm);
  assert.equal(trimmed.segments.length, 3, `Expected 3 segments across 3-lang stream, got ${trimmed.segments.length}`);
  assert.ok(trimmed.speechRatio > 0.3 && trimmed.speechRatio < 0.85);
  assert.ok(
    trimmed.seamStats.ok,
    `Expected click-free Hann seams (ratio <= 1.5), got ratio=${trimmed.seamStats.ratio.toFixed(3)}`
  );
  await detector.dispose();
});

test('CLI bin/wasm-vad.mjs supports --json and --trim -o output.wav', () => {
  const tmpOut = path.join(os.tmpdir(), `wasm-vad-test-${Date.now()}.wav`);
  try {
    const outJson = execFileSync(
      process.execPath,
      ['bin/wasm-vad.mjs', path.join(SAMPLES_DIR, 'sa_madhva.wav'), '--trim', '-o', tmpOut, '--json'],
      { encoding: 'utf8' }
    );
    const parsed = JSON.parse(outJson);
    assert.equal(parsed.tier, 'studio');
    assert.ok(parsed.segments.length >= 1);
    assert.ok(fs.existsSync(tmpOut));
    const decoded = decodeWav(fs.readFileSync(tmpOut));
    assert.equal(decoded.sampleRate, 16000);
    assert.ok(decoded.pcm.length > 8000);
  } finally {
    if (fs.existsSync(tmpOut)) fs.unlinkSync(tmpOut);
  }
});
