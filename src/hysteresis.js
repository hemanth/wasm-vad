/**
 * Stateful VAD Frame Accumulator, Pre-Roll Ring Buffer & Dual-Threshold Hysteresis Arbiter.
 *
 * Implements Lessons from `ninada` & `swanada`:
 *   - 64-sample (4 ms) causal receptive-field context prepended to every 512-sample (32 ms) frame (`[1, 576]`)
 *   - Pre-Roll circular buffer (`preSpeechPadMs = 128`, i.e., 4 frames = 2,048 samples) so initial plosives
 *     (`/p, t, k, s/`) before threshold crossing are never clipped
 *   - Dual-threshold hysteresis (`positiveSpeechThreshold = 0.50`, `negativeSpeechThreshold = 0.35`)
 *   - Indic Geminate Stop Hangover (`redemptionMs = 320` ms = 10 frames) so Sanskrit/Kannada/Tamil/Hindi
 *     voiceless geminate stops (`/kk/`, `/tt/`, `/pp/` ~120–240 ms closure) never split an utterance mid-word!
 */

import { TARGET_SAMPLE_RATE, FRAME_SAMPLES, CONTEXT_SAMPLES, WINDOW_SAMPLES, computeRms, encodeWav } from './dsp.js';

const FRAME_MS = (FRAME_SAMPLES / TARGET_SAMPLE_RATE) * 1000; // 32 ms

export const DEFAULT_VAD_OPTIONS = {
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  minSpeechMs: 250,
  redemptionMs: 320,
  preSpeechPadMs: 128,
  maxSpeechMs: Infinity,
};

export class VadHysteresisArbiter {
  constructor(options = {}) {
    const opts = { ...DEFAULT_VAD_OPTIONS, ...options };
    if (opts.threshold !== undefined && options.positiveSpeechThreshold === undefined) {
      opts.positiveSpeechThreshold = opts.threshold;
      opts.negativeSpeechThreshold = Math.max(0.05, opts.threshold - 0.15);
    }
    this.positiveSpeechThreshold = opts.positiveSpeechThreshold;
    this.negativeSpeechThreshold = opts.negativeSpeechThreshold;
    this.minSpeechFrames = Math.max(1, Math.round(opts.minSpeechMs / FRAME_MS));
    this.redemptionFrames = Math.max(1, Math.round(opts.redemptionMs / FRAME_MS));
    this.preSpeechPadFrames = Math.max(0, Math.round(opts.preSpeechPadMs / FRAME_MS));
    this.maxSpeechFrames = Number.isFinite(opts.maxSpeechMs)
      ? Math.max(this.minSpeechFrames, Math.round(opts.maxSpeechMs / FRAME_MS))
      : Infinity;

    this.reset();
  }

  reset() {
    this.speaking = false;
    this.frameIndex = 0;
    this.redemptionCount = 0;
    this.preRoll = []; // ring of `{ frameIndex, pcm }`
    this.activeFrames = [];
    this.activeProbs = [];
    this.activeSnrs = [];
    this.speechStartFrame = 0;
  }

  /**
   * Processes one 512-sample frame with its model probability `prob` and optional `snrDb`.
   * Returns `{ event: null | 'speechStart' | 'speechEnd', frameInfo, segment }`.
   */
  step(framePcm, prob, snrDb = 0.0) {
    const copy = new Float32Array(framePcm);
    const idx = this.frameIndex++;
    const timestampMs = idx * FRAME_MS;
    const rms = computeRms(copy);

    let event = null;
    let committedSegment = null;

    if (!this.speaking) {
      if (prob >= this.positiveSpeechThreshold) {
        this.speaking = true;
        this.redemptionCount = 0;
        const startFrame = this.preRoll.length > 0 ? this.preRoll[0].frameIndex : idx;
        this.speechStartFrame = startFrame;
        this.activeFrames = this.preRoll.map((p) => p.pcm);
        this.activeFrames.push(copy);
        this.activeProbs = [prob];
        this.activeSnrs = [snrDb];
        this.preRoll = [];
        event = 'speechStart';
      } else {
        if (this.preSpeechPadFrames > 0) {
          this.preRoll.push({ frameIndex: idx, pcm: copy });
          if (this.preRoll.length > this.preSpeechPadFrames) {
            this.preRoll.shift();
          }
        }
      }
    } else {
      this.activeFrames.push(copy);
      this.activeProbs.push(prob);
      this.activeSnrs.push(snrDb);

      const currentSpeechFrames = this.activeProbs.length;
      if (currentSpeechFrames >= this.maxSpeechFrames) {
        committedSegment = this._finalizeSegment(idx + 1, 0);
        event = 'speechEnd';
      } else if (prob < this.negativeSpeechThreshold) {
        this.redemptionCount++;
        if (this.redemptionCount >= this.redemptionFrames) {
          // Retain up to 2 frames (64 ms) of natural trailing breath/release, trim excess silence tail
          const trimTail = Math.max(0, this.redemptionCount - 2);
          committedSegment = this._finalizeSegment(idx + 1 - trimTail, trimTail);
          if (committedSegment) {
            event = 'speechEnd';
          }
        }
      } else {
        this.redemptionCount = 0;
      }
    }

    const frameInfo = {
      frameIndex: idx,
      startMs: timestampMs,
      endMs: timestampMs + FRAME_MS,
      probability: prob,
      speaking: this.speaking,
      rms,
      snrDb,
    };

    return { event, frameInfo, segment: committedSegment };
  }

  /**
   * Flushes any in-progress utterance at end-of-stream.
   */
  flush() {
    if (!this.speaking || this.activeFrames.length === 0) {
      this.reset();
      return null;
    }
    const trimTail = Math.min(this.redemptionCount, Math.max(0, this.activeFrames.length - 1));
    const endFrame = this.frameIndex - trimTail;
    return this._finalizeSegment(endFrame, trimTail);
  }

  _finalizeSegment(endFrame, trimTailFrames) {
    const keepCount = Math.max(0, this.activeFrames.length - trimTailFrames);
    const coreSpeechFrames = this.activeProbs.length - this.redemptionCount;

    const framesToConcat = this.activeFrames.slice(0, keepCount);
    const probs = this.activeProbs.slice();
    const snrs = this.activeSnrs.slice();
    const startFrame = this.speechStartFrame;

    // Seed preRoll with the trimmed tail frames so immediate back-to-back speech isn't starved
    const tailFrames = this.activeFrames.slice(keepCount);
    this.speaking = false;
    this.redemptionCount = 0;
    this.activeFrames = [];
    this.activeProbs = [];
    this.activeSnrs = [];
    this.preRoll = tailFrames.slice(-this.preSpeechPadFrames).map((pcm, i) => ({
      frameIndex: endFrame + i,
      pcm,
    }));

    if (coreSpeechFrames < this.minSpeechFrames || framesToConcat.length === 0) {
      return null;
    }

    const totalSamples = framesToConcat.length * FRAME_SAMPLES;
    const pcm = new Float32Array(totalSamples);
    let peakRms = 0.0;
    for (let i = 0; i < framesToConcat.length; i++) {
      pcm.set(framesToConcat[i], i * FRAME_SAMPLES);
      const r = computeRms(framesToConcat[i]);
      if (r > peakRms) peakRms = r;
    }

    const confidence = probs.reduce((a, b) => a + b, 0) / Math.max(1, probs.length);
    const snrDb = snrs.reduce((a, b) => a + b, 0) / Math.max(1, snrs.length);
    const startSample = startFrame * FRAME_SAMPLES;
    const endSample = startSample + totalSamples;
    const startMs = Math.round((startSample / TARGET_SAMPLE_RATE) * 1000);
    const endMs = Math.round((endSample / TARGET_SAMPLE_RATE) * 1000);

    let cachedWav = null;
    const seg = {
      startMs,
      endMs,
      durationMs: endMs - startMs,
      startSample,
      endSample,
      confidence: Number(confidence.toFixed(4)),
      peakRms: Number(peakRms.toFixed(5)),
      snrDb: Number(snrDb.toFixed(2)),
      pcm,
      get wav() {
        if (!cachedWav) cachedWav = encodeWav(pcm, TARGET_SAMPLE_RATE);
        return cachedWav;
      },
    };
    return seg;
  }
}

/**
 * Prepends 64-sample causal receptive context to each 512-sample frame -> `[N, 576]` contiguous Float32Array.
 */
export function buildContextWindows(pcm16k, initialContext = null) {
  const numFrames = Math.floor(pcm16k.length / FRAME_SAMPLES);
  const windows = new Float32Array(numFrames * WINDOW_SAMPLES);
  const ctx = new Float32Array(CONTEXT_SAMPLES);
  if (initialContext && initialContext.length === CONTEXT_SAMPLES) {
    ctx.set(initialContext);
  }

  for (let i = 0; i < numFrames; i++) {
    const frameOffset = i * FRAME_SAMPLES;
    const winOffset = i * WINDOW_SAMPLES;
    windows.set(ctx, winOffset);
    const frame = pcm16k.subarray(frameOffset, frameOffset + FRAME_SAMPLES);
    windows.set(frame, winOffset + CONTEXT_SAMPLES);
    ctx.set(frame.subarray(FRAME_SAMPLES - CONTEXT_SAMPLES));
  }
  return { windows, numFrames, nextContext: ctx };
}
