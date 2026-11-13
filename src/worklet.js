/**
 * Zero-Copy Inline Blob `AudioWorkletProcessor` for Live Browser Microphone Capture (`detector.listen()`).
 *
 * Implements Section 6 of `/browser-streaming-onnx` + `swanada` AudioWorklet architecture:
 *   1. Zero manual `.worklet.js` bundler copy — synthesized at runtime from an inline `Blob` URL
 *   2. 4th-order cascaded Butterworth anti-aliasing low-pass filter (`fc = 7,400 Hz`, `-24 dB/oct`)
 *      before fractional phase-accumulator decimation from `48 kHz`/`44.1 kHz` to `16,000 Hz`
 *   3. Causal 80 Hz Butterworth HPF + 50 Hz & 60 Hz Q=22 mains hum notches inside the audio thread
 *   4. 4-buffer `Float32Array(512)` ping-pong `recycle` pool transferred via zero-copy `ArrayBuffer` ownership
 *      so zero heap allocations occur on the real-time audio rendering thread!
 */

const WORKLET_PROCESSOR_SOURCE = /* js */ `
class WasmVadCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    const inSr = sampleRate || 48000;
    this.inSr = inSr;
    this.outSr = 16000;
    this.ratio = inSr / 16000;
    this.srcPhase = 0.0;
    this.prevSample = 0.0;
    this.hasPrev = false;

    // 4th-order Butterworth anti-aliasing LPF @ 7,400 Hz when input sampleRate > 16,000 Hz
    this.aa1 = inSr > 16000 ? this._makeLowPass(7400.0, inSr, 0.541196100146197) : null;
    this.aa2 = inSr > 16000 ? this._makeLowPass(7400.0, inSr, 1.3065629648763766) : null;

    // 16 kHz Speech Pre-Conditioner: 80 Hz HPF + 50/60 Hz Q=22 hum notches
    this.hpf = this._makeHighPass(80.0, 16000, Math.SQRT1_2);
    this.n50 = this._makeNotch(50.0, 16000, 22.0);
    this.n60 = this._makeNotch(60.0, 16000, 22.0);

    // 4-buffer lock-free ping-pong recycle pool (zero steady-state GC pauses)
    this.pool = [
      new Float32Array(512),
      new Float32Array(512),
      new Float32Array(512),
      new Float32Array(512),
    ];
    this.currentBuf = this.pool.pop();
    this.writeIdx = 0;
    this.stopped = false;

    this.port.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg) return;
      if (msg.type === 'recycle' && msg.buffer instanceof ArrayBuffer && msg.buffer.byteLength === 2048) {
        this.pool.push(new Float32Array(msg.buffer));
      } else if (msg.type === 'stop') {
        this.stopped = true;
      }
    };
  }

  _makeLowPass(fc, sr, q) {
    const w = (2.0 * Math.PI * fc) / sr;
    const cw = Math.cos(w);
    const alpha = Math.sin(w) / (2.0 * q);
    const a0 = 1.0 + alpha;
    return {
      b0: ((1.0 - cw) * 0.5) / a0,
      b1: (1.0 - cw) / a0,
      b2: ((1.0 - cw) * 0.5) / a0,
      a1: (-2.0 * cw) / a0,
      a2: (1.0 - alpha) / a0,
      x1: 0, x2: 0, y1: 0, y2: 0,
    };
  }

  _makeHighPass(fc, sr, q) {
    const w = (2.0 * Math.PI * fc) / sr;
    const cw = Math.cos(w);
    const alpha = Math.sin(w) / (2.0 * q);
    const a0 = 1.0 + alpha;
    return {
      b0: ((1.0 + cw) * 0.5) / a0,
      b1: -(1.0 + cw) / a0,
      b2: ((1.0 + cw) * 0.5) / a0,
      a1: (-2.0 * cw) / a0,
      a2: (1.0 - alpha) / a0,
      x1: 0, x2: 0, y1: 0, y2: 0,
    };
  }

  _makeNotch(f0, sr, q) {
    const w = (2.0 * Math.PI * f0) / sr;
    const cw = Math.cos(w);
    const alpha = Math.sin(w) / (2.0 * q);
    const a0 = 1.0 + alpha;
    return {
      b0: 1.0 / a0,
      b1: (-2.0 * cw) / a0,
      b2: 1.0 / a0,
      a1: (-2.0 * cw) / a0,
      a2: (1.0 - alpha) / a0,
      x1: 0, x2: 0, y1: 0, y2: 0,
    };
  }

  _stepBiquad(st, x) {
    const y = st.b0 * x + st.b1 * st.x1 + st.b2 * st.x2 - st.a1 * st.y1 - st.a2 * st.y2;
    st.x2 = st.x1;
    st.x1 = x;
    st.y2 = st.y1;
    st.y1 = y;
    return y;
  }

  _emitSample16k(s) {
    let y = this._stepBiquad(this.hpf, s);
    y = this._stepBiquad(this.n50, y);
    y = this._stepBiquad(this.n60, y);
    if (y > 1.0) y = 1.0;
    else if (y < -1.0) y = -1.0;

    this.currentBuf[this.writeIdx++] = y;
    if (this.writeIdx === 512) {
      const out = this.currentBuf;
      this.port.postMessage({ type: 'frame16k', pcm: out }, [out.buffer]);
      this.currentBuf = this.pool.length > 0 ? this.pool.pop() : new Float32Array(512);
      this.writeIdx = 0;
    }
  }

  process(inputs) {
    if (this.stopped) return false;
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch0 = input[0];
    if (!ch0 || ch0.length === 0) return true;

    const numCh = input.length;
    const n = ch0.length;

    for (let i = 0; i < n; i++) {
      let s = ch0[i];
      if (numCh > 1) {
        for (let c = 1; c < numCh; c++) s += input[c][i];
        s /= numCh;
      }
      if (this.aa1) {
        s = this._stepBiquad(this.aa2, this._stepBiquad(this.aa1, s));
      }

      if (this.inSr === 16000) {
        this._emitSample16k(s);
      } else {
        if (!this.hasPrev) {
          this.prevSample = s;
          this.hasPrev = true;
          this._emitSample16k(s);
          this.srcPhase = this.ratio;
          continue;
        }
        // Advance fractional phase across [i-1, i]
        while (this.srcPhase <= 1.0) {
          const interp = (1.0 - this.srcPhase) * this.prevSample + this.srcPhase * s;
          this._emitSample16k(interp);
          this.srcPhase += this.ratio;
        }
        this.srcPhase -= 1.0;
        this.prevSample = s;
      }
    }
    return true;
  }
}
registerProcessor('wasm-vad-capture', WasmVadCaptureProcessor);
`;

export async function startAudioWorkletCapture(detector, streamOrOptions = {}) {
  if (typeof window === 'undefined' || typeof AudioContext === 'undefined') {
    throw new Error('detector.listen() requires a browser environment with Web Audio API & AudioWorklet');
  }

  const isMediaStream =
    streamOrOptions && typeof streamOrOptions.getAudioTracks === 'function';
  const opts = isMediaStream ? { stream: streamOrOptions } : streamOrOptions || {};

  let ownStream = false;
  let mediaStream = opts.stream;
  if (!mediaStream) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('navigator.mediaDevices.getUserMedia is unavailable (requires HTTPS or localhost)');
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: opts.echoCancellation !== false,
        noiseSuppression: opts.noiseSuppression ?? false,
        autoGainControl: opts.autoGainControl !== false,
      },
    });
    ownStream = true;
  }

  const audioCtx = opts.audioContext || new AudioContext({ latencyHint: 'interactive' });
  if (audioCtx.state === 'suspended') {
    await audioCtx.resume();
  }

  const blob = new Blob([WORKLET_PROCESSOR_SOURCE], { type: 'application/javascript' });
  const blobUrl = URL.createObjectURL(blob);
  try {
    await audioCtx.audioWorklet.addModule(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }

  const sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  const workletNode = new AudioWorkletNode(audioCtx, 'wasm-vad-capture', {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    channelCount: 1,
  });

  let paused = false;
  let closed = false;
  let processingChain = Promise.resolve();

  workletNode.port.onmessage = (ev) => {
    const msg = ev.data;
    if (!msg || msg.type !== 'frame16k' || !(msg.pcm instanceof Float32Array)) return;
    const pcmFrame = msg.pcm;

    if (paused || closed) {
      workletNode.port.postMessage({ type: 'recycle', buffer: pcmFrame.buffer }, [pcmFrame.buffer]);
      return;
    }

    processingChain = processingChain
      .then(async () => {
        if (closed) return;
        const res = await detector.frame(pcmFrame, { preconditioned: true });
        if (opts.onFrame) opts.onFrame(res);
      })
      .catch((err) => {
        if (opts.onError) opts.onError(err);
      })
      .finally(() => {
        if (!closed) {
          try {
            workletNode.port.postMessage({ type: 'recycle', buffer: pcmFrame.buffer }, [pcmFrame.buffer]);
          } catch {
            // Port already closed
          }
        }
      });
  };

  sourceNode.connect(workletNode);

  const unsubStart = opts.onSpeechStart ? detector.on('speechStart', opts.onSpeechStart) : null;
  const unsubEnd = opts.onSpeechEnd ? detector.on('speechEnd', opts.onSpeechEnd) : null;

  return {
    stream: mediaStream,
    audioContext: audioCtx,
    pause() {
      paused = true;
    },
    resume() {
      paused = false;
    },
    async stop() {
      if (closed) return;
      closed = true;
      if (unsubStart) unsubStart();
      if (unsubEnd) unsubEnd();
      try {
        workletNode.port.postMessage({ type: 'stop' });
        sourceNode.disconnect();
        workletNode.disconnect();
      } catch {
        // Ignore teardown race
      }
      const flushSeg = detector.flush();
      if (flushSeg && opts.onSpeechEnd) {
        opts.onSpeechEnd(flushSeg);
      }
      if (ownStream) {
        for (const track of mediaStream.getTracks()) track.stop();
      }
      if (!opts.audioContext) {
        await audioCtx.close().catch(() => {});
      }
    },
  };
}
