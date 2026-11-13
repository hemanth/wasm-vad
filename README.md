# wasm-vad

Detect speech turns and strip silence at 16 kHz in browsers and Node.js via branch-free WebGPU/WASM ONNX specialists (432 KiB mobile / 1.24 MB studio) and a zero-dependency spectral fallback.

```bash
npm install wasm-vad
```

## Quick start

```js
import vad from 'wasm-vad';

const segments = await vad('./meeting.wav');
// [{ startMs: 352, endMs: 4416, durationMs: 4064, confidence: 0.9228, snrDb: 13.26, pcm, wav }]

const detector = await vad();
const clean = await detector.trim('./podcast.wav');
// clean.wav -> click-free Hann-crossfaded 16 kHz WAV ArrayBuffer
```

`vad(audio)` runs one-shot detection. `vad()` returns a reusable detector where `detect()` finds utterances, `trim()` strips dead air without boundary clicks, and `listen()` captures live microphone turns. That's the whole API.

## Live microphone

```js
const detector = await vad();

const mic = await detector.listen({
  onSpeechStart: ({ startMs, snrDb }) => console.log('Turn start:', startMs, snrDb),
  onSpeechEnd: (utterance) => sendToAsr(utterance.wav),
});

// Later: mic.pause(), mic.resume(), await mic.stop()
```

Spins up an inline `AudioWorkletProcessor` with 4th-order Butterworth anti-aliasing (`48 kHz -> 16 kHz`), `80 Hz` high-pass + `50/60 Hz` hum notches, a `128 ms` plosive pre-roll buffer, and `320 ms` stop-closure hangover—no `.worklet.js` bundler config required.

## Streaming chunks

```js
for await (const utterance of detector.stream(pcmAsyncIterable, { sampleRate: 48000 })) {
  console.log(utterance.startMs, utterance.endMs, utterance.pcm.length);
}
```

Pass `tier: 'mobile'` (`432 KiB` hybrid `W8A32+FP16` on single-thread WASM), `tier: 'studio'` (`1.24 MB` `FP32` with `<1 ms` binary protobuf graph splitting), or `tier: 'dsp'` (`0 KB` zero-weight spectral-LRT engine). Works out of the box in browsers and Node.js via bundled `onnxruntime-web` (or optional `onnxruntime-node` on servers).

## CLI

```bash
npx wasm-vad input.wav --json
npx wasm-vad input.wav --trim -o speech-only.wav --tier mobile --threshold 0.5
```

- `--json` outputs segment timestamps, confidence, SNR, and real-time factor
- `--trim -o <file.wav>` writes Hann-crossfaded speech-only 16 kHz WAV
- `--tier studio|mobile|dsp` selects the specialist checkpoint

## Benchmarks

Evaluated on `N = 120` 12-language multi-turn, noisy (`+12 dB` / `+5 dB` pink noise, `50/60 Hz` hum), and adversarial non-speech streams (`24,265` frames, `B = 1,000` bootstrap 95% CIs via `npm run bench`):

| Engine | Weights | Frame F1 [95% CI] | ROC-AUC | Non-Speech FPR | Node RTF | Chrome WASM RTF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Naive RMS gate | `0 KB` | `79.05%` `[76.65..81.26]` | `0.6735` | `77.08%` | `0.00002` | — |
| `wasm-vad` (`dsp`) | `0 KB` | `82.84%` `[79.52..86.14]` | `0.9053` | `8.07%` | `0.00037` | `0.0006` |
| `wasm-vad` (`mobile`) | `432 KiB` | `95.01%` `[94.35..95.67]` | `0.9822` | `0.00%` | `0.00268` | `0.0052` |
| `wasm-vad` (`studio`) | `1.24 MB` | `95.08%` `[94.41..95.78]` | `0.9828` | `0.00%` | `0.00216` | `0.0052` |

## License

MIT © [Hemanth.HM](https://h3manth.com)
