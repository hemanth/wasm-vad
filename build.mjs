import esbuild from 'esbuild';
import fs from 'node:fs/promises';

await fs.mkdir('dist', { recursive: true });
await fs.mkdir('docs/models', { recursive: true });

const EXTERNAL_PEERS = ['onnxruntime-node', 'onnxruntime-web', 'node:*'];

// 1. ESM Build (`dist/index.mjs`)
await esbuild.build({
  entryPoints: ['src/index.js'],
  outfile: 'dist/index.mjs',
  bundle: true,
  format: 'esm',
  platform: 'neutral',
  target: ['es2022'],
  external: EXTERNAL_PEERS,
  legalComments: 'none',
});

// 2. CJS Build (`dist/index.cjs`) — ensure `const vad = require('wasm-vad')` returns default function directly
await esbuild.build({
  entryPoints: ['src/index.js'],
  outfile: 'dist/index.cjs',
  bundle: true,
  format: 'cjs',
  platform: 'node',
  target: ['node18'],
  external: EXTERNAL_PEERS,
  legalComments: 'none',
  footer: {
    js: `if (module.exports && module.exports.default) {
  const fn = module.exports.default;
  Object.assign(fn, module.exports);
  module.exports = fn;
}`,
  },
});

// 3. Browser Playground Self-Contained ESM Bundle (`docs/wasm-vad.browser.mjs`)
// Bundles `wasm-vad` + `onnxruntime-web` stub-free interface (resolving `window.ort` loaded from local WASM assets)
await esbuild.build({
  entryPoints: ['src/index.js'],
  outfile: 'docs/wasm-vad.browser.mjs',
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: ['es2022'],
  external: ['onnxruntime-node', 'onnxruntime-web', 'node:*'],
  legalComments: 'none',
});

// 4. Copy TypeScript declarations
await fs.copyFile('src/index.d.ts', 'dist/index.d.ts');

// 5. Stage real ONNX checkpoints, ORT WASM runtime & real audio samples into `docs/` so the playground is 100% self-contained
await fs.copyFile('models/vad-mobile.onnx', 'docs/models/vad-mobile.onnx');
await fs.copyFile('models/vad-studio.onnx', 'docs/models/vad-studio.onnx');
await fs.copyFile('models/manifest.json', 'docs/models/manifest.json');

await fs.mkdir('docs/ort', { recursive: true });
const ortDist = 'node_modules/onnxruntime-web/dist';
for (const f of await fs.readdir(ortDist)) {
  if (
    f === 'ort.wasm.min.js' ||
    f === 'ort.min.js' ||
    (f.startsWith('ort-wasm') && (f.endsWith('.wasm') || f.endsWith('.mjs')))
  ) {
    await fs.copyFile(`${ortDist}/${f}`, `docs/ort/${f}`);
  }
}

const swanadaSamples = process.env.SWANADA_SAMPLES || '../swanada/docs/samples';
try {
  await fs.mkdir('docs/samples', { recursive: true });
  const { decodeWav, encodeWav } = await import('./dist/index.mjs');
  const en = decodeWav(await fs.readFile(`${swanadaSamples}/en.wav`)).pcm;
  const sa = decodeWav(await fs.readFile(`${swanadaSamples}/sa.wav`)).pcm;
  const hi = decodeWav(await fs.readFile(`${swanadaSamples}/hi.wav`)).pcm;
  const kn = decodeWav(await fs.readFile(`${swanadaSamples}/kn.wav`)).pcm;

  // Preset 1: Multi-turn conversation with distinct turn pauses (English -> 850ms pause -> Hindi -> 900ms pause -> Sanskrit)
  const p1 = new Float32Array(8000 + en.length + 13600 + hi.length + 14400 + sa.length + 8000);
  let off = 8000;
  p1.set(en, off); off += en.length + 13600;
  p1.set(hi, off); off += hi.length + 14400;
  p1.set(sa, off);
  await fs.writeFile('docs/samples/multiturn_3lang.wav', new Uint8Array(encodeWav(p1, 16000)));

  // Preset 2: +5 dB Street Pink Noise + 50/60 Hz Mains Hum mixed over Kannada + English
  const clean2 = new Float32Array(9600 + kn.length + 14400 + en.length + 9600);
  clean2.set(kn, 9600);
  clean2.set(en, 9600 + kn.length + 14400);
  let seed = 20261112;
  let b0 = 0, b1 = 0, b2 = 0;
  const noisy2 = new Float32Array(clean2.length);
  for (let i = 0; i < clean2.length; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const w = (seed / 0xffffffff) * 2 - 1;
    b0 = 0.99 * b0 + w * 0.08;
    b1 = 0.92 * b1 + w * 0.16;
    b2 = 0.60 * b2 + w * 0.30;
    const hum = 0.035 * Math.sin((2 * Math.PI * 50 * i) / 16000) + 0.025 * Math.sin((2 * Math.PI * 60 * i) / 16000);
    noisy2[i] = Math.max(-1, Math.min(1, clean2[i] + (b0 + b1 + b2) * 0.055 + hum));
  }
  await fs.writeFile('docs/samples/noisy_hum_5db.wav', new Uint8Array(encodeWav(noisy2, 16000)));

  // Preset 3 & 4: Classical Sanskrit & Code-Switched Kanglish
  await fs.copyFile(`${swanadaSamples}/sa_madhva.wav`, 'docs/samples/sa_madhva.wav');
  await fs.copyFile(`${swanadaSamples}/kanglish.wav`, 'docs/samples/kanglish.wav');
} catch {
  // Optional sample staging when local swanada corpus is available
}

console.log('Built dist/index.mjs, dist/index.cjs, dist/index.d.ts, and docs/wasm-vad.browser.mjs');
