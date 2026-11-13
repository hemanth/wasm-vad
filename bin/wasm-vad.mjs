#!/usr/bin/env node
import fs from 'node:fs/promises';
import vad from '../dist/index.mjs';

const HELP = `wasm-vad — Streaming 16 kHz Voice Activity Detection CLI

Usage:
  wasm-vad <input.wav> [options]

Options:
  --json                 Output detected speech segments as JSON
  --trim                 Concatenate speech segments with Hann crossfades
  -o, --out <file.wav>   Write trimmed 16 kHz WAV output to <file.wav>
  --tier <name>          Model tier: studio | mobile | dsp (default: studio)
  --threshold <0..1>     Positive speech probability threshold (default: 0.5)
  --min-speech <ms>      Minimum speech duration in ms (default: 250)
  --redemption <ms>      Silence hangover / geminate stop bridge in ms (default: 320)
  -h, --help             Show this help message
`;

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    process.stdout.write(HELP);
    process.exit(0);
  }

  let inputPath = null;
  let outPath = null;
  let json = false;
  let trim = false;
  let tier = 'studio';
  let threshold = 0.5;
  let minSpeechMs = 250;
  let redemptionMs = 320;

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--json') json = true;
    else if (a === '--trim') trim = true;
    else if (a === '-o' || a === '--out') outPath = args[++i];
    else if (a === '--tier') tier = args[++i];
    else if (a === '--threshold') threshold = Number(args[++i]);
    else if (a === '--min-speech') minSpeechMs = Number(args[++i]);
    else if (a === '--redemption') redemptionMs = Number(args[++i]);
    else if (!a.startsWith('-') && !inputPath) inputPath = a;
  }

  if (!inputPath) {
    process.stderr.write('Error: missing <input.wav> path.\n\n' + HELP);
    process.exit(1);
  }

  const detector = await vad({ tier, threshold, minSpeechMs, redemptionMs });
  try {
    if (trim || outPath) {
      const res = await detector.trim(inputPath);
      if (outPath) {
        await fs.writeFile(outPath, new Uint8Array(res.wav));
      }
      const summary = {
        input: inputPath,
        output: outPath || null,
        tier: detector.info.tier,
        backend: detector.info.backend,
        originalDurationMs: res.originalDurationMs,
        speechDurationMs: res.durationMs,
        speechRatio: res.speechRatio,
        seamMaxRatio: Number(res.seamStats.ratio.toFixed(4)),
        segments: res.segments.map((s) => ({
          startMs: s.startMs,
          endMs: s.endMs,
          durationMs: s.durationMs,
          confidence: s.confidence,
          snrDb: s.snrDb,
        })),
      };
      if (json) {
        process.stdout.write(JSON.stringify(summary, null, 2) + '\n');
      } else {
        process.stdout.write(
          `[${detector.info.tier}] ${summary.segments.length} segment(s) | ${res.durationMs}ms / ${res.originalDurationMs}ms (${(
            res.speechRatio * 100
          ).toFixed(1)}% speech)${outPath ? ` -> ${outPath}` : ''}\n`
        );
      }
    } else {
      const detailed = await detector.detect(inputPath, { returnFrames: true });
      const payload = {
        input: inputPath,
        tier: detector.info.tier,
        backend: detector.info.backend,
        durationMs: detailed.durationMs,
        inferenceMs: detailed.inferenceMs,
        rtf: detailed.rtf,
        segments: detailed.segments.map((s) => ({
          startMs: s.startMs,
          endMs: s.endMs,
          durationMs: s.durationMs,
          confidence: s.confidence,
          snrDb: s.snrDb,
        })),
      };
      if (json) {
        process.stdout.write(JSON.stringify(payload, null, 2) + '\n');
      } else {
        process.stdout.write(
          `[${detector.info.tier}] ${payload.segments.length} segment(s) in ${detailed.durationMs}ms audio (${detailed.inferenceMs}ms, RTF ${detailed.rtf}):\n`
        );
        for (const s of payload.segments) {
          process.stdout.write(
            `  ${String(s.startMs).padStart(6)}ms -> ${String(s.endMs).padStart(6)}ms  (${s.durationMs}ms, conf=${s.confidence}, snr=${s.snrDb}dB)\n`
          );
        }
      }
    }
  } finally {
    await detector.dispose();
  }
}

main().catch((err) => {
  process.stderr.write(`wasm-vad error: ${err.message}\n`);
  process.exit(1);
});
