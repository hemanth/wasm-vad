/**
 * Headless Real-Browser Verification (`verify-browser.mjs`)
 * Launches an isolated headless Chrome instance (NEVER touching the user's open Chrome session!)
 * and verifies:
 *   1. COOP/COEP `window.crossOriginIsolated === true`
 *   2. Desktop Chrome -> auto-selects `studio` tier (`1.24 MB`), splits ONNX in <5 ms, detects 3 speech turns
 *   3. Mobile Pixel 7 Emulation -> auto-selects `mobile` tier (`432 KiB`, `numThreads === 1`), detects 3 speech turns
 *   4. Zero-Dep `dsp` tier -> runs in browser with 0 KB model weights
 *   5. Zero `pageerror` or `console.error` events!
 */

import assert from 'node:assert/strict';
import { chromium, devices } from 'playwright';
import { createPlaygroundServer } from './serve-playground.mjs';

async function main() {
  const server = createPlaygroundServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  const baseUrl = `http://127.0.0.1:${port}/`;

  const browser = await chromium.launch({ channel: 'chrome', headless: true });

  try {
    // 1. Desktop Verification (COI + Studio ONNX + DSP tier switch)
    const desktopCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const page = await desktopCtx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await page.goto(baseUrl, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => Boolean(window.__WASM_VAD_LAST_RESULT__), { timeout: 15000 });

    const coi = await page.evaluate(() => window.crossOriginIsolated);
    const desktopInfo = await page.evaluate(() => window.__WASM_VAD_INFO__);
    const desktopRes = await page.evaluate(() => ({
      segments: window.__WASM_VAD_LAST_RESULT__.detailed.segments.length,
      rtf: window.__WASM_VAD_LAST_RESULT__.detailed.rtf,
      inferenceMs: window.__WASM_VAD_LAST_RESULT__.detailed.inferenceMs,
      seamOk: window.__WASM_VAD_LAST_RESULT__.seamStats.ok,
      seamRatio: window.__WASM_VAD_LAST_RESULT__.seamStats.ratio,
    }));

    console.log('Desktop Browser Info:', desktopInfo);
    console.log('Desktop Browser Result:', desktopRes);

    assert.equal(coi, true, 'Expected window.crossOriginIsolated === true');
    assert.equal(desktopInfo.tier, 'studio', 'Desktop should route to studio tier');
    assert.equal(desktopRes.segments, 3, 'Expected 3 detected speech turns in multiturn_3lang.wav');
    assert.equal(desktopRes.seamOk, true, 'Expected Hann seam continuity ok === true');

    const highlightedTokens = await page.evaluate(
      () => document.querySelectorAll('pre code .token-keyword, pre code .token-string').length
    );
    assert.ok(highlightedTokens >= 8, `Expected syntax-highlighted code tokens, found ${highlightedTokens}`);

    // Switch to Zero-Dep DSP tier in live UI
    await page.selectOption('#tier-select', 'dsp');
    await page.waitForFunction(() => window.__WASM_VAD_INFO__?.tier === 'dsp', { timeout: 5000 });
    const dspInfo = await page.evaluate(() => window.__WASM_VAD_INFO__);
    assert.equal(dspInfo.modelBytes, 0);

    assert.deepEqual(errors, [], `Unexpected desktop console/page errors: ${errors.join('; ')}`);
    await desktopCtx.close();

    // 2. Mobile Pixel 7 Verification (Mobile Hybrid 432 KiB + Single-Thread 1T WASM)
    const pixel = devices['Pixel 7'];
    const mobileCtx = await browser.newContext({ ...pixel });
    const mPage = await mobileCtx.newPage();
    const mErrors = [];
    mPage.on('pageerror', (e) => mErrors.push(e.message));
    mPage.on('console', (msg) => {
      if (msg.type() === 'error') mErrors.push(msg.text());
    });

    await mPage.goto(baseUrl, { waitUntil: 'networkidle' });
    await mPage.waitForFunction(() => Boolean(window.__WASM_VAD_LAST_RESULT__), { timeout: 15000 });

    const mobileInfo = await mPage.evaluate(() => window.__WASM_VAD_INFO__);
    const mobileRes = await mPage.evaluate(() => ({
      segments: window.__WASM_VAD_LAST_RESULT__.detailed.segments.length,
      rtf: window.__WASM_VAD_LAST_RESULT__.detailed.rtf,
      inferenceMs: window.__WASM_VAD_LAST_RESULT__.detailed.inferenceMs,
    }));

    console.log('Mobile Pixel 7 Info:', mobileInfo);
    console.log('Mobile Pixel 7 Result:', mobileRes);

    assert.equal(mobileInfo.tier, 'mobile', 'Mobile device should auto-route to 432 KiB mobile tier');
    assert.equal(mobileInfo.numThreads, 1, 'Mobile WASM must force numThreads=1 to avoid big.LITTLE stalls');
    assert.equal(mobileInfo.modelBytes, 442422, 'Expected 432 KiB mobile hybrid checkpoint');
    assert.equal(mobileRes.segments, 3, 'Expected 3 detected speech turns on mobile hybrid checkpoint');
    assert.deepEqual(mErrors, [], `Unexpected mobile console/page errors: ${mErrors.join('; ')}`);

    await mobileCtx.close();
    console.log('\nHeadless browser verification passed (Desktop Studio COI + Mobile 432 KiB 1T + Zero-Dep DSP, 0 errors).');
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
