/**
 * Capability-Routed Runtime & Checkpoint Selection (Desktop WebGPU vs Desktop WASM vs Mobile 1T WASM vs Zero-Dep DSP).
 * Avoids mobile WebGPU cold-shader compilation cliffs and mobile big.LITTLE multi-thread futex barrier stalls.
 */

export function detectCapabilities(overrides = {}) {
  const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node) && typeof window === 'undefined';

  if (overrides.backend === 'dsp' || overrides.backend === 'zero-dep' || overrides.tier === 'dsp') {
    return {
      deviceClass: 'zero-dep',
      tier: 'dsp',
      backend: 'dsp',
      executionProviders: [],
      numThreads: 1,
      crossOriginIsolated: isNode ? true : Boolean(globalThis.crossOriginIsolated),
      isMobile: false,
      hasWebGPU: false,
      sessionOptions: null,
    };
  }

  if (isNode && !overrides.isMobile && overrides.backend !== 'wasm' && overrides.backend !== 'webgpu') {
    const tier = overrides.tier === 'mobile' ? 'mobile' : 'studio';
    const cores = typeof navigator !== 'undefined' && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 4;
    const numThreads = overrides.numThreads || (tier === 'mobile' ? 1 : Math.min(4, Math.max(1, cores)));
    return {
      deviceClass: 'node-server',
      tier,
      backend: 'onnx-node',
      executionProviders: ['cpu'],
      numThreads,
      crossOriginIsolated: true,
      isMobile: false,
      hasWebGPU: false,
      sessionOptions: {
        intraOpNumThreads: numThreads,
        interOpNumThreads: 1,
        enableCpuMemArena: false,
        enableMemPattern: false,
        graphOptimizationLevel: 'all',
      },
    };
  }

  const nav = typeof navigator !== 'undefined' ? navigator : {};
  const ua = nav.userAgent || '';
  const platform = nav.platform || '';
  const maxTouch = nav.maxTouchPoints || 0;
  const screenW = typeof screen !== 'undefined' && screen.width ? screen.width : 1440;
  const deviceMemory = typeof nav.deviceMemory === 'number' ? nav.deviceMemory : 8;
  const cores = nav.hardwareConcurrency || 4;
  const coi = Boolean(globalThis.crossOriginIsolated);
  const hasWebGPU = typeof nav.gpu !== 'undefined';

  const isIOS = /iPhone|iPad|iPod|CriOS|FxiOS|EdgiOS/i.test(ua) || (platform === 'MacIntel' && maxTouch > 1);
  const isMobile =
    Boolean(overrides.isMobile) ||
    isIOS ||
    /Android|Mobile/i.test(ua) ||
    (maxTouch > 1 && screenW < 1024) ||
    deviceMemory <= 4;

  // Mobile / Tablet / Low-RAM: Mobile Distilled Hybrid Tier (432 KiB) + single-thread WASM (avoids big.LITTLE stalls & WebGPU cold-shader cliffs)
  if (isMobile && overrides.tier !== 'studio') {
    return {
      deviceClass: 'mobile-wasm',
      tier: 'mobile',
      backend: 'wasm',
      executionProviders: ['wasm'],
      numThreads: 1,
      crossOriginIsolated: coi,
      isMobile: true,
      hasWebGPU,
      sessionOptions: {
        executionProviders: ['wasm'],
        enableCpuMemArena: false,
        enableMemPattern: false,
        graphOptimizationLevel: 'all',
      },
    };
  }

  const tier = overrides.tier === 'mobile' ? 'mobile' : 'studio';
  const wantsWebGPU =
    overrides.backend === 'webgpu' ||
    (overrides.backend !== 'wasm' && hasWebGPU && !isMobile && deviceMemory >= 8 && overrides.preferWebGPU === true);

  if (wantsWebGPU && hasWebGPU) {
    return {
      deviceClass: 'desktop-webgpu',
      tier,
      backend: 'webgpu',
      executionProviders: ['webgpu', 'wasm'],
      numThreads: coi ? Math.min(4, cores) : 1,
      crossOriginIsolated: coi,
      isMobile: false,
      hasWebGPU: true,
      sessionOptions: {
        executionProviders: ['webgpu', 'wasm'],
        enableCpuMemArena: false,
        enableMemPattern: false,
        graphOptimizationLevel: 'all',
      },
    };
  }

  const numThreads = overrides.numThreads || (coi ? Math.min(4, Math.max(1, cores)) : 1);
  return {
    deviceClass: 'desktop-wasm',
    tier,
    backend: 'wasm',
    executionProviders: ['wasm'],
    numThreads,
    crossOriginIsolated: coi,
    isMobile,
    hasWebGPU,
    sessionOptions: {
      executionProviders: ['wasm'],
      enableCpuMemArena: false,
      enableMemPattern: false,
      graphOptimizationLevel: 'all',
    },
  };
}
