/**
 * Two-Layer Persistent Browser Model Cache (`CacheStorage` + `IndexedDB` fallback).
 * Ensures the 432 KiB mobile or 1.24 MB studio ONNX specialist is fetched once per origin,
 * validated against ONNX `ModelProto` wire magic (`0x08` + min byte floor), and loaded in <15 ms on repeat visits.
 */

const CACHE_NAME = 'wasm-vad-onnx-v1';
const IDB_NAME = 'wasm-vad-idb-v1';
const IDB_STORE = 'models';
const MIN_ONNX_BYTES = 16384;

function isValidOnnxBuffer(bytes) {
  return bytes instanceof Uint8Array && bytes.byteLength >= MIN_ONNX_BYTES && bytes[0] === 0x08;
}

function openIdb() {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(IDB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(IDB_STORE)) {
          db.createObjectStore(IDB_STORE);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function idbGet(key) {
  const db = await openIdb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const req = tx.objectStore(IDB_STORE).get(key);
      req.onsuccess = () => {
        const val = req.result;
        db.close();
        if (!val) return resolve(null);
        const bytes = val instanceof Uint8Array ? val : new Uint8Array(val);
        resolve(isValidOnnxBuffer(bytes) ? bytes : null);
      };
      req.onerror = () => {
        db.close();
        resolve(null);
      };
    } catch {
      db.close();
      resolve(null);
    }
  });
}

async function idbPut(key, bytes) {
  const db = await openIdb();
  if (!db) return;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      tx.objectStore(IDB_STORE).put(bytes, key);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        resolve();
      };
    } catch {
      db.close();
      resolve();
    }
  });
}

async function cacheStorageGet(url) {
  if (typeof caches === 'undefined') return null;
  try {
    const cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (!hit) return null;
    const bytes = new Uint8Array(await hit.arrayBuffer());
    if (!isValidOnnxBuffer(bytes)) {
      await cache.delete(url);
      return null;
    }
    return bytes;
  } catch {
    return null;
  }
}

async function cacheStoragePut(url, bytes) {
  if (typeof caches === 'undefined') return;
  try {
    const cache = await caches.open(CACHE_NAME);
    const res = new Response(bytes, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(bytes.byteLength),
      },
    });
    await cache.put(url, res);
  } catch {
    // Private browsing or quota limit — fall back silently to IndexedDB
  }
}

export async function fetchAndCacheModel(url, { onProgress, cache = true } = {}) {
  if (cache) {
    const fromCache = await cacheStorageGet(url);
    if (fromCache) {
      if (onProgress) onProgress({ loaded: fromCache.byteLength, total: fromCache.byteLength, percent: 100, cached: true });
      return fromCache;
    }
    const fromIdb = await idbGet(url);
    if (fromIdb) {
      if (onProgress) onProgress({ loaded: fromIdb.byteLength, total: fromIdb.byteLength, percent: 100, cached: true });
      return fromIdb;
    }
  }

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ONNX model (${res.status} ${res.statusText}): ${url}`);
  }

  const total = Number(res.headers.get('content-length')) || 0;
  let bytes;

  if (res.body && typeof res.body.getReader === 'function' && onProgress) {
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress({
        loaded,
        total: total || loaded,
        percent: total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : 0,
        cached: false,
      });
    }
    bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const c of chunks) {
      bytes.set(c, offset);
      offset += c.byteLength;
    }
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
    if (onProgress) {
      onProgress({ loaded: bytes.byteLength, total: bytes.byteLength, percent: 100, cached: false });
    }
  }

  if (!isValidOnnxBuffer(bytes)) {
    throw new Error(`Downloaded payload from ${url} is not a valid ONNX ModelProto (${bytes.byteLength} bytes)`);
  }

  if (cache) {
    await Promise.all([cacheStoragePut(url, bytes), idbPut(url, bytes)]);
  }
  return bytes;
}
