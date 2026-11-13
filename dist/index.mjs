// src/capabilities.js
function detectCapabilities(overrides = {}) {
  const isNode = typeof process !== "undefined" && Boolean(process.versions?.node) && typeof window === "undefined";
  if (overrides.backend === "dsp" || overrides.backend === "zero-dep" || overrides.tier === "dsp") {
    return {
      deviceClass: "zero-dep",
      tier: "dsp",
      backend: "dsp",
      executionProviders: [],
      numThreads: 1,
      crossOriginIsolated: isNode ? true : Boolean(globalThis.crossOriginIsolated),
      isMobile: false,
      hasWebGPU: false,
      sessionOptions: null
    };
  }
  if (isNode && !overrides.isMobile && overrides.backend !== "wasm" && overrides.backend !== "webgpu") {
    const tier2 = overrides.tier === "mobile" ? "mobile" : "studio";
    const cores2 = typeof navigator !== "undefined" && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 4;
    const numThreads2 = overrides.numThreads || (tier2 === "mobile" ? 1 : Math.min(4, Math.max(1, cores2)));
    return {
      deviceClass: "node-server",
      tier: tier2,
      backend: "onnx-node",
      executionProviders: ["cpu"],
      numThreads: numThreads2,
      crossOriginIsolated: true,
      isMobile: false,
      hasWebGPU: false,
      sessionOptions: {
        intraOpNumThreads: numThreads2,
        interOpNumThreads: 1,
        enableCpuMemArena: false,
        enableMemPattern: false,
        graphOptimizationLevel: "all"
      }
    };
  }
  const nav = typeof navigator !== "undefined" ? navigator : {};
  const ua = nav.userAgent || "";
  const platform = nav.platform || "";
  const maxTouch = nav.maxTouchPoints || 0;
  const screenW = typeof screen !== "undefined" && screen.width ? screen.width : 1440;
  const deviceMemory = typeof nav.deviceMemory === "number" ? nav.deviceMemory : 8;
  const cores = nav.hardwareConcurrency || 4;
  const coi = Boolean(globalThis.crossOriginIsolated);
  const hasWebGPU = typeof nav.gpu !== "undefined";
  const isIOS = /iPhone|iPad|iPod|CriOS|FxiOS|EdgiOS/i.test(ua) || platform === "MacIntel" && maxTouch > 1;
  const isMobile = Boolean(overrides.isMobile) || isIOS || /Android|Mobile/i.test(ua) || maxTouch > 1 && screenW < 1024 || deviceMemory <= 4;
  if (isMobile && overrides.tier !== "studio") {
    return {
      deviceClass: "mobile-wasm",
      tier: "mobile",
      backend: "wasm",
      executionProviders: ["wasm"],
      numThreads: 1,
      crossOriginIsolated: coi,
      isMobile: true,
      hasWebGPU,
      sessionOptions: {
        executionProviders: ["wasm"],
        enableCpuMemArena: false,
        enableMemPattern: false,
        graphOptimizationLevel: "all"
      }
    };
  }
  const tier = overrides.tier === "mobile" ? "mobile" : "studio";
  const wantsWebGPU = overrides.backend === "webgpu" || overrides.backend !== "wasm" && hasWebGPU && !isMobile && deviceMemory >= 8 && overrides.preferWebGPU === true;
  if (wantsWebGPU && hasWebGPU) {
    return {
      deviceClass: "desktop-webgpu",
      tier,
      backend: "webgpu",
      executionProviders: ["webgpu", "wasm"],
      numThreads: coi ? Math.min(4, cores) : 1,
      crossOriginIsolated: coi,
      isMobile: false,
      hasWebGPU: true,
      sessionOptions: {
        executionProviders: ["webgpu", "wasm"],
        enableCpuMemArena: false,
        enableMemPattern: false,
        graphOptimizationLevel: "all"
      }
    };
  }
  const numThreads = overrides.numThreads || (coi ? Math.min(4, Math.max(1, cores)) : 1);
  return {
    deviceClass: "desktop-wasm",
    tier,
    backend: "wasm",
    executionProviders: ["wasm"],
    numThreads,
    crossOriginIsolated: coi,
    isMobile,
    hasWebGPU,
    sessionOptions: {
      executionProviders: ["wasm"],
      enableCpuMemArena: false,
      enableMemPattern: false,
      graphOptimizationLevel: "all"
    }
  };
}

// src/dsp.js
var TARGET_SAMPLE_RATE = 16e3;
var FRAME_SAMPLES = 512;
var CONTEXT_SAMPLES = 64;
var WINDOW_SAMPLES = 576;
function designHighPassBiquad(fc, sr, q = Math.SQRT1_2) {
  const omega = 2 * Math.PI * fc / sr;
  const sinW = Math.sin(omega);
  const cosW = Math.cos(omega);
  const alpha = sinW / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 + cosW) * 0.5 / a0,
    b1: -(1 + cosW) / a0,
    b2: (1 + cosW) * 0.5 / a0,
    a1: -2 * cosW / a0,
    a2: (1 - alpha) / a0
  };
}
function designLowPassBiquad(fc, sr, q) {
  const omega = 2 * Math.PI * fc / sr;
  const sinW = Math.sin(omega);
  const cosW = Math.cos(omega);
  const alpha = sinW / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 - cosW) * 0.5 / a0,
    b1: (1 - cosW) / a0,
    b2: (1 - cosW) * 0.5 / a0,
    a1: -2 * cosW / a0,
    a2: (1 - alpha) / a0
  };
}
function designNotchBiquad(f0, sr, q = 22) {
  const omega = 2 * Math.PI * f0 / sr;
  const sinW = Math.sin(omega);
  const cosW = Math.cos(omega);
  const alpha = sinW / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: 1 / a0,
    b1: -2 * cosW / a0,
    b2: 1 / a0,
    a1: -2 * cosW / a0,
    a2: (1 - alpha) / a0
  };
}
var BiquadState = class {
  constructor(coeffs) {
    this.b0 = coeffs.b0;
    this.b1 = coeffs.b1;
    this.b2 = coeffs.b2;
    this.a1 = coeffs.a1;
    this.a2 = coeffs.a2;
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;
  }
  reset() {
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;
  }
  processSample(x) {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1;
    this.x1 = x;
    this.y2 = this.y1;
    this.y1 = y;
    return y;
  }
  processBufferInPlace(buf) {
    for (let i = 0; i < buf.length; i++) {
      buf[i] = this.processSample(buf[i]);
    }
    return buf;
  }
};
var PreConditioner16k = class {
  constructor({ highPassHz = 80, humNotch = true } = {}) {
    this.hpf = highPassHz > 0 ? new BiquadState(designHighPassBiquad(highPassHz, TARGET_SAMPLE_RATE)) : null;
    this.n50 = humNotch ? new BiquadState(designNotchBiquad(50, TARGET_SAMPLE_RATE, 22)) : null;
    this.n60 = humNotch ? new BiquadState(designNotchBiquad(60, TARGET_SAMPLE_RATE, 22)) : null;
  }
  reset() {
    if (this.hpf) this.hpf.reset();
    if (this.n50) this.n50.reset();
    if (this.n60) this.n60.reset();
  }
  process(input) {
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      let s = input[i];
      if (this.hpf) s = this.hpf.processSample(s);
      if (this.n50) s = this.n50.processSample(s);
      if (this.n60) s = this.n60.processSample(s);
      out[i] = s > 1 ? 1 : s < -1 ? -1 : s;
    }
    return out;
  }
};
function resample16k(pcm, inputSampleRate = TARGET_SAMPLE_RATE) {
  if (!(pcm instanceof Float32Array)) {
    pcm = Float32Array.from(pcm);
  }
  if (!inputSampleRate || inputSampleRate === TARGET_SAMPLE_RATE) {
    return pcm;
  }
  let filtered = pcm;
  if (inputSampleRate > TARGET_SAMPLE_RATE) {
    const fc = Math.min(7400, 0.45 * TARGET_SAMPLE_RATE);
    const s1 = new BiquadState(designLowPassBiquad(fc, inputSampleRate, 0.541196100146197));
    const s2 = new BiquadState(designLowPassBiquad(fc, inputSampleRate, 1.3065629648763766));
    filtered = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) {
      filtered[i] = s2.processSample(s1.processSample(pcm[i]));
    }
  }
  const ratio = inputSampleRate / TARGET_SAMPLE_RATE;
  const outLength = Math.max(1, Math.round(filtered.length / ratio));
  const out = new Float32Array(outLength);
  const maxIdx = filtered.length - 1;
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const s0 = filtered[Math.min(idx, maxIdx)];
    const s1 = filtered[Math.min(idx + 1, maxIdx)];
    out[i] = (1 - frac) * s0 + frac * s1;
  }
  return out;
}
function decodeWav(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 44) {
    throw new Error("Invalid WAV buffer: shorter than 44 bytes");
  }
  const riff = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  const wave = String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]);
  if (riff !== "RIFF" || wave !== "WAVE") {
    throw new Error(`Unsupported audio header "${riff}/${wave}" (expected RIFF/WAVE)`);
  }
  let audioFormat = 1;
  let numChannels = 1;
  let sampleRate = 16e3;
  let bitsPerSample = 16;
  let dataOffset = -1;
  let dataSize = 0;
  let offset = 12;
  while (offset + 8 <= bytes.byteLength) {
    const chunkId = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    if (chunkId === "fmt " && chunkSize >= 16) {
      audioFormat = view.getUint16(chunkStart, true);
      numChannels = view.getUint16(chunkStart + 2, true);
      sampleRate = view.getUint32(chunkStart + 4, true);
      bitsPerSample = view.getUint16(chunkStart + 14, true);
      if (audioFormat === 65534 && chunkSize >= 40) {
        audioFormat = view.getUint16(chunkStart + 24, true);
      }
    } else if (chunkId === "data") {
      dataOffset = chunkStart;
      dataSize = Math.min(chunkSize, bytes.byteLength - chunkStart);
      break;
    }
    offset = chunkStart + chunkSize + (chunkSize & 1);
  }
  if (dataOffset < 0) {
    throw new Error("Invalid WAV buffer: missing data chunk");
  }
  const bytesPerSample = bitsPerSample >> 3;
  const frameStride = bytesPerSample * numChannels;
  const totalFrames = Math.floor(dataSize / frameStride);
  const pcm = new Float32Array(totalFrames);
  for (let i = 0; i < totalFrames; i++) {
    let acc = 0;
    const base = dataOffset + i * frameStride;
    for (let ch = 0; ch < numChannels; ch++) {
      const p = base + ch * bytesPerSample;
      if (audioFormat === 3 && bitsPerSample === 32) {
        acc += view.getFloat32(p, true);
      } else if (bitsPerSample === 16) {
        acc += view.getInt16(p, true) / 32768;
      } else if (bitsPerSample === 24) {
        const b0 = bytes[p];
        const b1 = bytes[p + 1];
        const b2 = bytes[p + 2];
        let v = b0 | b1 << 8 | b2 << 16;
        if (v & 8388608) v |= ~16777215;
        acc += v / 8388608;
      } else if (bitsPerSample === 32) {
        acc += view.getInt32(p, true) / 2147483648;
      } else if (bitsPerSample === 8) {
        acc += (bytes[p] - 128) / 128;
      } else {
        throw new Error(`Unsupported WAV format=${audioFormat} bitsPerSample=${bitsPerSample}`);
      }
    }
    pcm[i] = acc / numChannels;
  }
  return { pcm, sampleRate, numChannels };
}
function encodeWav(pcm, sampleRate = TARGET_SAMPLE_RATE) {
  const numSamples = pcm.length;
  const buf = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buf);
  const writeStr = (off2, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off2 + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);
  let off = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(off, s < 0 ? Math.round(s * 32768) : Math.round(s * 32767), true);
    off += 2;
  }
  return buf;
}
function concatSegmentsHann(segments, overlapSamples = 128, edgeTaperSamples = 64) {
  const valid = segments.filter((s) => s && s.length > 0);
  if (valid.length === 0) return { pcm: new Float32Array(0), seamIndices: [] };
  if (valid.length === 1) {
    const out2 = new Float32Array(valid[0]);
    applyEdgeTaper(out2, edgeTaperSamples);
    return { pcm: out2, seamIndices: [] };
  }
  let totalLen = valid[0].length;
  for (let k = 1; k < valid.length; k++) {
    const ov = Math.min(overlapSamples, Math.floor(valid[k - 1].length / 3), Math.floor(valid[k].length / 3));
    totalLen += valid[k].length - ov;
  }
  const out = new Float32Array(totalLen);
  out.set(valid[0], 0);
  let writePos = valid[0].length;
  const seamIndices = [];
  for (let k = 1; k < valid.length; k++) {
    const curr = valid[k];
    const ov = Math.min(overlapSamples, Math.floor(writePos / 3), Math.floor(curr.length / 3));
    const seamStart = writePos - ov;
    seamIndices.push(seamStart);
    for (let i = 0; i < ov; i++) {
      const w = 0.5 * (1 - Math.cos(Math.PI * i / ov));
      out[seamStart + i] = out[seamStart + i] * (1 - w) + curr[i] * w;
    }
    out.set(curr.subarray(ov), writePos);
    writePos += curr.length - ov;
  }
  applyEdgeTaper(out, edgeTaperSamples);
  return { pcm: out.subarray(0, writePos), seamIndices };
}
function applyEdgeTaper(buf, taperSamples) {
  const n = Math.min(taperSamples, Math.floor(buf.length / 4));
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1 - Math.cos(Math.PI * i / n));
    buf[i] *= w;
    buf[buf.length - 1 - i] *= w;
  }
}
function verifySeamContinuity(pcm, seamIndices, windowRadius = 4) {
  if (!pcm || pcm.length < 16 || !seamIndices || seamIndices.length === 0) {
    return { ok: true, maxSeamJump: 0, p999InteriorJump: 0, ratio: 0 };
  }
  const seamMask = new Uint8Array(pcm.length);
  for (const idx of seamIndices) {
    for (let j = Math.max(1, idx - windowRadius); j <= Math.min(pcm.length - 1, idx + windowRadius); j++) {
      seamMask[j] = 1;
    }
  }
  const interiorDiffs = [];
  let maxSeamJump = 0;
  for (let i = 1; i < pcm.length; i++) {
    const d = Math.abs(pcm[i] - pcm[i - 1]);
    if (seamMask[i]) {
      if (d > maxSeamJump) maxSeamJump = d;
    } else {
      interiorDiffs.push(d);
    }
  }
  if (interiorDiffs.length === 0) {
    return { ok: true, maxSeamJump, p999InteriorJump: maxSeamJump, ratio: 1 };
  }
  interiorDiffs.sort((a, b) => a - b);
  const p999Idx = Math.min(interiorDiffs.length - 1, Math.floor(interiorDiffs.length * 0.999));
  const p999 = Math.max(interiorDiffs[p999Idx], 1e-6);
  const ratio = maxSeamJump / p999;
  return {
    ok: ratio <= 1.5,
    maxSeamJump,
    p999InteriorJump: p999,
    ratio
  };
}
function computeRms(frame) {
  if (!frame || frame.length === 0) return 0;
  let sumSq = 0;
  for (let i = 0; i < frame.length; i++) {
    sumSq += frame[i] * frame[i];
  }
  return Math.sqrt(sumSq / frame.length);
}

// src/onnx-splitter.js
var textDecoder = new TextDecoder();
var textEncoder = new TextEncoder();
function readVarint(buf, pos) {
  let result = 0;
  let shift = 0;
  let b = 0;
  let offset = pos;
  do {
    if (offset >= buf.length) throw new Error("Unexpected EOF reading protobuf varint");
    b = buf[offset++];
    result += (b & 127) * Math.pow(2, shift);
    shift += 7;
  } while (b & 128);
  return [result, offset];
}
function encodeVarint(val) {
  const out = [];
  let v = val;
  while (v >= 128) {
    out.push(v & 127 | 128);
    v = Math.floor(v / 128);
  }
  out.push(v & 127);
  return new Uint8Array(out);
}
function skipField(buf, pos, wireType) {
  if (wireType === 0) return readVarint(buf, pos)[1];
  if (wireType === 1) return pos + 8;
  if (wireType === 2) {
    const [len, nextPos] = readVarint(buf, pos);
    return nextPos + len;
  }
  if (wireType === 5) return pos + 4;
  throw new Error(`Unsupported protobuf wireType=${wireType} at offset=${pos}`);
}
function encodeLengthDelimitedField(fieldNumber, payloadBytes) {
  const tag = encodeVarint(fieldNumber << 3 | 2);
  const len = encodeVarint(payloadBytes.length);
  const out = new Uint8Array(tag.length + len.length + payloadBytes.length);
  out.set(tag, 0);
  out.set(len, tag.length);
  out.set(payloadBytes, tag.length + len.length);
  return out;
}
function concatViews(views) {
  let total = 0;
  for (const v of views) total += v.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const v of views) {
    out.set(v, off);
    off += v.byteLength;
  }
  return out;
}
function parseNodeProto(buf, start, end) {
  const inputs = [];
  const outputs = [];
  let pos = start;
  while (pos < end) {
    const [tag, p1] = readVarint(buf, pos);
    const fieldNum = tag >>> 3;
    const wireType = tag & 7;
    if ((fieldNum === 1 || fieldNum === 2) && wireType === 2) {
      const [len, p2] = readVarint(buf, p1);
      const str = textDecoder.decode(buf.subarray(p2, p2 + len));
      if (fieldNum === 1) inputs.push(str);
      else outputs.push(str);
      pos = p2 + len;
    } else {
      pos = skipField(buf, p1, wireType);
    }
  }
  return { inputs, outputs, rawFieldSlice: null };
}
function parseNamedEntry(buf, start, end, nameField = 1) {
  let pos = start;
  while (pos < end) {
    const [tag, p1] = readVarint(buf, pos);
    const fieldNum = tag >>> 3;
    const wireType = tag & 7;
    if (fieldNum === nameField && wireType === 2) {
      const [len, p2] = readVarint(buf, p1);
      return textDecoder.decode(buf.subarray(p2, p2 + len));
    }
    pos = skipField(buf, p1, wireType);
  }
  return "";
}
function synthesizeFloatTensorValueInfo(name, dims) {
  const dimEntries = dims.map((d) => {
    if (typeof d === "number") {
      const tag = encodeVarint(1 << 3 | 0);
      const val = encodeVarint(d);
      return encodeLengthDelimitedField(1, concatViews([tag, val]));
    }
    return encodeLengthDelimitedField(1, encodeLengthDelimitedField(2, textEncoder.encode(String(d))));
  });
  const shapeBytes = concatViews(dimEntries);
  const elemTypeBytes = concatViews([encodeVarint(1 << 3 | 0), encodeVarint(1)]);
  const tensorTypeBytes = concatViews([elemTypeBytes, encodeLengthDelimitedField(2, shapeBytes)]);
  const typeProtoBytes = encodeLengthDelimitedField(1, tensorTypeBytes);
  const nameFieldBytes = encodeLengthDelimitedField(1, textEncoder.encode(name));
  return concatViews([nameFieldBytes, encodeLengthDelimitedField(2, typeProtoBytes)]);
}
function pruneSubgraph(graphParsed, targetOutputs, stopAtInputs, extraInputs = [], extraOutputs = [], graphName = "sub") {
  const stopSet = new Set(stopAtInputs);
  const neededTensors = new Set(targetOutputs);
  const keptNodeIndices = /* @__PURE__ */ new Set();
  for (let i = graphParsed.nodes.length - 1; i >= 0; i--) {
    const node = graphParsed.nodes[i];
    if (node.outputs.some((o) => neededTensors.has(o))) {
      keptNodeIndices.add(i);
      for (const inp of node.inputs) {
        if (inp && !stopSet.has(inp)) {
          neededTensors.add(inp);
        }
      }
    }
  }
  const pieces = [];
  pieces.push(encodeLengthDelimitedField(2, textEncoder.encode(graphName)));
  for (let i = 0; i < graphParsed.nodes.length; i++) {
    if (keptNodeIndices.has(i)) {
      pieces.push(graphParsed.nodes[i].rawFieldSlice);
    }
  }
  for (const init of graphParsed.initializers) {
    if (neededTensors.has(init.name)) {
      pieces.push(init.rawFieldSlice);
    }
  }
  const addedInputs = /* @__PURE__ */ new Set();
  for (const inpName of stopAtInputs) {
    const orig = graphParsed.inputs.get(inpName) || graphParsed.valueInfos.get(inpName);
    if (orig) {
      pieces.push(encodeLengthDelimitedField(11, orig.payloadSlice));
      addedInputs.add(inpName);
    }
  }
  for (const ex of extraInputs) {
    if (!addedInputs.has(ex.name)) {
      pieces.push(encodeLengthDelimitedField(11, synthesizeFloatTensorValueInfo(ex.name, ex.dims)));
      addedInputs.add(ex.name);
    }
  }
  const addedOutputs = /* @__PURE__ */ new Set();
  for (const outName of targetOutputs) {
    const orig = graphParsed.outputs.get(outName) || graphParsed.valueInfos.get(outName);
    if (orig) {
      pieces.push(encodeLengthDelimitedField(12, orig.payloadSlice));
      addedOutputs.add(outName);
    }
  }
  for (const ex of extraOutputs) {
    if (!addedOutputs.has(ex.name)) {
      pieces.push(encodeLengthDelimitedField(12, synthesizeFloatTensorValueInfo(ex.name, ex.dims)));
      addedOutputs.add(ex.name);
    }
  }
  const graphPayload = concatViews(pieces);
  return concatViews([...graphParsed.outerFields, encodeLengthDelimitedField(7, graphPayload)]);
}
function splitOnnxGraph(onnxBytes, cutTensor = "acoustic_features") {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const buf = onnxBytes instanceof Uint8Array ? onnxBytes : new Uint8Array(onnxBytes);
  if (buf.length < 64 || buf[0] !== 8) {
    throw new Error("Invalid ONNX ModelProto buffer");
  }
  const outerFields = [];
  let graphStart = -1;
  let graphEnd = -1;
  let pos = 0;
  while (pos < buf.length) {
    const fieldStart = pos;
    const [tag, p1] = readVarint(buf, pos);
    const fieldNum = tag >>> 3;
    const wireType = tag & 7;
    if (fieldNum === 7 && wireType === 2) {
      const [len, p2] = readVarint(buf, p1);
      graphStart = p2;
      graphEnd = p2 + len;
      pos = graphEnd;
    } else {
      const nextPos = skipField(buf, p1, wireType);
      outerFields.push(buf.subarray(fieldStart, nextPos));
      pos = nextPos;
    }
  }
  if (graphStart < 0) {
    throw new Error("ONNX ModelProto missing GraphProto (field 7)");
  }
  const nodes = [];
  const initializers = [];
  const inputs = /* @__PURE__ */ new Map();
  const outputs = /* @__PURE__ */ new Map();
  const valueInfos = /* @__PURE__ */ new Map();
  pos = graphStart;
  while (pos < graphEnd) {
    const entryStart = pos;
    const [tag, p1] = readVarint(buf, pos);
    const fieldNum = tag >>> 3;
    const wireType = tag & 7;
    if (wireType === 2) {
      const [len, p2] = readVarint(buf, p1);
      const pEnd = p2 + len;
      const rawFieldSlice = buf.subarray(entryStart, pEnd);
      const payloadSlice = buf.subarray(p2, pEnd);
      if (fieldNum === 1) {
        const parsed = parseNodeProto(buf, p2, pEnd);
        parsed.rawFieldSlice = rawFieldSlice;
        nodes.push(parsed);
      } else if (fieldNum === 5) {
        const name = parseNamedEntry(buf, p2, pEnd, 8);
        initializers.push({ name, rawFieldSlice });
      } else if (fieldNum === 11) {
        const name = parseNamedEntry(buf, p2, pEnd, 1);
        inputs.set(name, { name, payloadSlice });
      } else if (fieldNum === 12) {
        const name = parseNamedEntry(buf, p2, pEnd, 1);
        outputs.set(name, { name, payloadSlice });
      } else if (fieldNum === 13) {
        const name = parseNamedEntry(buf, p2, pEnd, 1);
        valueInfos.set(name, { name, payloadSlice });
      }
      pos = pEnd;
    } else {
      pos = skipField(buf, p1, wireType);
    }
  }
  const graphParsed = { outerFields, nodes, initializers, inputs, outputs, valueInfos };
  const encoderBytes = pruneSubgraph(
    graphParsed,
    [cutTensor],
    ["input"],
    [{ name: "input", dims: ["batch", 576] }],
    [{ name: cutTensor, dims: ["batch", 128] }],
    "wasm_vad_acoustic_encoder"
  );
  const recurrentBytes = pruneSubgraph(
    graphParsed,
    ["output", "stateN"],
    [cutTensor, "state"],
    [
      { name: cutTensor, dims: ["batch", 128] },
      { name: "state", dims: [2, "batch", 128] }
    ],
    [
      { name: "output", dims: ["batch", 1] },
      { name: "stateN", dims: [2, "batch", 128] }
    ],
    "wasm_vad_recurrent_head"
  );
  const splitMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
  return {
    encoderBytes,
    recurrentBytes,
    splitMs,
    stats: {
      originalBytes: buf.byteLength,
      encoderBytes: encoderBytes.byteLength,
      recurrentBytes: recurrentBytes.byteLength,
      totalNodes: nodes.length
    }
  };
}

// src/cache.js
var CACHE_NAME = "wasm-vad-onnx-v1";
var IDB_NAME = "wasm-vad-idb-v1";
var IDB_STORE = "models";
var MIN_ONNX_BYTES = 16384;
function isValidOnnxBuffer(bytes) {
  return bytes instanceof Uint8Array && bytes.byteLength >= MIN_ONNX_BYTES && bytes[0] === 8;
}
function openIdb() {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
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
      const tx = db.transaction(IDB_STORE, "readonly");
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
      const tx = db.transaction(IDB_STORE, "readwrite");
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
  if (typeof caches === "undefined") return null;
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
  if (typeof caches === "undefined") return;
  try {
    const cache = await caches.open(CACHE_NAME);
    const res = new Response(bytes, {
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Length": String(bytes.byteLength)
      }
    });
    await cache.put(url, res);
  } catch {
  }
}
async function fetchAndCacheModel(url, { onProgress, cache = true } = {}) {
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
  const total = Number(res.headers.get("content-length")) || 0;
  let bytes;
  if (res.body && typeof res.body.getReader === "function" && onProgress) {
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
        percent: total > 0 ? Math.min(100, Math.round(loaded / total * 100)) : 0,
        cached: false
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

// src/spectral-vad.js
var FFT_N = 512;
var NUM_BINS = 257;
var HANN_WIN = new Float32Array(FFT_N);
var BIT_REV = new Uint16Array(FFT_N);
var TWIDDLE_COS = new Float32Array(FFT_N / 2);
var TWIDDLE_SIN = new Float32Array(FFT_N / 2);
for (let i = 0; i < FFT_N; i++) {
  HANN_WIN[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (FFT_N - 1)));
  let rev = 0;
  let x = i;
  for (let b = 0; b < 9; b++) {
    rev = rev << 1 | x & 1;
    x >>>= 1;
  }
  BIT_REV[i] = rev;
}
for (let k = 0; k < FFT_N / 2; k++) {
  const ang = -2 * Math.PI * k / FFT_N;
  TWIDDLE_COS[k] = Math.cos(ang);
  TWIDDLE_SIN[k] = Math.sin(ang);
}
function fft512RealPower(frame, reBuf, imBuf, psdOut) {
  for (let i = 0; i < FFT_N; i++) {
    reBuf[BIT_REV[i]] = (frame[i] || 0) * HANN_WIN[i];
    imBuf[BIT_REV[i]] = 0;
  }
  for (let size = 2; size <= FFT_N; size <<= 1) {
    const half = size >>> 1;
    const step = FFT_N / size;
    for (let i = 0; i < FFT_N; i += size) {
      let tw = 0;
      for (let j = 0; j < half; j++) {
        const wr = TWIDDLE_COS[tw];
        const wi = TWIDDLE_SIN[tw];
        tw += step;
        const idxA = i + j;
        const idxB = idxA + half;
        const tr = wr * reBuf[idxB] - wi * imBuf[idxB];
        const ti = wr * imBuf[idxB] + wi * reBuf[idxB];
        reBuf[idxB] = reBuf[idxA] - tr;
        imBuf[idxB] = imBuf[idxA] - ti;
        reBuf[idxA] += tr;
        imBuf[idxA] += ti;
      }
    }
  }
  for (let k = 0; k < NUM_BINS; k++) {
    psdOut[k] = reBuf[k] * reBuf[k] + imBuf[k] * imBuf[k] + 1e-12;
  }
}
var SpectralVadEngine = class {
  constructor() {
    this.conditioner = new PreConditioner16k({ highPassHz: 80, humNotch: true });
    this.reBuf = new Float32Array(FFT_N);
    this.imBuf = new Float32Array(FFT_N);
    this.psd = new Float32Array(NUM_BINS);
    this.smoothPsd = new Float32Array(NUM_BINS);
    this.noisePsd = new Float32Array(NUM_BINS);
    this.minTrack = new Float32Array(NUM_BINS);
    this.prevGainSqPsd = new Float32Array(NUM_BINS);
    this.prevProb = 0;
    this.frameCount = 0;
    this.reset();
  }
  reset() {
    this.conditioner.reset();
    this.noisePsd.fill(1e-5);
    this.smoothPsd.fill(1e-5);
    this.minTrack.fill(1e-5);
    this.prevGainSqPsd.fill(0);
    this.prevProb = 0;
    this.frameCount = 0;
  }
  /**
   * Scores a single 512-sample 16 kHz frame and returns `{ probability, snrDb }`.
   */
  scoreFrame(rawFrame) {
    const clean = this.conditioner.process(
      rawFrame.length === FRAME_SAMPLES ? rawFrame : rawFrame.subarray(0, FRAME_SAMPLES)
    );
    let energySq = 0;
    let zcr = 0;
    for (let i = 0; i < clean.length; i++) {
      energySq += clean[i] * clean[i];
      if (i > 0 && clean[i] >= 0 !== clean[i - 1] >= 0) zcr++;
    }
    const rms = Math.sqrt(energySq / Math.max(1, clean.length));
    const zcrRate = zcr / Math.max(1, clean.length);
    if (rms < 4e-4) {
      this.prevProb = this.prevProb * 0.65;
      this.frameCount++;
      return { probability: this.prevProb, snrDb: 0 };
    }
    fft512RealPower(clean, this.reBuf, this.imBuf, this.psd);
    let totalBandEnergy = 1e-12;
    let formantEnergy = 0;
    let fricativeEnergy = 0;
    let logSumFormant = 0;
    let logSumFull = 0;
    let formantBins = 0;
    let fullBins = 0;
    let maxBinPower = 0;
    let maxBinIdx = 9;
    for (let k = 3; k <= 224; k++) {
      const p = this.psd[k];
      totalBandEnergy += p;
      if (k >= 6) {
        logSumFull += Math.log(p);
        fullBins++;
      }
      if (k >= 9 && k <= 112) {
        formantEnergy += p;
        logSumFormant += Math.log(p);
        formantBins++;
        if (p > maxBinPower) {
          maxBinPower = p;
          maxBinIdx = k;
        }
      } else if (k > 112) {
        fricativeEnergy += p;
      }
    }
    const sfm = Math.min(
      1,
      Math.exp(logSumFormant / Math.max(1, formantBins)) / Math.max(formantEnergy / Math.max(1, formantBins), 1e-12)
    );
    const fullSfm = Math.min(
      1,
      Math.exp(logSumFull / Math.max(1, fullBins)) / Math.max(totalBandEnergy / Math.max(1, fullBins), 1e-12)
    );
    if (this.frameCount === 0) {
      const looksLikeSpeechAtZero = sfm < 0.36 && fullSfm < 0.32;
      for (let k = 0; k < NUM_BINS; k++) {
        this.smoothPsd[k] = this.psd[k];
        this.noisePsd[k] = looksLikeSpeechAtZero ? Math.min(this.psd[k], 5e-3) : this.psd[k] * 1.15;
        this.minTrack[k] = this.noisePsd[k];
      }
    }
    let llrSum = 0;
    let postSnrLinSum = 0;
    let llrCount = 0;
    const alphaDd = 0.92;
    for (let k = 6; k <= 180; k++) {
      const yk = this.psd[k];
      const nk = Math.max(this.noisePsd[k], 1e-9);
      const gamma = Math.min(40, yk / nk);
      const xi = alphaDd * (this.prevGainSqPsd[k] / nk) + (1 - alphaDd) * Math.max(0, gamma - 1);
      const v = xi * gamma / (1 + xi);
      const gain = xi / (1 + xi);
      this.prevGainSqPsd[k] = gain * gain * yk;
      const llr = v - Math.log(1 + xi);
      llrSum += llr;
      postSnrLinSum += gamma;
      llrCount++;
    }
    const meanLlr = llrSum / llrCount;
    const meanPostSnr = postSnrLinSum / llrCount;
    const snrDb = Math.max(0, 10 * Math.log10(Math.max(1, meanPostSnr)));
    const peak3Power = this.psd[maxBinIdx - 1] + this.psd[maxBinIdx] + this.psd[maxBinIdx + 1];
    const peak3Dominance = peak3Power / Math.max(formantEnergy, 1e-12);
    const formantRatio = formantEnergy / totalBandEnergy;
    const fricativeRatio = fricativeEnergy / totalBandEnergy;
    const isPureTone = peak3Dominance > 0.84;
    const isFlatNoise = fullSfm > 0.48 || sfm > 0.56;
    const isUnvoicedConsonant = !isFlatNoise && fricativeRatio > 0.45 && fricativeEnergy > 1.8 * formantEnergy && zcrRate > 0.14 && rms > 4e-3;
    let logit = -2.45;
    logit += 1.45 * Math.min(4.5, meanLlr);
    logit += 2.15 * Math.max(0, formantRatio - 0.35);
    logit += 2.55 * Math.max(0, 0.5 - sfm);
    if (rms > 0.012 && !isFlatNoise && !isPureTone) {
      logit += 0.5 * Math.min(1.5, Math.log10(rms / 8e-3 + 1) * 2);
    }
    if (isUnvoicedConsonant) logit += 1.1;
    if (isPureTone) logit -= 3.6;
    if (isFlatNoise) logit -= 3.1;
    const rawProb = 1 / (1 + Math.exp(-logit));
    const speechPresence = rawProb;
    const alphaNoise = 0.88 + 0.115 * speechPresence;
    for (let k = 0; k < NUM_BINS; k++) {
      this.smoothPsd[k] = 0.7 * this.smoothPsd[k] + 0.3 * this.psd[k];
      if (this.frameCount % 45 === 0) {
        this.minTrack[k] = Math.min(this.smoothPsd[k], this.minTrack[k] * 1.35 + 1e-9);
      } else {
        this.minTrack[k] = Math.min(this.minTrack[k], this.smoothPsd[k]);
      }
      const candidateNoise = speechPresence < 0.42 ? this.psd[k] : Math.min(this.noisePsd[k], this.minTrack[k] * 1.6);
      this.noisePsd[k] = alphaNoise * this.noisePsd[k] + (1 - alphaNoise) * candidateNoise;
    }
    const smoothProb = rawProb >= this.prevProb ? 0.72 * rawProb + 0.28 * this.prevProb : 0.42 * rawProb + 0.58 * this.prevProb;
    this.prevProb = Math.max(0, Math.min(1, smoothProb));
    this.frameCount++;
    return { probability: this.prevProb, snrDb };
  }
};

// src/hysteresis.js
var FRAME_MS = FRAME_SAMPLES / TARGET_SAMPLE_RATE * 1e3;
var DEFAULT_VAD_OPTIONS = {
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  minSpeechMs: 250,
  redemptionMs: 320,
  preSpeechPadMs: 128,
  maxSpeechMs: Infinity
};
var VadHysteresisArbiter = class {
  constructor(options = {}) {
    const opts = { ...DEFAULT_VAD_OPTIONS, ...options };
    if (opts.threshold !== void 0 && options.positiveSpeechThreshold === void 0) {
      opts.positiveSpeechThreshold = opts.threshold;
      opts.negativeSpeechThreshold = Math.max(0.05, opts.threshold - 0.15);
    }
    this.positiveSpeechThreshold = opts.positiveSpeechThreshold;
    this.negativeSpeechThreshold = opts.negativeSpeechThreshold;
    this.minSpeechFrames = Math.max(1, Math.round(opts.minSpeechMs / FRAME_MS));
    this.redemptionFrames = Math.max(1, Math.round(opts.redemptionMs / FRAME_MS));
    this.preSpeechPadFrames = Math.max(0, Math.round(opts.preSpeechPadMs / FRAME_MS));
    this.maxSpeechFrames = Number.isFinite(opts.maxSpeechMs) ? Math.max(this.minSpeechFrames, Math.round(opts.maxSpeechMs / FRAME_MS)) : Infinity;
    this.reset();
  }
  reset() {
    this.speaking = false;
    this.frameIndex = 0;
    this.redemptionCount = 0;
    this.preRoll = [];
    this.activeFrames = [];
    this.activeProbs = [];
    this.activeSnrs = [];
    this.speechStartFrame = 0;
  }
  /**
   * Processes one 512-sample frame with its model probability `prob` and optional `snrDb`.
   * Returns `{ event: null | 'speechStart' | 'speechEnd', frameInfo, segment }`.
   */
  step(framePcm, prob, snrDb = 0) {
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
        event = "speechStart";
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
        event = "speechEnd";
      } else if (prob < this.negativeSpeechThreshold) {
        this.redemptionCount++;
        if (this.redemptionCount >= this.redemptionFrames) {
          const trimTail = Math.max(0, this.redemptionCount - 2);
          committedSegment = this._finalizeSegment(idx + 1 - trimTail, trimTail);
          if (committedSegment) {
            event = "speechEnd";
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
      snrDb
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
    const tailFrames = this.activeFrames.slice(keepCount);
    this.speaking = false;
    this.redemptionCount = 0;
    this.activeFrames = [];
    this.activeProbs = [];
    this.activeSnrs = [];
    this.preRoll = tailFrames.slice(-this.preSpeechPadFrames).map((pcm2, i) => ({
      frameIndex: endFrame + i,
      pcm: pcm2
    }));
    if (coreSpeechFrames < this.minSpeechFrames || framesToConcat.length === 0) {
      return null;
    }
    const totalSamples = framesToConcat.length * FRAME_SAMPLES;
    const pcm = new Float32Array(totalSamples);
    let peakRms = 0;
    for (let i = 0; i < framesToConcat.length; i++) {
      pcm.set(framesToConcat[i], i * FRAME_SAMPLES);
      const r = computeRms(framesToConcat[i]);
      if (r > peakRms) peakRms = r;
    }
    const confidence = probs.reduce((a, b) => a + b, 0) / Math.max(1, probs.length);
    const snrDb = snrs.reduce((a, b) => a + b, 0) / Math.max(1, snrs.length);
    const startSample = startFrame * FRAME_SAMPLES;
    const endSample = startSample + totalSamples;
    const startMs = Math.round(startSample / TARGET_SAMPLE_RATE * 1e3);
    const endMs = Math.round(endSample / TARGET_SAMPLE_RATE * 1e3);
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
      }
    };
    return seg;
  }
};
function buildContextWindows(pcm16k, initialContext = null) {
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

// src/worklet.js
var WORKLET_PROCESSOR_SOURCE = (
  /* js */
  `
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
`
);
async function startAudioWorkletCapture(detector, streamOrOptions = {}) {
  if (typeof window === "undefined" || typeof AudioContext === "undefined") {
    throw new Error("detector.listen() requires a browser environment with Web Audio API & AudioWorklet");
  }
  const isMediaStream = streamOrOptions && typeof streamOrOptions.getAudioTracks === "function";
  const opts = isMediaStream ? { stream: streamOrOptions } : streamOrOptions || {};
  let ownStream = false;
  let mediaStream = opts.stream;
  if (!mediaStream) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("navigator.mediaDevices.getUserMedia is unavailable (requires HTTPS or localhost)");
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: opts.echoCancellation !== false,
        noiseSuppression: opts.noiseSuppression ?? false,
        autoGainControl: opts.autoGainControl !== false
      }
    });
    ownStream = true;
  }
  const audioCtx = opts.audioContext || new AudioContext({ latencyHint: "interactive" });
  if (audioCtx.state === "suspended") {
    await audioCtx.resume();
  }
  const blob = new Blob([WORKLET_PROCESSOR_SOURCE], { type: "application/javascript" });
  const blobUrl = URL.createObjectURL(blob);
  try {
    await audioCtx.audioWorklet.addModule(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
  const sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  const workletNode = new AudioWorkletNode(audioCtx, "wasm-vad-capture", {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    channelCount: 1
  });
  let paused = false;
  let closed = false;
  let processingChain = Promise.resolve();
  workletNode.port.onmessage = (ev) => {
    const msg = ev.data;
    if (!msg || msg.type !== "frame16k" || !(msg.pcm instanceof Float32Array)) return;
    const pcmFrame = msg.pcm;
    if (paused || closed) {
      workletNode.port.postMessage({ type: "recycle", buffer: pcmFrame.buffer }, [pcmFrame.buffer]);
      return;
    }
    processingChain = processingChain.then(async () => {
      if (closed) return;
      const res = await detector.frame(pcmFrame, { preconditioned: true });
      if (opts.onFrame) opts.onFrame(res);
    }).catch((err) => {
      if (opts.onError) opts.onError(err);
    }).finally(() => {
      if (!closed) {
        try {
          workletNode.port.postMessage({ type: "recycle", buffer: pcmFrame.buffer }, [pcmFrame.buffer]);
        } catch {
        }
      }
    });
  };
  sourceNode.connect(workletNode);
  const unsubStart = opts.onSpeechStart ? detector.on("speechStart", opts.onSpeechStart) : null;
  const unsubEnd = opts.onSpeechEnd ? detector.on("speechEnd", opts.onSpeechEnd) : null;
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
        workletNode.port.postMessage({ type: "stop" });
        sourceNode.disconnect();
        workletNode.disconnect();
      } catch {
      }
      const flushSeg = detector.flush();
      if (flushSeg && opts.onSpeechEnd) {
        opts.onSpeechEnd(flushSeg);
      }
      if (ownStream) {
        for (const track of mediaStream.getTracks()) track.stop();
      }
      if (!opts.audioContext) {
        await audioCtx.close().catch(() => {
        });
      }
    }
  };
}

// src/index.js
var CDN_MODEL_BASE = "https://cdn.jsdelivr.net/npm/wasm-vad@0.1.1/models/";
function isAudioInput(val) {
  if (!val) return false;
  if (typeof val === "string") return true;
  if (val instanceof Float32Array || val instanceof Int16Array || val instanceof Uint8Array || val instanceof ArrayBuffer) {
    return true;
  }
  if (typeof val === "object") {
    if (typeof val.getChannelData === "function") return true;
    if (val.pcm instanceof Float32Array || val.pcm instanceof Int16Array) return true;
  }
  return false;
}
function int16ToFloat32(int16) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    out[i] = int16[i] / 32768;
  }
  return out;
}
function hasRiffHeader(bytes) {
  return bytes.byteLength >= 12 && bytes[0] === 82 && // R
  bytes[1] === 73 && // I
  bytes[2] === 70 && // F
  bytes[3] === 70;
}
async function normalizeAudioTo16k(input, sampleRateHint = TARGET_SAMPLE_RATE) {
  const isNode = typeof process !== "undefined" && Boolean(process.versions?.node) && typeof window === "undefined";
  if (typeof input === "string") {
    let bytes;
    if (isNode && !/^https?:\/\//i.test(input)) {
      const fs = await import("node:fs/promises");
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
  if (input && typeof input.getChannelData === "function") {
    const numCh = input.numberOfChannels || 1;
    const len = input.length;
    const mono = new Float32Array(len);
    for (let c = 0; c < numCh; c++) {
      const ch = input.getChannelData(c);
      for (let i = 0; i < len; i++) mono[i] += ch[i] / numCh;
    }
    return resample16k(mono, input.sampleRate || TARGET_SAMPLE_RATE);
  }
  if (input && typeof input === "object" && input.pcm) {
    const raw = input.pcm instanceof Int16Array ? int16ToFloat32(input.pcm) : input.pcm;
    return resample16k(raw, input.sampleRate || sampleRateHint);
  }
  throw new TypeError(
    "Unsupported audio input: expected WAV path/URL, ArrayBuffer, Uint8Array, Float32Array, Int16Array, AudioBuffer, or { pcm, sampleRate }"
  );
}
async function loadOrtRuntime(caps, userOrt, wasmPaths) {
  if (caps.backend === "dsp") return { ort: null, caps };
  const isNodeEnv = typeof process !== "undefined" && Boolean(process.versions?.node) && typeof window === "undefined";
  if (userOrt) {
    if (isNodeEnv && userOrt?.env?.wasm) {
      userOrt.env.wasm.numThreads = 1;
      userOrt.env.wasm.simd = true;
    }
    return { ort: userOrt, caps };
  }
  if (typeof globalThis.ort !== "undefined" && globalThis.ort?.InferenceSession) {
    return { ort: globalThis.ort, caps };
  }
  if (isNodeEnv) {
    if (caps.backend !== "wasm") {
      try {
        const mod = await import("onnxruntime-node");
        return { ort: mod.default || mod, caps };
      } catch {
      }
    }
    try {
      const webMod = await import("onnxruntime-web");
      const ort = webMod.default || webMod;
      if (ort?.env?.wasm) {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;
      }
      const updatedCaps = {
        ...caps,
        backend: "wasm",
        executionProviders: ["wasm"],
        numThreads: 1,
        sessionOptions: {
          executionProviders: ["wasm"],
          enableCpuMemArena: false,
          enableMemPattern: false,
          graphOptimizationLevel: "all"
        }
      };
      return { ort, caps: updatedCaps };
    } catch {
      return { ort: null, caps };
    }
  } else {
    try {
      const webMod = await import("onnxruntime-web");
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
  const modelFile = caps.tier === "mobile" ? "vad-mobile.onnx" : "vad-studio.onnx";
  const isNode = typeof process !== "undefined" && Boolean(process.versions?.node) && typeof window === "undefined";
  if (isNode && (!options.modelUrl || !/^https?:\/\//i.test(options.modelUrl))) {
    const fs = await import("node:fs/promises");
    const path = await import("node:path");
    const urlMod = await import("node:url");
    if (options.modelUrl) {
      return new Uint8Array(await fs.readFile(options.modelUrl));
    }
    const candidates = [];
    if (typeof __dirname === "string") {
      candidates.push(path.resolve(__dirname, "..", "models", modelFile));
      candidates.push(path.resolve(__dirname, "models", modelFile));
    }
    try {
      if (import.meta && import.meta.url) {
        const here = path.dirname(urlMod.fileURLToPath(import.meta.url));
        candidates.push(path.resolve(here, "..", "models", modelFile));
        candidates.push(path.resolve(here, "models", modelFile));
      }
    } catch {
    }
    candidates.push(path.resolve(process.cwd(), "models", modelFile));
    for (const candidate of candidates) {
      try {
        return new Uint8Array(await fs.readFile(candidate));
      } catch {
      }
    }
    throw new Error(`Could not locate bundled ONNX checkpoint "${modelFile}" in [${candidates.join(", ")}]`);
  }
  let url = options.modelUrl;
  if (!url) {
    try {
      if (import.meta && import.meta.url && /^https?:\/\//i.test(import.meta.url)) {
        url = new URL(`../models/${modelFile}`, import.meta.url).href;
      }
    } catch {
    }
    if (!url) {
      url = `${CDN_MODEL_BASE}${modelFile}`;
    }
  }
  try {
    return await fetchAndCacheModel(url, {
      onProgress: options.onProgress,
      cache: options.cache !== false
    });
  } catch (primaryErr) {
    if (!options.modelUrl && url !== `${CDN_MODEL_BASE}${modelFile}`) {
      return await fetchAndCacheModel(`${CDN_MODEL_BASE}${modelFile}`, {
        onProgress: options.onProgress,
        cache: options.cache !== false
      });
    }
    throw primaryErr;
  }
}
var OnnxSplitBackend = class {
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
    this.context.fill(0);
    this.state.fill(0);
  }
  async scoreSingleFrame(framePcm, preconditioned = false) {
    const clean = preconditioned ? framePcm : this.conditioner.process(framePcm);
    const { snrDb } = this.spectralSnr.scoreFrame(clean);
    this.windowBuf.set(this.context, 0);
    this.windowBuf.set(clean, CONTEXT_SAMPLES);
    this.context.set(clean.subarray(FRAME_SAMPLES - CONTEXT_SAMPLES));
    const encOut = await this.encoderSession.run({
      input: new this.ort.Tensor("float32", this.windowBuf, [1, WINDOW_SAMPLES])
    });
    const feats = encOut.acoustic_features.data;
    const recOut = await this.recurrentSession.run({
      acoustic_features: new this.ort.Tensor("float32", feats, [1, 128]),
      state: new this.ort.Tensor("float32", this.state, [2, 1, 128])
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
    const BATCH_CHUNK = 2048;
    const probabilities = new Float32Array(numFrames);
    const snrs = new Float32Array(numFrames);
    const snrTracker = new SpectralVadEngine();
    let rnnState = new Float32Array(2 * 1 * 128);
    for (let start = 0; start < numFrames; start += BATCH_CHUNK) {
      const count = Math.min(BATCH_CHUNK, numFrames - start);
      const winSlice = windows.subarray(start * WINDOW_SAMPLES, (start + count) * WINDOW_SAMPLES);
      const encOut = await this.encoderSession.run({
        input: new this.ort.Tensor("float32", winSlice, [count, WINDOW_SAMPLES])
      });
      const feats = encOut.acoustic_features.data;
      for (let i = 0; i < count; i++) {
        const globalIdx = start + i;
        const frameSlice = cleanPcm.subarray(globalIdx * FRAME_SAMPLES, (globalIdx + 1) * FRAME_SAMPLES);
        snrs[globalIdx] = snrTracker.scoreFrame(frameSlice).snrDb;
        const featSlice = feats.subarray(i * 128, (i + 1) * 128);
        const recOut = await this.recurrentSession.run({
          acoustic_features: new this.ort.Tensor("float32", featSlice, [1, 128]),
          state: new this.ort.Tensor("float32", rnnState, [2, 1, 128])
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
      this.recurrentSession?.release?.()
    ]);
  }
};
var DspBackend = class {
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
  async dispose() {
  }
};
function createDetectorInstance(engine, info, baseOptions = {}) {
  const listeners = {
    frame: /* @__PURE__ */ new Set(),
    speechStart: /* @__PURE__ */ new Set(),
    speechEnd: /* @__PURE__ */ new Set()
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
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const { probability, snrDb } = await engine.scoreSingleFrame(frame512, preconditioned);
    const latencyMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
    const { event, frameInfo, segment } = streamArbiter.step(frame512, probability, snrDb);
    const enrichedFrame = { ...frameInfo, latencyMs: Number(latencyMs.toFixed(4)), event, segment };
    emit("frame", enrichedFrame);
    if (event === "speechStart") {
      emit("speechStart", { startMs: frameInfo.startMs, probability, snrDb });
    } else if (event === "speechEnd" && segment) {
      emit("speechEnd", segment);
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
      if (seg) emit("speechEnd", seg);
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
      return lastResult || {
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
        segment: null
      };
    },
    async *stream(asyncSource, opts = {}) {
      const localQueue = [];
      const unsub = detector.on("speechEnd", (seg) => localQueue.push(seg));
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
      const durationMs = Math.round(pcm16k.length / TARGET_SAMPLE_RATE * 1e3);
      const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
      const { probabilities, snrs } = await engine.scoreBatchPcm16k(pcm16k, mergedOpts.precondition !== false);
      const inferenceMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
      const arbiter = new VadHysteresisArbiter(mergedOpts);
      const segments = [];
      const frames = mergedOpts.returnFrames ? [] : null;
      for (let i = 0; i < probabilities.length; i++) {
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
          rtf: durationMs > 0 ? Number((inferenceMs / durationMs).toFixed(6)) : 0
        };
      }
      return segments;
    },
    async trim(audioInput, opts = {}) {
      const mergedOpts = { ...baseOptions, ...opts };
      const pcm16k = await normalizeAudioTo16k(audioInput, mergedOpts.sampleRate || TARGET_SAMPLE_RATE);
      const originalDurationMs = Math.round(pcm16k.length / TARGET_SAMPLE_RATE * 1e3);
      const segments = await detector.detect({ pcm: pcm16k, sampleRate: TARGET_SAMPLE_RATE }, mergedOpts);
      const overlapSamples = mergedOpts.overlapSamples ?? 128;
      const { pcm: trimmedPcm, seamIndices } = concatSegmentsHann(
        segments.map((s) => s.pcm),
        overlapSamples,
        64
      );
      const seamStats = verifySeamContinuity(trimmedPcm, seamIndices);
      const durationMs = Math.round(trimmedPcm.length / TARGET_SAMPLE_RATE * 1e3);
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
        seamStats
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
    }
  };
  return detector;
}
async function vad(inputOrOptions = {}, maybeOptions = {}) {
  const oneShot = isAudioInput(inputOrOptions);
  const options = oneShot ? maybeOptions || {} : inputOrOptions || {};
  let caps = detectCapabilities(options);
  const loaded = await loadOrtRuntime(caps, options.ort, options.wasmPaths);
  const ort = loaded.ort;
  caps = loaded.caps;
  if (!ort && caps.backend !== "dsp") {
    if (options.strictOnnx) {
      throw new Error(
        'ONNX Runtime not found: install `onnxruntime-web` (browser) or `onnxruntime-node` (Node.js), or pass `{ tier: "dsp" }`'
      );
    }
    caps = detectCapabilities({ ...options, tier: "dsp" });
  }
  let engine;
  let modelBytesLen = 0;
  let splitMs = 0;
  if (caps.backend === "dsp") {
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
        ort.InferenceSession.create(split.recurrentBytes, sessOpts)
      ]);
    } catch (err) {
      if (caps.backend === "webgpu") {
        const fallbackOpts = { ...sessOpts, executionProviders: ["wasm"] };
        [encoderSession, recurrentSession] = await Promise.all([
          ort.InferenceSession.create(split.encoderBytes, fallbackOpts),
          ort.InferenceSession.create(split.recurrentBytes, fallbackOpts)
        ]);
        caps = { ...caps, backend: "wasm", deviceClass: "desktop-wasm", executionProviders: ["wasm"] };
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
    splitMs
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
  CONTEXT_SAMPLES,
  DEFAULT_VAD_OPTIONS,
  FRAME_SAMPLES,
  PreConditioner16k,
  SpectralVadEngine,
  TARGET_SAMPLE_RATE,
  VadHysteresisArbiter,
  WINDOW_SAMPLES,
  computeRms,
  concatSegmentsHann,
  decodeWav,
  vad as default,
  detectCapabilities,
  encodeWav,
  fetchAndCacheModel,
  resample16k,
  splitOnnxGraph,
  vad,
  verifySeamContinuity
};
