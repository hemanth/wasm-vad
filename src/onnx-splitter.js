/**
 * Zero-Copy Binary Protobuf ONNX Graph Splitter in Browser / Node RAM (Section 4 of /browser-streaming-onnx).
 *
 * Scans a serialized ONNX `ModelProto` -> `GraphProto` wire buffer without copying float/int8 weight bytes,
 * performs backward reachability pruning across `cutPoint` (default: `'acoustic_features'`), and emits two
 * self-contained `Uint8Array` ONNX models in <3 ms:
 *   1. `encoderBytes`:   stateless batched `input [batch, 576]` -> `acoustic_features [batch, 128]`
 *   2. `recurrentBytes`: stateful step `acoustic_features [batch, 128] + state [2, batch, 128]` -> `output + stateN`
 */

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

function readVarint(buf, pos) {
  let result = 0;
  let shift = 0;
  let b = 0;
  let offset = pos;
  do {
    if (offset >= buf.length) throw new Error('Unexpected EOF reading protobuf varint');
    b = buf[offset++];
    result += (b & 0x7f) * Math.pow(2, shift);
    shift += 7;
  } while (b & 0x80);
  return [result, offset];
}

function encodeVarint(val) {
  const out = [];
  let v = val;
  while (v >= 0x80) {
    out.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v & 0x7f);
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
  const tag = encodeVarint((fieldNumber << 3) | 2);
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
  return '';
}

function synthesizeFloatTensorValueInfo(name, dims) {
  // ValueInfoProto: field 1 = name (string), field 2 = TypeProto
  // TypeProto: field 1 = Tensor (TypeProto.Tensor)
  // TypeProto.Tensor: field 1 = elem_type (1 = FLOAT), field 2 = TensorShapeProto
  // TensorShapeProto: repeated field 1 = Dimension (field 1 = dim_value varint, or field 2 = dim_param string)
  const dimEntries = dims.map((d) => {
    if (typeof d === 'number') {
      const tag = encodeVarint((1 << 3) | 0);
      const val = encodeVarint(d);
      return encodeLengthDelimitedField(1, concatViews([tag, val]));
    }
    return encodeLengthDelimitedField(1, encodeLengthDelimitedField(2, textEncoder.encode(String(d))));
  });
  const shapeBytes = concatViews(dimEntries);
  const elemTypeBytes = concatViews([encodeVarint((1 << 3) | 0), encodeVarint(1)]); // FLOAT = 1
  const tensorTypeBytes = concatViews([elemTypeBytes, encodeLengthDelimitedField(2, shapeBytes)]);
  const typeProtoBytes = encodeLengthDelimitedField(1, tensorTypeBytes);
  const nameFieldBytes = encodeLengthDelimitedField(1, textEncoder.encode(name));
  return concatViews([nameFieldBytes, encodeLengthDelimitedField(2, typeProtoBytes)]);
}

function pruneSubgraph(graphParsed, targetOutputs, stopAtInputs, extraInputs = [], extraOutputs = [], graphName = 'sub') {
  const stopSet = new Set(stopAtInputs);
  const neededTensors = new Set(targetOutputs);
  const keptNodeIndices = new Set();

  // Reverse topological backward reachability walk
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
  // GraphProto field 2 = name
  pieces.push(encodeLengthDelimitedField(2, textEncoder.encode(graphName)));

  // Kept NodeProto slices (field 1)
  for (let i = 0; i < graphParsed.nodes.length; i++) {
    if (keptNodeIndices.has(i)) {
      pieces.push(graphParsed.nodes[i].rawFieldSlice);
    }
  }

  // Kept Initializer TensorProto slices (field 5 — zero float copy, direct subarray view!)
  for (const init of graphParsed.initializers) {
    if (neededTensors.has(init.name)) {
      pieces.push(init.rawFieldSlice);
    }
  }

  // Graph Inputs (field 11)
  const addedInputs = new Set();
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

  // Graph Outputs (field 12)
  const addedOutputs = new Set();
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

/**
 * Splits a unified `vad-studio.onnx` or `vad-mobile.onnx` buffer into stateless batched
 * `encoderBytes` (`[B, 576] -> [B, 128]`) and stateful `recurrentBytes` (`[B, 128] + state -> output + stateN`).
 */
export function splitOnnxGraph(onnxBytes, cutTensor = 'acoustic_features') {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const buf = onnxBytes instanceof Uint8Array ? onnxBytes : new Uint8Array(onnxBytes);
  if (buf.length < 64 || buf[0] !== 0x08) {
    throw new Error('Invalid ONNX ModelProto buffer');
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
    throw new Error('ONNX ModelProto missing GraphProto (field 7)');
  }

  const nodes = [];
  const initializers = [];
  const inputs = new Map();
  const outputs = new Map();
  const valueInfos = new Map();

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
        // TensorProto.name is field 8 in onnx.proto!
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
    ['input'],
    [{ name: 'input', dims: ['batch', 576] }],
    [{ name: cutTensor, dims: ['batch', 128] }],
    'wasm_vad_acoustic_encoder'
  );

  const recurrentBytes = pruneSubgraph(
    graphParsed,
    ['output', 'stateN'],
    [cutTensor, 'state'],
    [
      { name: cutTensor, dims: ['batch', 128] },
      { name: 'state', dims: [2, 'batch', 128] },
    ],
    [
      { name: 'output', dims: ['batch', 1] },
      { name: 'stateN', dims: [2, 'batch', 128] },
    ],
    'wasm_vad_recurrent_head'
  );

  const splitMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
  return {
    encoderBytes,
    recurrentBytes,
    splitMs,
    stats: {
      originalBytes: buf.byteLength,
      encoderBytes: encoderBytes.byteLength,
      recurrentBytes: recurrentBytes.byteLength,
      totalNodes: nodes.length,
    },
  };
}
