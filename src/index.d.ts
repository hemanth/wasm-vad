export type VadTier = 'studio' | 'mobile' | 'dsp';
export type VadBackend = 'webgpu' | 'wasm' | 'onnx-node' | 'dsp';

export interface VadProgress {
  loaded: number;
  total: number;
  percent: number;
  cached: boolean;
}

export interface VadOptions {
  tier?: VadTier;
  backend?: VadBackend;
  preferWebGPU?: boolean;
  isMobile?: boolean;
  numThreads?: number;
  threshold?: number;
  positiveSpeechThreshold?: number;
  negativeSpeechThreshold?: number;
  minSpeechMs?: number;
  redemptionMs?: number;
  preSpeechPadMs?: number;
  maxSpeechMs?: number;
  sampleRate?: number;
  precondition?: boolean;
  modelUrl?: string;
  modelBytes?: Uint8Array | ArrayBuffer;
  wasmPaths?: string | Record<string, string>;
  cache?: boolean;
  strictOnnx?: boolean;
  ort?: unknown;
  onProgress?: (progress: VadProgress) => void;
}

export interface VadSegment {
  startMs: number;
  endMs: number;
  durationMs: number;
  startSample: number;
  endSample: number;
  confidence: number;
  peakRms: number;
  snrDb: number;
  pcm: Float32Array;
  readonly wav: ArrayBuffer;
}

export interface VadFrameResult {
  frameIndex: number;
  startMs: number;
  endMs: number;
  probability: number;
  speaking: boolean;
  rms: number;
  snrDb: number;
  latencyMs: number;
  event: 'speechStart' | 'speechEnd' | null;
  segment: VadSegment | null;
}

export interface VadDetailedResult {
  segments: VadSegment[];
  frames: Omit<VadFrameResult, 'latencyMs' | 'event' | 'segment'>[];
  probabilities: Float32Array;
  durationMs: number;
  inferenceMs: number;
  rtf: number;
}

export interface SeamContinuityStats {
  ok: boolean;
  maxSeamJump: number;
  p999InteriorJump: number;
  ratio: number;
}

export interface VadTrimResult {
  pcm: Float32Array;
  readonly wav: ArrayBuffer;
  sampleRate: 16000;
  durationMs: number;
  originalDurationMs: number;
  speechRatio: number;
  segments: VadSegment[];
  seamStats: SeamContinuityStats;
}

export interface VadListenOptions {
  stream?: MediaStream;
  audioContext?: AudioContext;
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
  onFrame?: (frame: VadFrameResult) => void;
  onSpeechStart?: (ev: { startMs: number; probability: number; snrDb: number }) => void;
  onSpeechEnd?: (segment: VadSegment) => void;
  onError?: (err: unknown) => void;
}

export interface VadListenController {
  stream: MediaStream;
  audioContext: AudioContext;
  pause(): void;
  resume(): void;
  stop(): Promise<void>;
}

export interface VadRuntimeInfo {
  readonly tier: VadTier;
  readonly backend: VadBackend;
  readonly deviceClass: string;
  readonly numThreads: number;
  readonly crossOriginIsolated: boolean;
  readonly isMobile: boolean;
  readonly modelBytes: number;
  readonly splitMs: number;
}

export type AudioInput =
  | string
  | Float32Array
  | Int16Array
  | Uint8Array
  | ArrayBuffer
  | AudioBuffer
  | { pcm: Float32Array | Int16Array; sampleRate?: number };

export interface VadDetector {
  readonly info: VadRuntimeInfo;
  detect(input: AudioInput, options?: VadOptions & { returnFrames?: false }): Promise<VadSegment[]>;
  detect(input: AudioInput, options: VadOptions & { returnFrames: true }): Promise<VadDetailedResult>;
  trim(input: AudioInput, options?: VadOptions & { overlapSamples?: number }): Promise<VadTrimResult>;
  frame(
    chunk: Float32Array | Int16Array,
    options?: { sampleRate?: number; preconditioned?: boolean }
  ): Promise<VadFrameResult>;
  stream(
    source: AsyncIterable<Float32Array | Int16Array>,
    options?: { sampleRate?: number }
  ): AsyncGenerator<VadSegment, void, unknown>;
  listen(streamOrOptions?: MediaStream | VadListenOptions): Promise<VadListenController>;
  on(event: 'frame', fn: (frame: VadFrameResult) => void): () => void;
  on(event: 'speechStart', fn: (ev: { startMs: number; probability: number; snrDb: number }) => void): () => void;
  on(event: 'speechEnd', fn: (segment: VadSegment) => void): () => void;
  off(event: 'frame' | 'speechStart' | 'speechEnd', fn: (...args: any[]) => void): void;
  flush(): VadSegment | null;
  reset(): void;
  dispose(): Promise<void>;
}

export default function vad(options?: VadOptions): Promise<VadDetector>;
export default function vad(input: AudioInput, options?: VadOptions): Promise<VadSegment[]>;

export { vad };
export function detectCapabilities(overrides?: Partial<VadOptions>): Record<string, unknown>;
export function splitOnnxGraph(
  onnxBytes: Uint8Array | ArrayBuffer,
  cutTensor?: string
): {
  encoderBytes: Uint8Array;
  recurrentBytes: Uint8Array;
  splitMs: number;
  stats: { originalBytes: number; encoderBytes: number; recurrentBytes: number; totalNodes: number };
};
export function decodeWav(buffer: ArrayBuffer | Uint8Array): {
  pcm: Float32Array;
  sampleRate: number;
  numChannels: number;
};
export function encodeWav(pcm: Float32Array, sampleRate?: number): ArrayBuffer;
export function resample16k(pcm: Float32Array, inputSampleRate?: number): Float32Array;
export function concatSegmentsHann(
  segments: Float32Array[],
  overlapSamples?: number,
  edgeTaperSamples?: number
): { pcm: Float32Array; seamIndices: number[] };
export function verifySeamContinuity(
  pcm: Float32Array,
  seamIndices: number[],
  windowRadius?: number
): SeamContinuityStats;
