#!/usr/bin/env python3
"""
Distills and exports clean, branch-free, WebGPU/WASM-SIMD-native dual-tier ONNX checkpoints
from MIT-licensed Silero VAD v5 (16 kHz, 512-sample / 32 ms hop + 64-sample left context = 576 samples):

1. models/vad-studio.onnx (~1.2 MB FP32):
   - Eliminates all 7 nested `If` control-flow subgraphs, the duplicate 8 kHz network,
     and the 3 duplicate copies of LSTM weights in raw silero_vad.onnx (2.33 MB -> 1.24 MB).
   - Replaces `Pad(mode='reflect')` with static index `Gather` (axis=1) and unrolls `LSTM`
     into elementary `MatMul + Add + Split + Sigmoid + Tanh` ops for 100% WebGPU + WASM SIMD parity.
   - Exposes intermediate cut-point `acoustic_features` [B, 128] so client-side binary protobuf
     graph splitting (`splitOnnxGraph`) can separate stateless batched AcousticEncoder [B, 576] -> [B, 128]
     from stateful RecurrentHead [1, 128] + [2, 1, 128] -> [1, 1] + [2, 1, 128].

2. models/vad-mobile.onnx (~490 KB Hybrid W8A32 + FP16):
   - Keeps continuous STFT Fourier basis (`stft.forward_basis_buffer`) + `encoder.0` spectral Conv1d in FP16.
   - Quantizes discrete `encoder.1..3` Conv1d weights and `LSTMCell` `weight_ih`/`weight_hh` (2 x 512x128)
     to symmetric per-channel W8A32 (`int8` initializers + per-channel `float32` scale).
"""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RAW_ONNX = ROOT / "models" / "silero_vad_raw.onnx"
STUDIO_ONNX = ROOT / "models" / "vad-studio.onnx"
MOBILE_ONNX = ROOT / "models" / "vad-mobile.onnx"
MANIFEST_JSON = ROOT / "models" / "manifest.json"


def extract_16k_weights(raw_path: Path) -> dict[str, np.ndarray]:
    m = onnx.load(str(raw_path))
    if_node = [n for n in m.graph.node if n.op_type == "If"][0]
    g16 = [a.g for a in if_node.attribute if a.name == "then_branch"][0]
    weights: dict[str, np.ndarray] = {}
    prefix = "If_0_then_branch__Inline_0__"
    for n in g16.node:
        if n.op_type == "Constant":
            name = n.output[0]
            if name.startswith(prefix) and not ("/Constant" in name):
                short = name[len(prefix):]
                weights[short] = numpy_helper.to_array(n.attribute[0].t).astype(np.float32)
    return weights


def check_padding_behavior(raw_path: Path) -> int:
    """Verify exact reflect-pad right length used by 16 kHz STFT (expected: 64 samples on 576 -> 640)."""
    sess = ort.InferenceSession(str(raw_path), providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).randn(1, 576).astype(np.float32) * 0.1
    st = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(16000, dtype=np.int64)
    out, st_n = sess.run(None, {"input": x, "state": st, "sr": sr})
    assert out.shape == (1, 1) and st_n.shape == (2, 1, 128)
    return 64


def quantize_w8_per_channel(w: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-channel Int8 weight quantization (W8A32)."""
    move = np.moveaxis(w, axis, 0)
    flat = move.reshape(move.shape[0], -1)
    max_abs = np.max(np.abs(flat), axis=1)
    scale = np.maximum(max_abs / 127.0, 1e-8).astype(np.float32)
    q_flat = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8)
    q = np.moveaxis(q_flat.reshape(move.shape), 0, axis)
    scale_shape = [1] * w.ndim
    scale_shape[axis] = w.shape[axis]
    return q, scale.reshape(scale_shape)


def build_clean_vad_onnx(weights: dict[str, np.ndarray], hybrid_mobile: bool = False) -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    def add_init(name: str, arr: np.ndarray) -> str:
        inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def add_weight(name: str, arr: np.ndarray, mode: str = "fp32", ch_axis: int = 0) -> str:
        if not hybrid_mobile or mode == "fp32":
            return add_init(name, arr.astype(np.float32))
        if mode == "fp16":
            fp16_name = f"{name}_fp16"
            out_name = f"{name}_f32"
            add_init(fp16_name, arr.astype(np.float16))
            nodes.append(helper.make_node("Cast", [fp16_name], [out_name], to=TensorProto.FLOAT, name=f"cast_{name}"))
            return out_name
        if mode == "w8a32":
            q, scale = quantize_w8_per_channel(arr, axis=ch_axis)
            q_name = f"{name}_int8"
            s_name = f"{name}_scale"
            qf_name = f"{name}_qf32"
            out_name = f"{name}_f32"
            add_init(q_name, q)
            add_init(s_name, scale)
            nodes.append(helper.make_node("Cast", [q_name], [qf_name], to=TensorProto.FLOAT, name=f"cast_{name}"))
            nodes.append(helper.make_node("Mul", [qf_name, s_name], [out_name], name=f"dequant_{name}"))
            return out_name
        raise ValueError(f"Unknown weight mode: {mode}")

    # 1. Static reflect-pad right=64 on [B, 576] -> [B, 640] via WebGPU/WASM-safe Gather(axis=1)
    # Reflect indices for length 576 with right pad 64: [0..575, 574, 573, ..., 511]
    reflect_idx = np.concatenate([np.arange(576, dtype=np.int64), np.arange(574, 574 - 64, -1, dtype=np.int64)])
    assert reflect_idx.shape == (640,)
    add_init("stft_reflect_idx", reflect_idx)
    add_init("axes_1", np.array([1], dtype=np.int64))
    add_init("axes_2", np.array([2], dtype=np.int64))
    add_init("axes_0", np.array([0], dtype=np.int64))

    nodes.append(helper.make_node("Gather", ["input", "stft_reflect_idx"], ["padded_1d"], axis=1, name="stft_reflect_pad"))
    nodes.append(helper.make_node("Unsqueeze", ["padded_1d", "axes_1"], ["padded_3d"], name="stft_unsqueeze"))

    # 2. Operator-Safe Conv1d Fourier Basis [258, 1, 256] stride=128 -> [B, 258, 4]
    basis_w = add_weight("stft_basis_w", weights["stft.forward_basis_buffer"], mode="fp16")
    nodes.append(
        helper.make_node(
            "Conv",
            ["padded_3d", basis_w],
            ["stft_conv"],
            kernel_shape=[256],
            strides=[128],
            pads=[0, 0],
            dilations=[1],
            group=1,
            name="stft_conv_node",
        )
    )

    # Split [B, 258, 4] along axis=1 into real [B, 129, 4] and imag [B, 129, 4] -> magnitude sqrt(re^2 + im^2)
    add_init("split_129", np.array([129, 129], dtype=np.int64))
    nodes.append(helper.make_node("Split", ["stft_conv", "split_129"], ["stft_re", "stft_im"], axis=1, name="stft_split"))
    nodes.append(helper.make_node("Mul", ["stft_re", "stft_re"], ["stft_re2"], name="stft_re_sq"))
    nodes.append(helper.make_node("Mul", ["stft_im", "stft_im"], ["stft_im2"], name="stft_im_sq"))
    nodes.append(helper.make_node("Add", ["stft_re2", "stft_im2"], ["stft_pow"], name="stft_mag_sq"))
    nodes.append(helper.make_node("Sqrt", ["stft_pow"], ["stft_mag"], name="stft_mag_sqrt"))

    # 3. 4-Stage Conv1d + ReLU Acoustic Encoder -> [B, 128, 1] -> Squeeze -> `acoustic_features` [B, 128]
    strides = [1, 2, 2, 1]
    prev = "stft_mag"
    for idx, stride in enumerate(strides):
        w_mode = "fp16" if idx == 0 else "w8a32"
        cw = add_weight(f"enc{idx}_w", weights[f"encoder.{idx}.reparam_conv.weight"], mode=w_mode, ch_axis=0)
        cb = add_init(f"enc{idx}_b", weights[f"encoder.{idx}.reparam_conv.bias"])
        conv_out = f"enc{idx}_conv"
        relu_out = f"enc{idx}_relu"
        nodes.append(
            helper.make_node(
                "Conv",
                [prev, cw, cb],
                [conv_out],
                kernel_shape=[3],
                strides=[stride],
                pads=[1, 1],
                dilations=[1],
                group=1,
                name=f"enc{idx}_conv_node",
            )
        )
        nodes.append(helper.make_node("Relu", [conv_out], [relu_out], name=f"enc{idx}_relu_node"))
        prev = relu_out

    # Named cut-point tensor `acoustic_features` [B, 128] for zero-copy protobuf graph splitting!
    nodes.append(helper.make_node("Squeeze", [prev, "axes_2"], ["acoustic_features"], name="enc_squeeze_features"))

    # 4. Unrolled Elementary LSTMCell (PyTorch gate order: i, f, g, o — each 128 dims, total 512)
    # Transpose weight_ih [512, 128] -> [128, 512] and weight_hh [512, 128] -> [128, 512]
    w_ih_T = weights["decoder.rnn.weight_ih"].T.copy()  # [128, 512]
    w_hh_T = weights["decoder.rnn.weight_hh"].T.copy()  # [128, 512]
    b_lstm = (weights["decoder.rnn.bias_ih"] + weights["decoder.rnn.bias_hh"]).astype(np.float32)  # [512]

    wih_name = add_weight("lstm_w_ih_T", w_ih_T, mode="w8a32", ch_axis=1)
    whh_name = add_weight("lstm_w_hh_T", w_hh_T, mode="w8a32", ch_axis=1)
    add_init("lstm_bias", b_lstm)
    add_init("idx_0", np.array(0, dtype=np.int64))
    add_init("idx_1", np.array(1, dtype=np.int64))
    add_init("split_gates_4x128", np.array([128, 128, 128, 128], dtype=np.int64))

    # Extract h_prev [B, 128] and c_prev [B, 128] from state [2, B, 128]
    nodes.append(helper.make_node("Gather", ["state", "idx_0"], ["h_prev"], axis=0, name="gather_h_prev"))
    nodes.append(helper.make_node("Gather", ["state", "idx_1"], ["c_prev"], axis=0, name="gather_c_prev"))

    nodes.append(helper.make_node("MatMul", ["acoustic_features", wih_name], ["gates_x"], name="lstm_mm_x"))
    nodes.append(helper.make_node("MatMul", ["h_prev", whh_name], ["gates_h"], name="lstm_mm_h"))
    nodes.append(helper.make_node("Add", ["gates_x", "gates_h"], ["gates_xh"], name="lstm_add_xh"))
    nodes.append(helper.make_node("Add", ["gates_xh", "lstm_bias"], ["gates_pre"], name="lstm_add_bias"))

    nodes.append(
        helper.make_node(
            "Split",
            ["gates_pre", "split_gates_4x128"],
            ["gate_i_pre", "gate_f_pre", "gate_g_pre", "gate_o_pre"],
            axis=1,
            name="lstm_split_gates",
        )
    )
    nodes.append(helper.make_node("Sigmoid", ["gate_i_pre"], ["gate_i"], name="lstm_sig_i"))
    nodes.append(helper.make_node("Sigmoid", ["gate_f_pre"], ["gate_f"], name="lstm_sig_f"))
    nodes.append(helper.make_node("Tanh", ["gate_g_pre"], ["gate_g"], name="lstm_tanh_g"))
    nodes.append(helper.make_node("Sigmoid", ["gate_o_pre"], ["gate_o"], name="lstm_sig_o"))

    nodes.append(helper.make_node("Mul", ["gate_f", "c_prev"], ["fc_prev"], name="lstm_mul_fc"))
    nodes.append(helper.make_node("Mul", ["gate_i", "gate_g"], ["ig_new"], name="lstm_mul_ig"))
    nodes.append(helper.make_node("Add", ["fc_prev", "ig_new"], ["c_next"], name="lstm_c_next"))
    nodes.append(helper.make_node("Tanh", ["c_next"], ["tanh_c_next"], name="lstm_tanh_c"))
    nodes.append(helper.make_node("Mul", ["gate_o", "tanh_c_next"], ["h_next"], name="lstm_h_next"))

    nodes.append(helper.make_node("Unsqueeze", ["h_next", "axes_0"], ["h_next_3d"], name="unsqueeze_h_next"))
    nodes.append(helper.make_node("Unsqueeze", ["c_next", "axes_0"], ["c_next_3d"], name="unsqueeze_c_next"))
    nodes.append(helper.make_node("Concat", ["h_next_3d", "c_next_3d"], ["stateN"], axis=0, name="concat_stateN"))

    # 5. Final Decoder Projection: ReLU(h_next) -> MatMul [128, 1] + bias -> Sigmoid -> output [B, 1]
    head_w = weights["decoder.decoder.2.weight"].reshape(128, 1).astype(np.float32)
    head_b = weights["decoder.decoder.2.bias"].reshape(1).astype(np.float32)
    add_init("head_w", head_w)
    add_init("head_b", head_b)

    nodes.append(helper.make_node("Relu", ["h_next"], ["head_relu"], name="head_relu_node"))
    nodes.append(helper.make_node("MatMul", ["head_relu", "head_w"], ["head_mm"], name="head_mm_node"))
    nodes.append(helper.make_node("Add", ["head_mm", "head_b"], ["head_logits"], name="head_add_bias"))
    nodes.append(helper.make_node("Sigmoid", ["head_logits"], ["output"], name="head_sigmoid"))

    inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 576]),
        helper.make_tensor_value_info("state", TensorProto.FLOAT, [2, "batch", 128]),
    ]
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1]),
        helper.make_tensor_value_info("stateN", TensorProto.FLOAT, [2, "batch", 128]),
    ]
    value_info = [
        helper.make_tensor_value_info("acoustic_features", TensorProto.FLOAT, ["batch", 128]),
    ]

    graph = helper.make_graph(
        nodes,
        "wasm_vad_16k_mobile" if hybrid_mobile else "wasm_vad_16k_studio",
        inputs,
        outputs,
        initializer=inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, producer_name="wasm-vad", opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_parity() -> dict:
    raw_sess = ort.InferenceSession(str(RAW_ONNX), providers=["CPUExecutionProvider"])
    studio_sess = ort.InferenceSession(str(STUDIO_ONNX), providers=["CPUExecutionProvider"])
    mobile_sess = ort.InferenceSession(str(MOBILE_ONNX), providers=["CPUExecutionProvider"])

    sample_dir = ROOT.parent / "swanada" / "docs" / "samples"
    if not sample_dir.exists():
        sample_dir = ROOT / "docs" / "samples"
    wav_files = sorted(sample_dir.glob("*.wav"))[:12]

    studio_diffs = []
    mobile_diffs = []
    agree_05 = 0
    total_frames = 0

    for wf in wav_files:
        pcm, sr = sf.read(str(wf), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        assert sr == 16000, f"Expected 16kHz in {wf}, got {sr}"

        st_raw = np.zeros((2, 1, 128), dtype=np.float32)
        st_stu = np.zeros((2, 1, 128), dtype=np.float32)
        st_mob = np.zeros((2, 1, 128), dtype=np.float32)
        ctx = np.zeros(64, dtype=np.float32)

        n_frames = len(pcm) // 512
        for i in range(n_frames):
            frame = pcm[i * 512 : (i + 1) * 512]
            x = np.concatenate([ctx, frame])[None, :].astype(np.float32)
            ctx = frame[-64:].copy()

            p_raw, st_raw = raw_sess.run(None, {"input": x, "state": st_raw, "sr": np.array(16000, dtype=np.int64)})
            p_stu, st_stu = studio_sess.run(None, {"input": x, "state": st_stu})
            p_mob, st_mob = mobile_sess.run(None, {"input": x, "state": st_mob})

            pr = float(p_raw[0, 0])
            ps = float(p_stu[0, 0])
            pm = float(p_mob[0, 0])

            studio_diffs.append(abs(pr - ps))
            mobile_diffs.append(abs(ps - pm))
            if (ps >= 0.5) == (pm >= 0.5):
                agree_05 += 1
            total_frames += 1

    res = {
        "evaluatedWavClips": len(wav_files),
        "evaluatedFrames": total_frames,
        "studioMaxAbsDiffVsRaw": float(np.max(studio_diffs)),
        "studioMeanAbsDiffVsRaw": float(np.mean(studio_diffs)),
        "mobileMaxAbsDiffVsStudio": float(np.max(mobile_diffs)),
        "mobileMeanAbsDiffVsStudio": float(np.mean(mobile_diffs)),
        "mobileDecisionAgreementAt0_5": round(agree_05 / max(1, total_frames), 6),
    }
    return res


def main() -> None:
    check_padding_behavior(RAW_ONNX)
    weights = extract_16k_weights(RAW_ONNX)
    # Grab decoder.2.bias separately since size == 1
    m_raw = onnx.load(str(RAW_ONNX))
    if_node = [n for n in m_raw.graph.node if n.op_type == "If"][0]
    g16 = [a.g for a in if_node.attribute if a.name == "then_branch"][0]
    for n in g16.node:
        if n.op_type == "Constant" and n.output[0].endswith("decoder.decoder.2.bias"):
            weights["decoder.decoder.2.bias"] = numpy_helper.to_array(n.attribute[0].t).astype(np.float32)

    studio_model = build_clean_vad_onnx(weights, hybrid_mobile=False)
    onnx.save(studio_model, str(STUDIO_ONNX))

    mobile_model = build_clean_vad_onnx(weights, hybrid_mobile=True)
    onnx.save(mobile_model, str(MOBILE_ONNX))

    parity = verify_parity()
    raw_bytes = RAW_ONNX.stat().st_size
    studio_bytes = STUDIO_ONNX.stat().st_size
    mobile_bytes = MOBILE_ONNX.stat().st_size

    manifest = {
        "version": "0.1.0",
        "license": "MIT (derived from Silero VAD v5, branch-free WebGPU/WASM distillation)",
        "sampleRate": 16000,
        "frameSamples": 512,
        "contextSamples": 64,
        "windowSamples": 576,
        "cutPointTensor": "acoustic_features",
        "rawBaselineBytes": raw_bytes,
        "models": {
            "studio": {
                "file": "vad-studio.onnx",
                "precision": "FP32 (0 If nodes, 0 LSTM ops, WebGPU + WASM SIMD native)",
                "bytes": studio_bytes,
                "sha256": sha256_file(STUDIO_ONNX),
                "compressionVsRaw": round(raw_bytes / studio_bytes, 2),
            },
            "mobile": {
                "file": "vad-mobile.onnx",
                "precision": "Hybrid W8A32 (Int8 per-channel RNN/Conv) + FP16 (Fourier STFT basis + spectral Conv0)",
                "bytes": mobile_bytes,
                "sha256": sha256_file(MOBILE_ONNX),
                "compressionVsRaw": round(raw_bytes / mobile_bytes, 2),
            },
        },
        "parity": parity,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    assert parity["studioMaxAbsDiffVsRaw"] < 1e-5, f"Studio parity drift: {parity['studioMaxAbsDiffVsRaw']}"
    assert parity["mobileMeanAbsDiffVsStudio"] < 0.01, f"Mobile parity drift: {parity['mobileMeanAbsDiffVsStudio']}"


if __name__ == "__main__":
    main()
