#!/usr/bin/env python3
"""
Hybrid Gradio app: voice clone (reference audio) + voice design (caption)
using a merged dual-conditioned checkpoint.
"""
from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    save_wav,
)

FIXED_SECONDS = 30.0
MAX_GRADIO_CANDIDATES = 32
GRADIO_AUDIO_COLS_PER_ROW = 8


def _default_checkpoint() -> str:
    # Prefer merged checkpoint if present
    merged = Path("model/merged_normal_plus_caption.safetensors")
    if merged.exists():
        return str(merged)
    candidates = sorted(
        [
            *Path(".").glob("**/checkpoint_*.pt"),
            *Path(".").glob("**/checkpoint_*.safetensors"),
        ]
    )
    if not candidates:
        return "Aratako/Irodori-TTS-500M-v2"
    return str(candidates[-1])


def _default_model_device() -> str:
    return default_runtime_device()


def _default_codec_device() -> str:
    return default_runtime_device()


def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)


def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float or blank.") from exc


def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc


def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)


def _resolve_ref_wav(uploaded_audio: str | None) -> str | None:
    if uploaded_audio is not None and str(uploaded_audio).strip() != "":
        return str(uploaded_audio)
    return None


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")
    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint
    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    print(f"[gradio-hybrid] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def _build_runtime_key(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        enable_watermark=bool(enable_watermark),
        compile_model=False,
        compile_dynamic=False,
    )


def _load_model(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> str:
    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
    )
    runtime, reloaded = get_cached_runtime(runtime_key)
    if reloaded:
        status = "loaded model into memory"
    else:
        status = "model already loaded; reused existing runtime"
    caps = []
    if runtime.model_cfg.use_speaker_condition:
        caps.append("speaker (voice clone)")
    if runtime.model_cfg.use_caption_condition:
        caps.append("caption (voice design)")
    return (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}\n"
        f"model_device: {runtime_key.model_device}\n"
        f"model_precision: {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device}\n"
        f"codec_precision: {runtime_key.codec_precision}\n"
        f"conditioning: {', '.join(caps) if caps else 'text only'}"
    )


# =================================================================================
# メタデータ埋め込み用・読み取り用関数群 (ID3v2.3 準拠 / 外部ライブラリなし)
# =================================================================================
def _encode_synchsafe(size: int) -> bytes:
    return bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F])


def _parse_synchsafe(b: bytes) -> int:
    return (b[0] << 21) | (b[1] << 14) | (b[2] << 7) | b[3]


def _create_id3v2_3_frame(frame_id: str, payload: bytes) -> bytes:
    frame_id_bytes = frame_id.encode("ascii")
    size_bytes = struct.pack(">I", len(payload))
    flags = b"\x00\x00"
    return frame_id_bytes + size_bytes + flags + payload


def _create_text_payload_utf16(text: str) -> bytes:
    return b"\x01" + text.encode("utf-16")


def _create_txxx_payload_utf16(desc: str, text: str) -> bytes:
    desc_bytes = desc.encode("utf-16") + b"\x00\x00"
    val_bytes = text.encode("utf-16")
    return b"\x01" + desc_bytes + val_bytes


def _create_comm_payload_utf16(text: str) -> bytes:
    lang = b"eng"
    short_desc = "".encode("utf-16") + b"\x00\x00"
    val_bytes = text.encode("utf-16")
    return b"\x01" + lang + short_desc + val_bytes


def _parse_txxx(payload: bytes) -> tuple[str, str]:
    if len(payload) < 2:
        return ("", "")
    encoding = payload[0]
    data = payload[1:]
    if encoding == 1:
        parts = []
        last_idx = 0
        for i in range(0, len(data) - 1, 2):
            if data[i:i + 2] == b"\x00\x00":
                parts.append(data[last_idx:i])
                last_idx = i + 2
                break
        parts.append(data[last_idx:])
        desc = parts[0].decode("utf-16", errors="ignore").rstrip("\x00") if len(parts) > 0 else ""
        val = parts[1].decode("utf-16", errors="ignore").rstrip("\x00") if len(parts) > 1 else ""
        return desc, val
    else:
        parts = data.split(b"\x00", 1)
        desc = parts[0].decode("iso-8859-1", errors="ignore").rstrip("\x00")
        val = parts[1].decode("iso-8859-1", errors="ignore").rstrip("\x00") if len(parts) > 1 else ""
        return desc, val


def _parse_id3v2_3_text(payload: bytes) -> str:
    if not payload:
        return ""
    encoding = payload[0]
    data = payload[1:]
    if encoding == 1:
        return data.decode("utf-16", errors="ignore").rstrip("\x00")
    else:
        return data.decode("iso-8859-1", errors="ignore").rstrip("\x00")


def _embed_metadata_to_wav(filepath: str | Path, text: str, params_json: str) -> None:
    frames = b""
    title = text.replace("\n", " ")[:64] + ("..." if len(text) > 64 else "")
    frames += _create_id3v2_3_frame("TIT2", _create_text_payload_utf16(title))
    frames += _create_id3v2_3_frame("TSSE", _create_text_payload_utf16("Irodori-TTS Gradio Hybrid"))
    frames += _create_id3v2_3_frame("TXXX", _create_txxx_payload_utf16("Prompt", text))
    frames += _create_id3v2_3_frame("TXXX", _create_txxx_payload_utf16("Generation Parameters", params_json))
    frames += _create_id3v2_3_frame("COMM", _create_comm_payload_utf16(params_json))

    id3_tag = b"ID3\x03\x00\x00" + _encode_synchsafe(len(frames)) + frames

    filepath = Path(filepath)
    with open(filepath, "r+b") as f:
        data = f.read()
        if not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
            return
        pad = b"\x00" if len(id3_tag) % 2 != 0 else b""
        id3_chunk = b"id3 " + struct.pack("<I", len(id3_tag)) + id3_tag + pad
        new_riff_size = len(data) - 8 + len(id3_chunk)
        f.seek(4)
        f.write(struct.pack("<I", new_riff_size))
        f.seek(0, 2)
        f.write(id3_chunk)


def _read_metadata_from_wav(filepath: str | Path) -> dict[str, str]:
    filepath = Path(filepath)
    if not filepath.exists():
        return {"error": "File not found"}
    with open(filepath, "rb") as f:
        data = f.read()
    if not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
        return {"error": "Not a valid WAV file"}
    offset = 12
    id3_data = None
    while offset < len(data):
        chunk_id = data[offset:offset + 4]
        if len(chunk_id) < 4:
            break
        chunk_size = struct.unpack("<I", data[offset + 4:offset + 8])[0]
        if chunk_id == b"id3 ":
            id3_data = data[offset + 8:offset + 8 + chunk_size]
            break
        offset += 8 + chunk_size + (chunk_size % 2)
    if not id3_data:
        return {"error": "No ID3 metadata found in the WAV file"}
    if not id3_data.startswith(b"ID3"):
        return {"error": "Invalid ID3 header"}
    tag_size = _parse_synchsafe(id3_data[6:10])
    frames_data = id3_data[10:10 + tag_size]
    offset = 0
    parsed_metadata: dict[str, str] = {}
    while offset < len(frames_data):
        frame_id_bytes = frames_data[offset:offset + 4]
        frame_id = frame_id_bytes.decode("ascii", errors="ignore")
        if not frame_id.isalnum() and not frame_id.strip('\x00'):
            break
        if offset + 10 > len(frames_data):
            break
        frame_size = struct.unpack(">I", frames_data[offset + 4:offset + 8])[0]
        payload = frames_data[offset + 10:offset + 10 + frame_size]
        offset += 10 + frame_size
        if not payload:
            continue
        if frame_id == "TXXX":
            desc, val = _parse_txxx(payload)
            if desc:
                parsed_metadata[desc] = val
        elif frame_id == "COMM":
            if payload[0] == 1:
                parts = []
                data_inner = payload[4:]
                last_idx = 0
                for i in range(0, len(data_inner) - 1, 2):
                    if data_inner[i:i + 2] == b"\x00\x00":
                        parts.append(data_inner[last_idx:i])
                        last_idx = i + 2
                        break
                parts.append(data_inner[last_idx:])
                val = parts[-1].decode("utf-16", errors="ignore").rstrip("\x00") if parts else ""
                parsed_metadata["Comment"] = val
            else:
                val = payload[4:].split(b"\x00")[-1].decode("iso-8859-1", errors="ignore").rstrip("\x00")
                parsed_metadata["Comment"] = val
        elif frame_id.startswith("T"):
            val = _parse_id3v2_3_text(payload)
            parsed_metadata[frame_id] = val
    return parsed_metadata


def _extract_metadata_ui(filepath: str | None) -> tuple[str, str, dict | str]:
    if filepath is None or str(filepath).strip() == "":
        return "", "", {}
    metadata = _read_metadata_from_wav(filepath)
    if "error" in metadata:
        return f"Error: {metadata['error']}", "", {}
    prompt = metadata.get("Prompt", "")
    params_json_str = metadata.get("Generation Parameters", "{}")
    try:
        params = json.loads(params_json_str) if params_json_str else {}
    except json.JSONDecodeError:
        params = {"raw_text": params_json_str}
    other_info_lines = []
    for k, v in metadata.items():
        if k not in ["Prompt", "Generation Parameters", "Comment"]:
            other_info_lines.append(f"{k}: {v}")
    return prompt, "\n".join(other_info_lines), params


# =================================================================================


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
    text: str,
    caption: str,
    uploaded_audio: str | None,
    num_steps: int,
    num_candidates: int,
    seed_raw: str,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    max_text_len_raw: str,
    max_caption_len_raw: str,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
) -> tuple[object, ...]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
    )

    text_value = str(text).strip()
    caption_value = str(caption).strip() if caption else ""

    if text_value == "":
        raise ValueError("text is required.")

    requested_candidates = int(num_candidates)
    if requested_candidates <= 0:
        raise ValueError("num_candidates must be >= 1.")
    if requested_candidates > MAX_GRADIO_CANDIDATES:
        raise ValueError(f"num_candidates must be <= {MAX_GRADIO_CANDIDATES}.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    max_text_len = _parse_optional_int(max_text_len_raw, "max_text_len")
    max_caption_len = _parse_optional_int(max_caption_len_raw, "max_caption_len")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")

    ref_wav = _resolve_ref_wav(uploaded_audio=uploaded_audio)
    no_ref = ref_wav is None

    runtime, reloaded = get_cached_runtime(runtime_key)
    stdout_log(f"[gradio-hybrid] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        "[gradio-hybrid] conditioning: text={} caption={} speaker={}".format(
            "on",
            "on" if caption_value else "off",
            "ref" if ref_wav else "no-ref",
        )
    )

    result = runtime.synthesize(
        SamplingRequest(
            text=text_value,
            caption=caption_value or None,
            ref_wav=ref_wav,
            ref_latent=None,
            no_ref=bool(no_ref),
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=requested_candidates,
            decode_mode="sequential",
            seconds=FIXED_SECONDS,
            max_ref_seconds=30.0,
            max_text_len=max_text_len,
            max_caption_len=max_caption_len,
            num_steps=int(num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            trim_tail=True,
        ),
        log_fn=stdout_log,
    )

    # メタデータ用のJSON構築
    gen_params = {
        "text": text_value,
        "caption": caption_value or None,
        "checkpoint": checkpoint,
        "num_steps": int(num_steps),
        "seed_used": result.used_seed,
        "cfg_guidance_mode": str(cfg_guidance_mode),
        "cfg_scale_text": float(cfg_scale_text),
        "cfg_scale_caption": float(cfg_scale_caption),
        "cfg_scale_speaker": float(cfg_scale_speaker),
        "cfg_scale": cfg_scale,
        "cfg_min_t": float(cfg_min_t),
        "cfg_max_t": float(cfg_max_t),
        "truncation_factor": truncation_factor,
        "rescale_k": rescale_k,
        "rescale_sigma": rescale_sigma,
        "context_kv_cache": bool(context_kv_cache),
        "speaker_kv_scale": speaker_kv_scale,
        "speaker_kv_min_t": speaker_kv_min_t,
        "speaker_kv_max_layers": speaker_kv_max_layers,
        "enable_watermark": enable_watermark,
    }
    if ref_wav is not None:
        gen_params["ref_wav"] = Path(ref_wav).name
    gen_params = {k: v for k, v in gen_params.items() if v is not None}
    params_json = json.dumps(gen_params, ensure_ascii=False)

    out_dir = Path("gradio_outputs_hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_paths: list[str] = []

    for i, audio in enumerate(result.audios, start=1):
        out_path = save_wav(
            out_dir / f"{text_value[:10]}_{stamp}_{i:03d}.wav",
            audio.float(),
            result.sample_rate,
        )
        try:
            _embed_metadata_to_wav(out_path, text=text_value, params_json=params_json)
        except Exception as exc:
            stdout_log(f"[gradio-hybrid] Failed to embed metadata to {out_path}: {exc}")
        out_paths.append(str(out_path))

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [
        runtime_msg,
        f"seed_used: {result.used_seed}",
        f"candidates: {len(result.audios)}",
        *[f"saved[{i}]: {path}" for i, path in enumerate(out_paths, start=1)],
        *result.messages,
    ]
    detail_text = "\n".join(detail_lines)
    timing_text = _format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[gradio-hybrid] saved {len(out_paths)} candidates")

    audio_updates: list[object] = []
    for i in range(MAX_GRADIO_CANDIDATES):
        if i < len(out_paths):
            audio_updates.append(gr.update(value=out_paths[i], visible=True))
        else:
            audio_updates.append(gr.update(value=None, visible=False))
    return (*audio_updates, detail_text, timing_text)


def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="Irodori-TTS Hybrid") as demo:
        gr.Markdown("# Irodori-TTS Hybrid (Voice Clone + Voice Design)")
        gr.Markdown(
            "マージ済みチェックポイントで、リファレンス音声（ボイスクローン）と"
            "キャプション（ボイスデザイン）の両方を同時に使えます。"
        )

        with gr.Tabs():
            # ========================
            # Tab 1: Generation
            # ========================
            with gr.Tab("Generation"):
                with gr.Row():
                    checkpoint = gr.Textbox(
                        label="Checkpoint (.pt/.safetensors or HF repo id)",
                        value=default_checkpoint,
                        scale=4,
                    )
                    model_device = gr.Dropdown(
                        label="Model Device",
                        choices=device_choices,
                        value=default_model_device,
                        scale=1,
                    )
                    model_precision = gr.Dropdown(
                        label="Model Precision",
                        choices=model_precision_choices,
                        value=model_precision_choices[0],
                        scale=1,
                    )
                    codec_device = gr.Dropdown(
                        label="Codec Device",
                        choices=device_choices,
                        value=default_codec_device,
                        scale=1,
                    )
                    codec_precision = gr.Dropdown(
                        label="Codec Precision",
                        choices=codec_precision_choices,
                        value=codec_precision_choices[0],
                        scale=1,
                    )
                    enable_watermark = gr.State(False)

                with gr.Row():
                    load_model_btn = gr.Button("Load Model")
                    clear_cache_btn = gr.Button("Unload Model")
                    clear_cache_msg = gr.Textbox(label="Model Status", interactive=False)

                text = gr.Textbox(label="Text", lines=4)

                with gr.Row():
                    with gr.Column(scale=1):
                        caption = gr.Textbox(
                            label="Caption / Style Prompt (optional — voice design)",
                            lines=3,
                            placeholder="例: 落ち着いた女性の声、低めのトーン",
                        )
                    with gr.Column(scale=1):
                        uploaded_audio = gr.Audio(
                            label="Reference Audio (optional — voice clone)",
                            type="filepath",
                        )

                with gr.Accordion("Sampling", open=True):
                    with gr.Row():
                        num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=40, step=1)
                        num_candidates = gr.Slider(
                            label="Num Candidates",
                            minimum=1,
                            maximum=MAX_GRADIO_CANDIDATES,
                            value=1,
                            step=1,
                        )
                        seed_raw = gr.Textbox(label="Seed (blank=random)", value="")

                    with gr.Row():
                        cfg_guidance_mode = gr.Dropdown(
                            label="CFG Guidance Mode",
                            choices=["independent", "joint", "alternating"],
                            value="independent",
                        )
                        cfg_scale_text = gr.Slider(
                            label="CFG Scale Text",
                            minimum=0.0,
                            maximum=100.0,
                            value=3.0,
                            step=0.1,
                        )
                        cfg_scale_caption = gr.Slider(
                            label="CFG Scale Caption",
                            minimum=0.0,
                            maximum=100.0,
                            value=4.0,
                            step=0.1,
                        )
                        cfg_scale_speaker = gr.Slider(
                            label="CFG Scale Speaker",
                            minimum=0.0,
                            maximum=100.0,
                            value=5.0,
                            step=0.1,
                        )

                with gr.Accordion("Advanced (Optional)", open=False):
                    cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
                    with gr.Row():
                        cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                        cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                        context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
                    with gr.Row():
                        max_text_len_raw = gr.Textbox(label="Max Text Len (optional)", value="")
                        max_caption_len_raw = gr.Textbox(label="Max Caption Len (optional)", value="")
                    with gr.Row():
                        truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                        rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                        rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
                    with gr.Row():
                        speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                        speaker_kv_min_t_raw = gr.Textbox(label="Speaker KV Min t (optional)", value="0.9")
                        speaker_kv_max_layers_raw = gr.Textbox(
                            label="Speaker KV Max Layers (optional)", value=""
                        )

                generate_btn = gr.Button("Generate", variant="primary")

                out_audios: list[gr.Audio] = []
                num_rows = (MAX_GRADIO_CANDIDATES + GRADIO_AUDIO_COLS_PER_ROW - 1) // GRADIO_AUDIO_COLS_PER_ROW
                with gr.Column():
                    for row_idx in range(num_rows):
                        with gr.Row():
                            for col_idx in range(GRADIO_AUDIO_COLS_PER_ROW):
                                i = row_idx * GRADIO_AUDIO_COLS_PER_ROW + col_idx
                                if i >= MAX_GRADIO_CANDIDATES:
                                    break
                                out_audios.append(
                                    gr.Audio(
                                        label=f"Generated Audio {i + 1}",
                                        type="filepath",
                                        interactive=False,
                                        visible=(i == 0),
                                        min_width=160,
                                    )
                                )
                out_log = gr.Textbox(label="Run Log", lines=8)
                out_timing = gr.Textbox(label="Timing", lines=8)

                generate_btn.click(
                    _run_generation,
                    inputs=[
                        checkpoint,
                        model_device,
                        model_precision,
                        codec_device,
                        codec_precision,
                        enable_watermark,
                        text,
                        caption,
                        uploaded_audio,
                        num_steps,
                        num_candidates,
                        seed_raw,
                        cfg_guidance_mode,
                        cfg_scale_text,
                        cfg_scale_caption,
                        cfg_scale_speaker,
                        cfg_scale_raw,
                        cfg_min_t,
                        cfg_max_t,
                        context_kv_cache,
                        max_text_len_raw,
                        max_caption_len_raw,
                        truncation_factor_raw,
                        rescale_k_raw,
                        rescale_sigma_raw,
                        speaker_kv_scale_raw,
                        speaker_kv_min_t_raw,
                        speaker_kv_max_layers_raw,
                    ],
                    outputs=[*out_audios, out_log, out_timing],
                )
                model_device.change(
                    _on_model_device_change, inputs=[model_device], outputs=[model_precision]
                )
                codec_device.change(
                    _on_codec_device_change, inputs=[codec_device], outputs=[codec_precision]
                )

                load_model_btn.click(
                    _load_model,
                    inputs=[
                        checkpoint,
                        model_device,
                        model_precision,
                        codec_device,
                        codec_precision,
                        enable_watermark,
                    ],
                    outputs=[clear_cache_msg],
                )
                clear_cache_btn.click(_clear_runtime_cache, outputs=[clear_cache_msg])

            # ========================
            # Tab 2: Metadata Reader
            # ========================
            with gr.Tab("Metadata Reader"):
                gr.Markdown("WAVファイルをアップロードして、埋め込まれたプロンプトや生成パラメータを読み取ります。")
                with gr.Row():
                    with gr.Column(scale=1):
                        meta_audio_in = gr.Audio(type="filepath", label="Upload Generated WAV File")
                    with gr.Column(scale=2):
                        meta_prompt_out = gr.Textbox(label="Prompt", lines=4, interactive=False)
                        meta_params_out = gr.JSON(label="Generation Parameters")
                        meta_raw_out = gr.Textbox(label="Other Tag Data", lines=3, interactive=False)

                meta_audio_in.change(
                    _extract_metadata_ui,
                    inputs=[meta_audio_in],
                    outputs=[meta_prompt_out, meta_raw_out, meta_params_out]
                )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Gradio app for Irodori-TTS (voice clone + voice design).")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    main()
