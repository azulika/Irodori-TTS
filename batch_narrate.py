#!/usr/bin/env python3
"""
テキストファイルをバッチ分割し、Voice Clone + Voice Design マージモデルで
朗読音声を生成、結合して1つのWAVファイルとして出力する。

分割ルール (優先度順):
  1. 改行で区切る
  2. 150字超のチャンクは以下の区切り文字のうち150字に最も近い位置で分割:
     a. 「。」
     b.  ！、？、…
     c.  読点「、」
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import soundfile as sf
import torch

from duration_estimator import estimate_duration
from inference_tts_utils import normalize_text_with_lang
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    save_wav,
)

# ──────────────────────────────────────────────
# デフォルト設定
# ──────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("NARRATE_MODEL", "")
# https://drive.google.com/file/d/1PpyDukF4RIzccMXoLOok99cC8iL190YB/view?usp=drive_link
DEFAULT_CAPTION = os.getenv("NARRATE_CAPTION", "")
DEFAULT_REF_WAV = os.getenv("NARRATE_REF_WAV", "")
DEFAULT_REF_TRANSCRIPT = os.getenv("NARRATE_REF_TRANSCRIPT", "")
DURATION_MULTIPLIER = float(os.getenv("DURATION_MULTIPLIER", "1.0"))
DEFAULT_CFG_CAPTION = 11.9
DEFAULT_CFG_TEXT = 3.0
DEFAULT_CFG_SPEAKER = 5.0
DEFAULT_NUM_STEPS = 40
CHUNK_LIMIT = 150
CHUNK_MIN = 20  # これ以下なら改行で分割せず次に結合
SILENCE_SEC = 0.001  # チャンク間の無音（秒）


# ──────────────────────────────────────────────
# テキスト分割
# ──────────────────────────────────────────────
def _split_by_delimiter(text: str, limit: int) -> list[str]:
    """limit字超のテキストを区切り文字で分割する。

    優先度: 。 > ！？… > 、
    150字に最も近い位置で切る。
    """
    if len(text) <= limit:
        return [text]

    # 区切り文字を優先度グループに分ける
    delimiters_priority = [
        re.compile(r"。"),
        re.compile(r"[！？!?…]"),
        re.compile(r"[、,]"),
    ]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > limit:
        best_pos = -1

        for pattern in delimiters_priority:
            # limit字以内の全マッチ位置を探し、limitに最も近いものを選ぶ
            candidates: list[int] = []
            for m in pattern.finditer(remaining[:limit]):
                candidates.append(m.end())  # 区切り文字の直後で切る
            if candidates:
                best_pos = max(candidates)  # limit以内で最も後ろ
                break

        if best_pos <= 0:
            # どの区切り文字もない場合、limit字でハード分割
            best_pos = limit

        chunks.append(remaining[:best_pos])
        remaining = remaining[best_pos:]

    if remaining:
        chunks.append(remaining)

    return chunks


SEP_MARKER = "[sep]"


def split_text(text: str, limit: int = CHUNK_LIMIT, minimum: int = CHUNK_MIN) -> list[str]:
    """テキストを朗読バッチ用に分割する。"""
    # 0. [sep]マーカーで強制分割してから各セグメントを個別に処理
    segments = re.split(re.escape(SEP_MARKER), text, flags=re.IGNORECASE)
    if len(segments) > 1:
        chunks: list[str] = []
        for seg in segments:
            seg = seg.strip()
            if seg:
                chunks.extend(split_text(seg, limit, minimum))
        return chunks

    # 1. 改行で分割しつつ、短すぎる行は次の行と結合
    lines = text.splitlines()

    merged_lines: list[str] = []
    buf = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if buf:
            buf += line
        else:
            buf = line
        # バッファがminimum字を超えたらチャンク候補として確定
        if len(buf) > minimum:
            merged_lines.append(buf)
            buf = ""
    if buf:
        # 残りがある場合: 前のチャンクに結合するか単独で追加
        if merged_lines and len(buf) <= minimum:
            merged_lines[-1] += buf
        else:
            merged_lines.append(buf)

    # 2. limit字超なら区切り文字で再分割
    chunks: list[str] = []
    for line in merged_lines:
        chunks.extend(_split_by_delimiter(line, limit))

    return chunks


# ──────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="テキストファイルをバッチ朗読して1つのWAVに結合する",
    )
    parser.add_argument(
        "input",
        help="入力テキストファイル (.txt / .md)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="出力WAVパス (省略時: 入力ファイル名.wav)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--caption", default=DEFAULT_CAPTION)
    parser.add_argument("--ref-wav", default=DEFAULT_REF_WAV)
    parser.add_argument("--ref-transcript", default=DEFAULT_REF_TRANSCRIPT,
                        help="リファレンス音声の書き起こし (duration推定の精度向上)")
    parser.add_argument("--cfg-scale-text", type=float, default=DEFAULT_CFG_TEXT)
    parser.add_argument("--cfg-scale-caption", type=float, default=DEFAULT_CFG_CAPTION)
    parser.add_argument("--cfg-scale-speaker", type=float, default=DEFAULT_CFG_SPEAKER)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seconds", type=float, default=None,
                        help="生成秒数 (省略時: テキストから自動推定)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chunk-limit", type=int, default=CHUNK_LIMIT)
    parser.add_argument("--silence", type=float, default=SILENCE_SEC,
                        help="チャンク間の無音時間 (秒)")
    parser.add_argument("--model-device", default=default_runtime_device())
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-device", default=default_runtime_device())
    parser.add_argument("--dry-run", action="store_true",
                        help="分割結果だけ表示して終了")
    args = parser.parse_args()

    # ── 入力読み込み ──
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        print(f"[error] ファイルが見つかりません: {input_path}", file=sys.stderr)
        sys.exit(1)

    raw_text = input_path.read_text(encoding="utf-8")
    chunks = split_text(raw_text, limit=int(args.chunk_limit))

    if not chunks:
        print("[error] テキストが空です。", file=sys.stderr)
        sys.exit(1)

    print(f"[split] {len(chunks)} チャンクに分割:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i:3d}] ({len(c):4d}字) {c[:60]}{'…' if len(c) > 60 else ''}")

    if args.dry_run:
        return

    # ── 出力パス ──
    output_path = Path(args.output) if args.output else input_path.with_suffix(".mp3")

    # ── ランタイム初期化 ──
    print("\n[init] モデル読み込み中…", flush=True)
    t0 = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(args.model),
            model_device=str(args.model_device),
            model_precision=str(args.model_precision),
            codec_device=str(args.codec_device),
            codec_precision="fp32",
        )
    )
    print(f"[init] 完了 ({time.perf_counter() - t0:.1f}s)\n", flush=True)

    # ── コーデック実サンプルレート取得 ──
    sample_rate = runtime.codec.sample_rate
    print(f"[info] codec sample_rate = {sample_rate}")

    # ── 無音テンソル ──
    silence_samples = int(args.silence * sample_rate)
    silence = torch.zeros(1, silence_samples)

    # ── バッチ生成 ──
    audio_parts: list[torch.Tensor] = []
    total_gen = 0.0

    # リファレンス書き起こしを正規化（1回だけ）
    ref_transcript_normalized = args.ref_transcript or None
    if ref_transcript_normalized:
        ref_transcript_normalized, _ = normalize_text_with_lang(ref_transcript_normalized, "ja")
    print(f"[debug] ref_wav={args.ref_wav}")
    print(f"[debug] ref_transcript={ref_transcript_normalized!r}")

    for i, chunk in enumerate(chunks, 1):
        # 秒数の決定: 明示指定 > estimate_duration による自動推定
        if args.seconds is not None:
            chunk_seconds = float(args.seconds)
        else:
            normalized_chunk, lang_code = normalize_text_with_lang(chunk, "ja")
            chunk_seconds = estimate_duration(
                target_text=normalized_chunk,
                reference_speech=args.ref_wav,
                reference_transcript=ref_transcript_normalized if ref_transcript_normalized else None,
                target_lang=lang_code,
                reference_lang=lang_code,
            )
        chunk_seconds *= DURATION_MULTIPLIER  # 補正
        attempt = 0
        while True:
            attempt += 1
            print(f"[gen {i}/{len(chunks)}] ({len(chunk)}字, {chunk_seconds:.1f}s{f', retry#{attempt-1}' if attempt > 1 else ''}) {chunk[:50]}{'…' if len(chunk) > 50 else ''}")
            t1 = time.perf_counter()

            result = runtime.synthesize(
                SamplingRequest(
                    text=chunk,
                    caption=args.caption,
                    ref_wav=args.ref_wav,
                    no_ref=False,
                    num_candidates=1,
                    decode_mode="sequential",
                    seconds=float(chunk_seconds),
                    num_steps=int(args.num_steps),
                    seed=args.seed,
                    cfg_scale_text=float(args.cfg_scale_text),
                    cfg_scale_caption=float(args.cfg_scale_caption),
                    cfg_scale_speaker=float(args.cfg_scale_speaker),
                    cfg_guidance_mode="independent",
                    context_kv_cache=True,
                    trim_tail=True,
                ),
                log_fn=None,
            )

            elapsed = time.perf_counter() - t1
            total_gen += elapsed
            audio_sec = result.audio.shape[-1] / result.sample_rate
            margin = chunk_seconds - audio_sec

            if margin >= 1.0:
                # 余裕が1秒以上 → OK
                print(f"        → {audio_sec:.2f}s 音声 (余裕{margin:.1f}s, 生成: {elapsed:.1f}s, seed={result.used_seed})")
                break
            else:
                # 尺が足りない → estimate を +1s してリトライ
                print(f"        → {audio_sec:.2f}s 音声 (余裕{margin:.1f}s < 1.0s, リトライ)")
                chunk_seconds += 1.0

        audio_parts.append(result.audio)
        # 最後のチャンク以外は無音を挟む
        if i < len(chunks):
            audio_parts.append(silence)

    # ── 結合 & 保存 ──
    combined = torch.cat(audio_parts, dim=-1)
    total_audio_sec = combined.shape[-1] / sample_rate

    audio_np = combined.squeeze(0).cpu().numpy()
    sf.write(str(output_path), audio_np, sample_rate, format="MP3")
    print(f"\n[done] 保存: {output_path}")
    print(f"       合計音声: {total_audio_sec:.1f}s / 生成時間: {total_gen:.1f}s")
    print(f"       RTF: {total_gen / total_audio_sec:.2f}x")


if __name__ == "__main__":
    main()
