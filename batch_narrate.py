#!/usr/bin/env python3
"""
テキストファイルをバッチ分割し、Irodori-TTS v4 で朗読音声を生成、
結合して1つの音声ファイル (MP3) として出力する。

Irodori-TTS (v1) フォークの batch_narrate.py を v4 向けに移植したもの。

分割ルール (優先度順):
  1. [sep] マーカーで強制分割
  2. 改行で区切る (合計20字以下の行は次行と結合)
  3. 150字超のチャンクは以下の区切り文字のうち150字に最も近い位置で分割:
     a. 「。」
     b.  ！、？、…
     c.  読点「、」

v4 での変更点:
  - 発話秒数はチェックポイント内蔵の duration predictor が自動推定
    (--seconds で明示指定した場合のみ v1 同様の余裕リトライを行う)
  - --ref-wav は複数指定可。各ファイルは個別に speaker encoder でエンコードされ
    話者 state が連結される (再学習不要)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    # このスクリプトと同じディレクトリの .env のみを読む (override=True)。
    # デフォルトの load_dotenv() は親ディレクトリまで遡って探すため、
    # 隣接する v1 プロジェクト等の .env (v1用チェックポイント設定) を
    # 誤って拾わないよう明示的にパスを固定する。
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
                override=True)
except ImportError:
    pass

import soundfile as sf
import torch

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
)

# ──────────────────────────────────────────────
# デフォルト設定 (.env / 環境変数で上書き可)
# ──────────────────────────────────────────────
def _auto_detect_checkpoint() -> str:
    """NARRATE_MODEL 未設定時、スクリプト隣の model/*.safetensors を新しい順に探す。"""
    model_dir = Path(__file__).resolve().parent / "model"
    candidates = sorted(
        model_dir.glob("*.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else ""


DEFAULT_MODEL = os.getenv("NARRATE_MODEL", "") or _auto_detect_checkpoint()
DEFAULT_CAPTION = os.getenv("NARRATE_CAPTION", "")
# NARRATE_REF_WAV は「;」区切りで複数ファイルを指定可能
DEFAULT_REF_WAVS = [p.strip() for p in os.getenv("NARRATE_REF_WAV", "").split(";") if p.strip()]
DURATION_SCALE = float(os.getenv("DURATION_MULTIPLIER", "1.0"))
DEFAULT_CFG_TEXT = 3.0
DEFAULT_CFG_CAPTION = 3.0
DEFAULT_CFG_SPEAKER = 5.0
DEFAULT_NUM_STEPS = 40
CHUNK_LIMIT = 150
CHUNK_MIN = 20  # これ以下なら改行で分割せず次に結合
SILENCE_SEC = 0.001  # チャンク間の無音（秒）
MEGA_BATCH_SIZE = 500  # メガバッチ分割の目安文字数


# ──────────────────────────────────────────────
# マークダウン記号除去
# ──────────────────────────────────────────────
def strip_markdown(text: str) -> str:
    """マークダウン記号を除去する。"""
    # 見出し: 行頭の # を除去
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # 太字・斜体: ***text***, **text**, *text*, ___text___, __text__, _text_
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    # 取り消し線: ~~text~~
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    # リンク: [text](url) → text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # 画像: ![alt](url) → 除去
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", "", text)
    # インラインコード: `code`
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # リスト記号: 行頭の - や * や + (番号付きリストも)
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # 水平線: --- や *** や ___ のみの行
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # 引用: 行頭の >
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    return text


# ──────────────────────────────────────────────
# テキスト分割
# ──────────────────────────────────────────────
def _split_by_delimiter(text: str, limit: int) -> list[str]:
    """limit字超のテキストを区切り文字で分割する。

    優先度: 。 > ！？… > 、
    limit字に最も近い位置で切る。
    """
    if len(text) <= limit:
        return [text]

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


def split_mega_batches(text: str, size: int = MEGA_BATCH_SIZE) -> list[str]:
    """テキストをsize文字ごとに改行位置で分割する（メガバッチ）。

    size文字未満のテキストはそのまま1バッチとして返す。
    """
    if len(text) <= size:
        return [text]

    lines = text.splitlines(keepends=True)
    batches: list[str] = []
    buf = ""

    for line in lines:
        if buf and len(buf) + len(line) > size:
            batches.append(buf)
            buf = line
        else:
            buf += line

    if buf:
        batches.append(buf)

    return [b for b in batches if b.strip()]


# ──────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="テキストファイルをバッチ朗読して1つのMP3に結合する (Irodori-TTS v4)",
    )
    parser.add_argument(
        "input",
        help="入力テキストファイル (.txt / .md)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="出力MP3パス (省略時: 入力ファイル名.mp3)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="チェックポイント (.pt/.safetensors) [env: NARRATE_MODEL]")
    parser.add_argument("--caption", default=DEFAULT_CAPTION,
                        help="Voice Design 用キャプション [env: NARRATE_CAPTION]")
    parser.add_argument("--ref-wav", nargs="+", action="extend", default=None,
                        help="リファレンス音声WAV (複数指定可。各ファイルが個別にエンコードされ話者stateが連結される) "
                             "[env: NARRATE_REF_WAV（;区切り）]")
    parser.add_argument("--no-ref", action="store_true",
                        help="リファレンス音声なしで生成 (caption/text のみ)")
    parser.add_argument("--cfg-scale-text", type=float, default=DEFAULT_CFG_TEXT)
    parser.add_argument("--cfg-scale-caption", type=float, default=DEFAULT_CFG_CAPTION)
    parser.add_argument("--cfg-scale-speaker", type=float, default=DEFAULT_CFG_SPEAKER)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seconds", type=float, default=None,
                        help="生成秒数を明示指定 (省略時: duration predictor による自動推定)")
    parser.add_argument("--duration-scale", type=float, default=DURATION_SCALE,
                        help="自動推定秒数の補正係数 [env: DURATION_MULTIPLIER]")
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

    # マークダウン記号を除去
    raw_text = strip_markdown(raw_text)

    # メガバッチ分割（500文字以上の場合）
    mega_batches = split_mega_batches(raw_text, size=MEGA_BATCH_SIZE)
    is_multi_batch = len(mega_batches) > 1

    if is_multi_batch:
        print(f"[mega] {len(mega_batches)} メガバッチに分割:")
        for i, mb in enumerate(mega_batches, 1):
            print(f"  [batch {i}] ({len(mb)}字)")

    # 各メガバッチごとにチャンク分割
    all_batch_chunks: list[list[str]] = []
    for mb in mega_batches:
        chunks = split_text(mb, limit=int(args.chunk_limit))
        all_batch_chunks.append(chunks)

    total_chunks = sum(len(c) for c in all_batch_chunks)
    if total_chunks == 0:
        print("[error] テキストが空です。", file=sys.stderr)
        sys.exit(1)

    for bi, chunks in enumerate(all_batch_chunks, 1):
        prefix = f"[batch {bi}] " if is_multi_batch else ""
        print(f"\n{prefix}[split] {len(chunks)} チャンクに分割:")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i:3d}] ({len(c):4d}字) {c[:60]}{'…' if len(c) > 60 else ''}")

    if args.dry_run:
        return

    # ── リファレンスWAVの決定: CLI指定 > 環境変数 (";"区切りで複数可) ──
    ref_wavs: list[str] | None = args.ref_wav if args.ref_wav else (DEFAULT_REF_WAVS or None)
    if args.no_ref:
        ref_wavs = None
    print(f"[debug] ref_wavs={ref_wavs} no_ref={args.no_ref}")

    # ── 出力パス ──
    base_output = Path(args.output) if args.output else input_path.with_suffix(".mp3")

    # ── ランタイム初期化 ──
    if not str(args.model).strip():
        print(
            "[error] チェックポイントが未指定です。--model か NARRATE_MODEL を設定するか、"
            "model/ に .safetensors を配置してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\n[init] checkpoint: {args.model}")
    print("[init] モデル読み込み中…", flush=True)
    t0 = time.perf_counter()
    try:
        runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=str(args.model),
                model_device=str(args.model_device),
                model_precision=str(args.model_precision),
                codec_device=str(args.codec_device),
                codec_precision="fp32",
            )
        )
    except ValueError as exc:
        if "use_speaker_condition_override" in str(exc):
            print(
                "[error] このチェックポイントは v1 (Irodori-TTS) 用のため v4 では読み込めません。\n"
                "        v4 用チェックポイント (例: model/v4-quant.safetensors) を "
                "--model か NARRATE_MODEL で指定してください。\n"
                f"        詳細: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    print(f"[init] 完了 ({time.perf_counter() - t0:.1f}s)\n", flush=True)

    if runtime.model_cfg.use_speaker_condition_resolved and ref_wavs is None and not args.no_ref:
        print(
            "[error] このチェックポイントは speaker conditioning を使用します。"
            "--ref-wav か --no-ref を指定してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── コーデック実サンプルレート取得 ──
    sample_rate = runtime.codec.sample_rate
    print(f"[info] codec sample_rate = {sample_rate}")

    # ── 無音テンソル ──
    silence_samples = int(args.silence * sample_rate)
    silence = torch.zeros(1, silence_samples)

    # ── 既存バッチ番号の検出（続きから出力） ──
    batch_start = 0
    if is_multi_batch:
        existing = sorted(base_output.parent.glob(f"{base_output.stem}_[0-9][0-9][0-9]{base_output.suffix}"))
        if existing:
            # 最後のファイル名から番号を取得
            last_num = int(existing[-1].stem.rsplit("_", 1)[-1])
            batch_start = last_num
            print(f"[mega] 既存バッチ検出: {existing[-1].name} → {batch_start + 1:03d} から続行")

    # ── メガバッチごとに生成 & 保存 ──
    total_gen_all = 0.0
    for bi, chunks in enumerate(all_batch_chunks, 1):
        batch_num = batch_start + bi
        # 出力パス: メガバッチが1つなら base_output、複数なら _001.mp3 等
        if is_multi_batch:
            output_path = base_output.with_stem(f"{base_output.stem}_{batch_num:03d}")
            print(f"\n{'=' * 50}")
            print(f"[batch {batch_num} ({bi}/{len(all_batch_chunks)})] {len(chunks)} チャンク → {output_path.name}")
            print(f"{'=' * 50}")
        else:
            output_path = base_output

        audio_parts: list[torch.Tensor] = []
        total_gen = 0.0

        for i, chunk in enumerate(chunks, 1):
            # 秒数: 明示指定時のみ余裕リトライ、省略時は duration predictor に任せる
            manual_seconds = None if args.seconds is None else float(args.seconds)
            chunk_seconds = manual_seconds
            attempt = 0
            while True:
                attempt += 1
                batch_label = f"batch{batch_num} " if is_multi_batch else ""
                seconds_label = f"{chunk_seconds:.1f}s" if chunk_seconds is not None else "auto"
                print(f"[{batch_label}gen {i}/{len(chunks)}] ({len(chunk)}字, {seconds_label}{f', retry#{attempt - 1}' if attempt > 1 else ''}) {chunk[:50]}{'…' if len(chunk) > 50 else ''}")
                t1 = time.perf_counter()

                result = runtime.synthesize(
                    SamplingRequest(
                        text=chunk,
                        caption=str(args.caption) if str(args.caption).strip() else None,
                        ref_wavs=ref_wavs,
                        no_ref=bool(args.no_ref),
                        num_candidates=1,
                        decode_mode="sequential",
                        seconds=chunk_seconds,
                        duration_scale=float(args.duration_scale),
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

                if chunk_seconds is None:
                    # 自動推定: リトライなし
                    print(f"        → {audio_sec:.2f}s 音声 (生成: {elapsed:.1f}s, seed={result.used_seed})")
                    break

                margin = chunk_seconds - audio_sec
                if margin >= 1.0:
                    print(f"        → {audio_sec:.2f}s 音声 (余裕{margin:.1f}s, 生成: {elapsed:.1f}s, seed={result.used_seed})")
                    break
                else:
                    print(f"        → {audio_sec:.2f}s 音声 (余裕{margin:.1f}s < 1.0s, リトライ)")
                    chunk_seconds += 1.0

            audio_parts.append(result.audio)
            if i < len(chunks):
                audio_parts.append(silence)

        # ── 結合 & 保存 ──
        combined = torch.cat(audio_parts, dim=-1)
        total_audio_sec = combined.shape[-1] / sample_rate

        audio_np = combined.squeeze(0).float().cpu().numpy()
        sf.write(str(output_path), audio_np, sample_rate, format="MP3")
        print(f"\n[done] 保存: {output_path}")
        print(f"       合計音声: {total_audio_sec:.1f}s / 生成時間: {total_gen:.1f}s")
        print(f"       RTF: {total_gen / total_audio_sec:.2f}x")
        total_gen_all += total_gen

    if is_multi_batch:
        print(f"\n[done] 全{len(all_batch_chunks)}バッチ完了 / 総生成時間: {total_gen_all:.1f}s")


if __name__ == "__main__":
    main()
