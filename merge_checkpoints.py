#!/usr/bin/env python3
"""
Merge caption conditioning weights from VoiceDesign checkpoint
into the Normal checkpoint, producing a dual-conditioned model.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NORMAL_PATH = r"C:\Users\komur\.cache\huggingface\hub\models--Aratako--Irodori-TTS-500M-v2\snapshots\d148eafca51aaa078e57b2e140ea1bd4e1a9a06f\model.safetensors"
VD_PATH = r"C:\Users\komur\.cache\huggingface\hub\models--Aratako--Irodori-TTS-500M-v2-VoiceDesign\snapshots\aa7255055652a886db85d8c80c08fc6de9ca7826\model.safetensors"
OUTPUT_PATH = Path("model/merged_normal_plus_caption.safetensors")


def main() -> None:
    print("Loading Normal checkpoint...")
    f_normal = safe_open(NORMAL_PATH, framework="pt")
    normal_meta = f_normal.metadata() or {}
    normal_cfg = json.loads(normal_meta.get("config_json", "{}"))

    print("Loading VoiceDesign checkpoint...")
    f_vd = safe_open(VD_PATH, framework="pt")
    vd_cfg = json.loads((f_vd.metadata() or {}).get("config_json", "{}"))

    # Start with all Normal weights
    merged: dict[str, torch.Tensor] = {}
    for key in f_normal.keys():
        merged[key] = f_normal.get_tensor(key)
    print(f"  Normal keys: {len(merged)}")

    # Copy caption-specific weights from VoiceDesign
    caption_keys = [k for k in f_vd.keys() if k not in f_normal.keys()]
    for key in caption_keys:
        merged[key] = f_vd.get_tensor(key)
    print(f"  Added caption keys from VD: {len(caption_keys)}")
    print(f"  Total merged keys: {len(merged)}")

    # Build merged config: Normal base + caption fields from VD + force both conditions on
    merged_cfg = dict(normal_cfg)
    merged_cfg["use_caption_condition"] = True
    merged_cfg["use_speaker_condition_override"] = True
    merged_cfg["caption_vocab_size"] = vd_cfg.get("caption_vocab_size", 99574)
    merged_cfg["caption_tokenizer_repo"] = vd_cfg.get("caption_tokenizer_repo", "llm-jp/llm-jp-3-150m")
    merged_cfg["caption_add_bos"] = vd_cfg.get("caption_add_bos", True)
    merged_cfg["caption_dim"] = vd_cfg.get("caption_dim", 512)
    merged_cfg["caption_layers"] = vd_cfg.get("caption_layers", 10)
    merged_cfg["caption_heads"] = vd_cfg.get("caption_heads", 8)
    merged_cfg["caption_mlp_ratio"] = vd_cfg.get("caption_mlp_ratio", 2.6)
    merged_cfg["max_caption_len"] = vd_cfg.get("max_caption_len", 512)

    print("\nMerged config:")
    print(json.dumps(merged_cfg, indent=2, ensure_ascii=False))

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"config_json": json.dumps(merged_cfg, ensure_ascii=False)}
    print(f"\nSaving to {OUTPUT_PATH}...")
    save_file(merged, str(OUTPUT_PATH), metadata=metadata)

    # Verify
    f_out = safe_open(str(OUTPUT_PATH), framework="pt")
    print(f"Saved: {len(f_out.keys())} keys")
    out_cfg = json.loads((f_out.metadata() or {}).get("config_json", "{}"))
    print(f"  use_caption_condition: {out_cfg.get('use_caption_condition')}")
    print(f"  use_speaker_condition_override: {out_cfg.get('use_speaker_condition_override')}")
    print("Done.")


if __name__ == "__main__":
    main()
