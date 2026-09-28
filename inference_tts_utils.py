"""T5Gemma-TTS から移植した日本語テキスト正規化ユーティリティ。"""
import re
from typing import Optional, Tuple

from duration_estimator import detect_language

# Text normalization (only applied when language is Japanese)
_REPLACE_MAP = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
    r"[\uff5e\u301C]": "ー",
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}
_FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)
_HALFWIDTH_KATAKANA_CHARS = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
_FULLWIDTH_KATAKANA_CHARS = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
_HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)
_FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))
    }
)


def _normalize_japanese_text(text: str) -> str:
    """Mirror dataset-side normalization; used only when lang=ja."""
    for pattern, replacement in _REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = text.translate(_FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(_FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(_HALFWIDTH_KATAKANA_TO_FULLWIDTH)

    # Collapse long ellipses
    text = re.sub(r"…{3,}", "……", text)
    return text


def normalize_text_with_lang(text: str, lang: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Normalize text if (and only if) the language is Japanese.
    Returns normalized_text and the resolved language code to avoid re-detecting downstream.
    """
    resolved_lang = lang.lower() if isinstance(lang, str) else None
    if not text:
        return text, resolved_lang
    if resolved_lang is None:
        resolved_lang = detect_language(text)
    if resolved_lang and resolved_lang.startswith("ja"):
        return _normalize_japanese_text(text), resolved_lang
    return text, resolved_lang
