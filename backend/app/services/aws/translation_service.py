# backend/app/services/aws/translation_service.py
from __future__ import annotations

import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError, BotoCoreError
except Exception:
    boto3 = None
    BotoConfig = None
    ClientError = Exception  # type: ignore
    BotoCoreError = Exception  # type: ignore

REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# Feature flag: allow fully disabling AWS usage at runtime
AWS_TRANSLATION_ENABLED = os.getenv("AWS_TRANSLATION_ENABLED", "0") in ("1", "true", "True")

# Practical request size caps (AWS docs: ~5KB for Comprehend, ~5K chars for Translate)
MAX_TEXT_BYTES = 4500  # leave headroom for UTF-8 multibyte


def _clip_text(s: str) -> str:
    if not s:
        return s
    # conservative: clip by chars; for CJK this keeps us well under 5KB
    return s[:2000]


def _likely_traditional(s: str) -> bool:
    """
    Heuristic: return True if the text looks like Traditional Chinese.
    - Presence of Bopomofo block
    - Presence of common Traditional-only ideographs (粗略)
    This is a simple, fast check sufficient for routing zh → zh-TW.
    """
    if not s:
        return False
    for ch in s:
        code = ord(ch)
        # Bopomofo ranges
        if 0x3100 <= code <= 0x312F or 0x31A0 <= code <= 0x31BF:
            return True
    # quick scan for a few Traditional-only characters
    trad_markers = "臺灣萬與學體國區廣門車嗎麼書曆戰後歷"
    return any(c in s for c in trad_markers)


class AWSTranslationService:
    """Optional AWS-based language detection & translation (stateless)."""

    def __init__(self):
        self.comprehend = None
        self.translate = None
        self.enabled = AWS_TRANSLATION_ENABLED and boto3 is not None
        if self.enabled:
            try:
                cfg = BotoConfig( # type: ignore
                    region_name=REGION,
                    retries={"max_attempts": 3, "mode": "standard"},
                    read_timeout=3,
                    connect_timeout=2,
                )
                if boto3 is None:
                    raise ImportError("boto3 is not available. Ensure it is installed and properly imported.")
                session = boto3.Session()
                self.comprehend = session.client("comprehend", config=cfg)
                self.translate = session.client("translate", config=cfg)
                logger.info("AWS translation enabled in %s", REGION)
            except Exception as e:
                logger.warning("AWS init failed, disabling AWS translation: %s", e)
                self.enabled = False
        else:
            logger.info("AWS translation disabled (flag off or boto3 missing).")

    @staticmethod
    def normalize_language_code(code: str) -> str:
        if not code:
            return "en"
        c = code.lower()
        if c.startswith("en"):
            return "en"
        if c in ("zh-tw", "zh-hant"):
            return "zh-tw"
        if c in ("zh", "zh-cn", "zh-hans"):
            return "zh-cn"
        # default to 'en' instead of region-qualified
        return "en"

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language. Returns normalized code ('en' | 'zh-tw' | 'zh-cn' | other) and confidence.
        Stateless: no shared attributes mutated.
        """
        if not text or not text.strip():
            return "en", 1.0

        txt = _clip_text(text)

        if not self.enabled or self.comprehend is None:
            # Lightweight local heuristic as fallback:
            # if we see many CJK, pick zh; bias to zh-tw if looks Traditional
            cjk = sum(1 for c in txt if '\u4e00' <= c <= '\u9fff')
            alnum = len([c for c in txt if c.isalnum()])
            if alnum == 0 and cjk == 0:
                return "en", 0.5
            if cjk > max(1, 0.3 * max(alnum, 1)):
                return ("zh-tw" if _likely_traditional(txt) else "zh-cn"), 0.8
            return "en", 0.8

        try:
            resp = self.comprehend.detect_dominant_language(Text=txt)
            langs = (resp or {}).get("Languages") or []
            if not langs:
                # fall back to heuristic
                return ("zh-tw" if _likely_traditional(txt) else "en"), 0.3
            top = max(langs, key=lambda x: float(x.get("Score", 0)))
            norm = self.normalize_language_code(str(top.get("LanguageCode", "")))
            conf = float(top.get("Score", 0.0))
            # Comprehend returns 'zh' → refine with script heuristic
            if norm in ("zh-cn", "zh-tw"):
                # leave as is if already tw; if 'zh-cn' but looks Traditional, flip to 'zh-tw'
                if norm == "zh-cn" and _likely_traditional(txt):
                    norm = "zh-tw"
            elif norm == "zh":
                norm = "zh-tw" if _likely_traditional(txt) else "zh-cn"
            return norm, conf
        except (ClientError, BotoCoreError) as e:
            logger.warning("Comprehend detect failed: %s", e)
            return ("zh-tw" if _likely_traditional(txt) else "en"), 0.3

    def translate_to_english(self, text: str, src_lang: str) -> str:
        """Translate to English. If disabled/unavailable, returns input unchanged."""
        if not text:
            return text
        if not self.enabled or self.translate is None:
            return text
        code = src_lang.lower()
        if code.startswith("en"):
            return text
        src = "zh-TW" if code in ("zh-tw", "zh-hant") else ("zh" if code.startswith("zh") else code)
        try:
            res = self.translate.translate_text(
                Text=_clip_text(text),
                SourceLanguageCode=src,
                TargetLanguageCode="en"
            )
            return res.get("TranslatedText") or text
        except (ClientError, BotoCoreError) as e:
            logger.error("Translate->EN failed: %s", e, exc_info=True)
            return text

    def translate_from_english(self, text: str, tgt_lang: str) -> str:
        """Translate from English to target. If disabled/unavailable, returns input unchanged."""
        if not text:
            return text
        if not self.enabled or self.translate is None:
            return text
        code = tgt_lang.lower()
        if code.startswith("en"):
            return text
        tgt = "zh-TW" if code in ("zh-tw", "zh-hant") else ("zh" if code.startswith("zh") else code)
        try:
            res = self.translate.translate_text(
                Text=_clip_text(text),
                SourceLanguageCode="en",
                TargetLanguageCode=tgt
            )
            return res.get("TranslatedText") or text
        except (ClientError, BotoCoreError) as e:
            logger.warning("Translate EN->%s failed: %s", tgt, e)
            return text
