# app/api/speech_routes.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import io
import logging

from app.services.tts_service import PollyService

logger = logging.getLogger(__name__)

speech_router = APIRouter(prefix="/api", tags=["speech"])
_polly = PollyService()


# --- helpers ---
def normalize_lang(lang: Optional[str]) -> str:
    """
    Normalize various language inputs to the values your TTS service expects.
    Supports English (en-US) and Traditional Chinese (zh-TW).
    Accepts common aliases like 'zh-tw', 'zh_hant', 'zh', etc.
    """
    if not lang:
        return "en-US"
    l = lang.strip().lower()

    # Traditional Chinese (Taiwan) – route all common variants here
    if l in {"zh-tw", "zh_tw", "zh-hant", "zh_hant", "zh"}:
        return "zh-TW"

    # Simplified Chinese (fallback support if ever used)
    if l in {"zh-cn", "zh_cn", "zh-hans", "zh_hans"}:
        return "zh-CN"

    # Default English
    if l in {"en", "en-us", "en_us"}:
        return "en-US"

    # Fallback to English if unknown
    return "en-US"


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    lang: str = "en-US"          # accepts 'zh-tw' etc., normalized internally
    voice_id: Optional[str] = None


@speech_router.post("/tts")
async def tts(request: TTSRequest):
    """
    Returns MP3 audio for the given text (Blob on the frontend).
    - Normalizes lang (supports 'zh-tw' & 'en-us', plus aliases).
    - Uses optional voice_id if provided; otherwise PollyService defaults.
    """
    try:
        text = (request.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        lang = normalize_lang(request.lang)

        audio = _polly.synthesize(
            text=text,
            lang=lang,
            voice_id=request.voice_id,
        )
        if not audio:
            raise HTTPException(status_code=500, detail="TTS failed: empty audio")

        return StreamingResponse(
            io.BytesIO(audio),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="speech.mp3"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS error")
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")
