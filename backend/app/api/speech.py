# app/routes/speech.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
import logging
import contextlib
from app.services.speech_service import PollyService, STTService

router = APIRouter(prefix="/api", tags=["speech"])
logger = logging.getLogger(__name__)

_polly = PollyService()
_stt = STTService()

@router.post("/tts")
def tts(text: str, lang: str = "en-US", voiceId: str | None = None):
    """
    Returns MP3 audio for given text.
    """
    try:
        audio_bytes = _polly.synthesize(text=text, lang=lang, voice_id=voiceId)
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="TTS failed")
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="speech.mp3"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

@router.websocket("/ws/stt")
async def stt_ws(websocket: WebSocket, lang: str = "en-US"):
    """
    WebSocket for streaming speech-to-text.
    Client sends PCM 16kHz, 16-bit LE buffers; receives JSON transcripts.
    """
    await websocket.accept()
    try:
        await _stt.stream_transcription(websocket, lang=lang)
    except WebSocketDisconnect:
        logger.info("Client disconnected from STT WebSocket")
    except Exception as e:
        logger.error(f"STT stream error: {e}")
        # send best-effort error
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
