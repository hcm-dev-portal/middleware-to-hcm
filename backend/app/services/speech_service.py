# app/services/speech_service.py
import contextlib
import os
import io
import asyncio
import logging
from typing import AsyncIterator, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, StartStreamTranscriptionRequest

logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_REGION") or "ap-southeast-1"
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")


class PollyService:
    def __init__(self, region_name: str = REGION):
        self.client = boto3.client(
            "polly",
            region_name=region_name,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )

    def synthesize(self, text: str, lang: str = "en-US", voice_id: Optional[str] = None) -> bytes:
        """
        Simple TTS:
          - en-US default voice: Joanna
          - zh-CN default voice: Zhiyu
          - zh-TW default voice: (set a valid Polly TW-zh voice you prefer) fallback to Zhiyu
        """
        if not text or not text.strip():
            raise ValueError("text is required")

        if not voice_id:
            if lang == "zh-CN":
                voice_id = "Zhiyu"
            elif lang == "zh-TW":
                # NOTE: Pick a valid zh-TW voice in your account; fallback to Mandarin (Zhiyu)
                voice_id = "Zhiyu"
            else:
                voice_id = "Joanna"

        try:
            resp = self.client.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId=voice_id,
                Engine="neural"  # fallback handled by Polly if voice not neural
            )
            audio_stream = resp.get("AudioStream")
            return audio_stream.read() if audio_stream else b""
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Polly synthesize error: {e}")
            raise


class TranscribeStreamHandler(TranscriptResultStreamHandler):
    """Forward Transcribe events to the FastAPI WebSocket as JSON strings."""
    def __init__(self, output_stream, websocket):
        super().__init__(output_stream)
        self.websocket = websocket

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for res in results:
            if not res.alternatives:
                continue
            text = res.alternatives[0].transcript
            payload = {
                "type": "transcript",
                "is_final": not res.is_partial,
                "text": text
            }
            await self.websocket.send_json(payload)


class STTService:
    def __init__(self, region_name: str = REGION):
        # Credentials picked from env vars (same as other AWS clients)
        self.client = TranscribeStreamingClient(
            region=region_name,
            aws_access_key=AWS_ACCESS_KEY,
            aws_secret_key=AWS_SECRET_KEY,
        )

    async def stream_transcription(self, websocket, lang: str = "en-US"):
        """
        WebSocket <-> AWS Transcribe Streaming bridge.
        Frontend must send 16kHz, 16-bit PCM little-endian chunks (ArrayBuffer).
        """
        # Map only supported languages
        if lang not in ("en-US", "zh-CN", "zh-TW"):
            lang = "en-US"

        request = StartStreamTranscriptionRequest(
            language_code=lang,
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        stream = await self.client.start_stream_transcription(request)
        handler = TranscribeStreamHandler(stream.output_stream, websocket)

        async def recv_audio():
            try:
                while True:
                    # Each message is an ArrayBuffer -> bytes
                    message = await websocket.receive_bytes()
                    if message == b"__EOS__":
                        await stream.input_stream.end_stream()
                        break
                    await stream.input_stream.send_audio_event(audio_chunk=message)
            except Exception:
                # Client probably closed; end input stream
                with contextlib.suppress(Exception):
                    await stream.input_stream.end_stream()

        async def send_transcripts():
            try:
                await handler.handle_events()
            except Exception:
                # Transcribe stream ended or websocket closed
                pass

        await asyncio.gather(recv_audio(), send_transcripts())
