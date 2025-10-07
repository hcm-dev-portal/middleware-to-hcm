# app/services/tts_service.py
import os
import io
import logging
from typing import Optional

from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, ClientError

load_dotenv()
logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")  # optional


class PollyService:
    """Thin wrapper over Amazon Polly for TTS (MP3 output)."""

    def __init__(self, region_name: str = AWS_REGION):
        # You can omit explicit creds to use default chain (env/role/instance profile)
        self.client = boto3.client(
            "polly",
            region_name=region_name,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            aws_session_token=AWS_SESSION_TOKEN,
        )

    def synthesize(
        self, text: str, lang: str = "en-US", voice_id: Optional[str] = None
    ) -> bytes:
        """
        Returns MP3 bytes for the given text.
        Default voices:
          - en-US -> Joanna
          - zh-CN -> Zhiyu
          - zh-TW -> Zhiyu (fallback; pick your preferred TW voice if available)
        """
        if not text or not text.strip():
            raise ValueError("text is required")

        if not voice_id:
            if lang == "zh-CN":
                voice_id = "Zhiyu"
            elif lang == "zh-TW":
                voice_id = "Zhiyu"  # change if you prefer another TW voice
            else:
                voice_id = "Joanna"

        try:
            resp = self.client.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId=voice_id,
                Engine="neural",  # Polly will fallback if not supported
            )
            stream = resp.get("AudioStream")
            return stream.read() if stream else b""
        except (BotoCoreError, ClientError) as e:
            logger.error("Polly synthesize error: %s", e)
            raise
