# ================================================================================
# backend/app/services/aws/translation_service.py
from __future__ import annotations

import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Optional AWS imports
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

REGION = os.getenv("AWS_REGION", "ap-southeast-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")


class AWSTranslationService:
    """Handles language detection and translation via AWS services."""
    
    def __init__(self):
        self.comprehend_client = None
        self.translate_client = None
        self.last_language: Optional[str] = None
        self.last_confidence: Optional[float] = None
        
        self._initialize_aws_clients()
    
    def _initialize_aws_clients(self):
        """Initialize AWS Comprehend and Translate clients if possible."""
        try:
            if boto3 and AWS_ACCESS_KEY and AWS_SECRET_KEY:
                self.comprehend_client = boto3.client(
                    "comprehend", region_name=REGION,
                    aws_access_key_id=AWS_ACCESS_KEY, 
                    aws_secret_access_key=AWS_SECRET_KEY
                )
                self.translate_client = boto3.client(
                    "translate", region_name=REGION,
                    aws_access_key_id=AWS_ACCESS_KEY, 
                    aws_secret_access_key=AWS_SECRET_KEY
                )
                logger.info("AWS clients initialized for region=%s", REGION)
            else:
                logger.info("AWS disabled (missing boto3 or credentials). Defaulting to English.")
        except Exception as e:
            logger.warning("AWS init failed: %s", e)
    
    @staticmethod
    def normalize_language_code(code: str) -> str:
        """Normalize language codes to standard format."""
        if not code:
            return "en-US"
        c = code.lower()
        if c.startswith("en"):
            return "en-US"
        if c in ("zh-tw", "zh-hant"):
            return "zh-TW"
        if c in ("zh", "zh-cn", "zh-hans"):
            return "zh-CN"
        return "en-US"
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of input text."""
        if not (self.comprehend_client and text):
            self.last_language, self.last_confidence = "en-US", 1.0
            return "en-US", 1.0
        
        try:
            resp = self.comprehend_client.detect_dominant_language(Text=text)
            if resp.get("Languages"):
                top = resp["Languages"][0]
                lang = self.normalize_language_code(top["LanguageCode"])
                conf = float(top.get("Score", 0.0))
                self.last_language, self.last_confidence = lang, conf
                return lang, conf
        except ClientError as e:
            logger.error("AWS Comprehend error: %s", e)
        except Exception as e:
            logger.error("Comprehend failure: %s", e, exc_info=True)
        
        self.last_language, self.last_confidence = "en-US", 0.0
        return "en-US", 0.0
    
    def translate_to_english(self, text: str, src_lang: str) -> str:
        """Translate text to English if not already English."""
        if src_lang == "en-US" or not (self.translate_client and text):
            return text
        
        code = "zh-TW" if src_lang == "zh-TW" else "zh"
        try:
            result = self.translate_client.translate_text(
                Text=text, 
                SourceLanguageCode=code, 
                TargetLanguageCode="en"
            )
            return result["TranslatedText"]
        except Exception as e:
            logger.warning("Translation to English failed: %s", e)
            return text
    
    def translate_from_english(self, text: str, tgt_lang: str) -> str:
        """Translate text from English to target language."""
        if tgt_lang == "en-US" or not (self.translate_client and text):
            return text
        
        code = "zh-TW" if tgt_lang == "zh-TW" else "zh"
        try:
            result = self.translate_client.translate_text(
                Text=text, 
                SourceLanguageCode="en", 
                TargetLanguageCode=code
            )
            return result["TranslatedText"]
        except Exception as e:
            logger.warning("Translation from English failed: %s", e)
            return text
