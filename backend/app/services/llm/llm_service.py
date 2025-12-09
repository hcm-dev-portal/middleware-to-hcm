# backend/app/services/llm/llm_service.py
"""
LLM Service for Leave AI Assistant - AWS Bedrock Edition
=========================================================

Handles all LLM calls via AWS Bedrock (Claude) for:
1. SQL generation from natural language
2. SQL repair when errors occur
3. Explanation generation (summarizing results in Chinese)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# .env loading (robust, with logging)
# ──────────────────────────────────────────────────────────────────────


def _load_env_for_llm() -> None:
    """
    Try to load .env from a few likely locations.
    """
    here = Path(__file__).resolve()
    candidates: List[Path] = []

    try:
        project_root = here.parents[4]
        candidates.append(project_root / ".env")
    except Exception:
        pass

    try:
        backend_root = here.parents[3]
        candidates.append(backend_root / ".env")
    except Exception:
        pass

    for p in candidates:
        try:
            if p.is_file():
                load_dotenv(dotenv_path=p, override=False)
                logger.info("[LLMService] Loaded .env from %s", p)
                return
        except Exception as e:
            logger.warning(
                "[LLMService] Failed to load .env from %s: %s: %s",
                p,
                type(e).__name__,
                e,
            )

    loaded = False
    try:
        loaded = load_dotenv(override=False)
    except Exception as e:
        logger.warning(
            "[LLMService] Default load_dotenv() failed: %s: %s", type(e).__name__, e
        )

    if loaded:
        logger.info(
            "[LLMService] Loaded .env using default search path (starting from CWD)."
        )
    else:
        logger.warning(
            "[LLMService] No .env file found via explicit paths or default search."
        )


# Load env **before** reading any env vars
_load_env_for_llm()

# ──────────────────────────────────────────────────────────────────────
# AWS Bedrock Configuration
# ──────────────────────────────────────────────────────────────────────
Language = Literal["zh-tw", "en"]

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", 
    "anthropic.claude-3-sonnet-20240229-v1:0"
)
# Optional: Use inference profile ARN for cross-region inference
BEDROCK_INFERENCE_PROFILE_ID = (
    os.getenv("BEDROCK_INFERENCE_PROFILE_ID")
    or os.getenv("BEDROCK_INFERENCE_PROFILE_ARN")
)

# Bedrock settings
BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))
BEDROCK_TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", "0.1"))

logger.info("[LLMService] AWS_REGION=%s", AWS_REGION)
logger.info("[LLMService] BEDROCK_MODEL_ID=%s", BEDROCK_MODEL_ID)
if BEDROCK_INFERENCE_PROFILE_ID:
    logger.info("[LLMService] BEDROCK_INFERENCE_PROFILE_ID=%s", BEDROCK_INFERENCE_PROFILE_ID)

# ──────────────────────────────────────────────────────────────────────
# Optional boto3 import (soft dependency)
# ──────────────────────────────────────────────────────────────────────
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
    logger.info("[LLMService] boto3 is available.")
except ImportError:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore
    NoCredentialsError = Exception  # type: ignore
    BOTO3_AVAILABLE = False
    logger.warning(
        "[LLMService] boto3 is not installed; Bedrock LLM integration will be disabled."
    )

# ──────────────────────────────────────────────────────────────────────
# Optional DB exception types (from your db_service)
# ──────────────────────────────────────────────────────────────────────
try:
    from app.services.db_service import (
        DatabaseQueryError as DBServiceQueryError,
        DatabaseSyntaxError as DBServiceSyntaxError,
        DatabaseTableNotFoundError as DBServiceTableNotFoundError,
        DatabaseColumnNotFoundError as DBServiceColumnNotFoundError,
        DatabaseDataError as DBServiceDataError,
        DatabaseIntegrityError as DBServiceIntegrityError,
        DatabaseOperationalError as DBServiceOperationalError,
        DatabaseTimeoutError as DBServiceTimeoutError,
        DatabaseConnectionError as DBServiceConnectionError,
        PermissionDeniedError as DBServicePermissionDeniedError,
    )
except ImportError:
    class DBServiceQueryError(Exception):
        pass

    DBServiceSyntaxError = (
        DBServiceTableNotFoundError
    ) = (
        DBServiceColumnNotFoundError
    ) = (
        DBServiceDataError
    ) = (
        DBServiceIntegrityError
    ) = (
        DBServiceOperationalError
    ) = DBServiceQueryError
    DBServiceTimeoutError = (
        DBServiceConnectionError
    ) = DBServicePermissionDeniedError = DBServiceQueryError  # type: ignore
    logger.warning(
        "[LLMService] app.services.db_service not importable; using generic DB error types."
    )


# ──────────────────────────────────────────────────────────────────────
# Lightweight language detection (kept for logging/metadata)
# ──────────────────────────────────────────────────────────────────────
def detect_query_language(text: str) -> Language:
    """
    Very lightweight zh-tw vs en detector.
    """
    if not text or not text.strip():
        return "en"

    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin_num = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isdigit()))

    if chinese_chars >= 2 and chinese_chars >= latin_num:
        return "zh-tw"
    if any(k in text for k in ["請假", "考勤", "部門", "員工", "今天", "現在", "統計", "趨勢"]):
        return "zh-tw"
    return "en"


# ──────────────────────────────────────────────────────────────────────
# Bedrock LLM Client
# ──────────────────────────────────────────────────────────────────────
class BedrockClient:
    """
    AWS Bedrock client wrapper for Claude models.
    
    Handles:
    - Client initialization with proper error handling
    - Message formatting for Anthropic Claude models
    - Response parsing and error handling
    - Automatic model ID selection based on availability
    
    Model Selection Priority:
    1. BEDROCK_INFERENCE_PROFILE_ID (for cross-region or Opus 4.5)
    2. BEDROCK_MODEL_ID (for on-demand models like Claude 3 Sonnet/Haiku)
    
    Note: Claude Opus 4.5 and some newer models REQUIRE an inference profile.
    Cross-region inference profiles start with region prefix (e.g., "apac.", "eu.", "us.")
    """
    
    # Models that support on-demand invocation (no inference profile needed)
    ON_DEMAND_MODELS = [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
    ]
    
    # Models that REQUIRE inference profile (cannot use on-demand)
    INFERENCE_PROFILE_REQUIRED = [
        "anthropic.claude-opus-4-5-20251101-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
    ]
    
    # Cross-region inference profile prefixes
    CROSS_REGION_PREFIXES = ["apac.", "eu.", "us."]
    
    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
        inference_profile_id: Optional[str] = None,
        max_tokens: int = BEDROCK_MAX_TOKENS,
        temperature: float = BEDROCK_TEMPERATURE,
    ):
        self.region = region or AWS_REGION
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        self._initialized = False
        self._init_error: Optional[str] = None
        
        # Determine model target
        self.inference_profile_id = inference_profile_id or BEDROCK_INFERENCE_PROFILE_ID
        self.model_id = model_id or BEDROCK_MODEL_ID
        
        # Select the actual target to use
        self._model_target = self._select_model_target()
        
        self._initialize_client()
    
    def _select_model_target(self) -> str:
        """
        Select the appropriate model target based on configuration.
        
        Returns the model ID or inference profile ARN to use.
        """
        # If inference profile is set, use it (supports all models including cross-region)
        if self.inference_profile_id:
            # Check if it's a cross-region inference profile (starts with apac., eu., us., etc.)
            is_cross_region = any(
                self.inference_profile_id.startswith(prefix) 
                for prefix in self.CROSS_REGION_PREFIXES
            )
            if is_cross_region:
                logger.info(
                    "[BedrockClient] Using cross-region inference profile: %s",
                    self.inference_profile_id,
                )
            else:
                logger.info(
                    "[BedrockClient] Using inference profile: %s",
                    self.inference_profile_id,
                )
            return self.inference_profile_id
        
        # Check if the model requires an inference profile
        if self.model_id in self.INFERENCE_PROFILE_REQUIRED:
            logger.warning(
                "[BedrockClient] Model %s requires an inference profile but none configured. "
                "Falling back to claude-3-5-sonnet.",
                self.model_id,
            )
            # Fall back to a supported on-demand model
            fallback = "anthropic.claude-3-5-sonnet-20241022-v2:0"
            self.model_id = fallback
            return fallback
        
        # Check if model supports on-demand
        if self.model_id in self.ON_DEMAND_MODELS:
            logger.info(
                "[BedrockClient] Using on-demand model: %s",
                self.model_id,
            )
            return self.model_id
        
        # Unknown model - try it anyway but warn
        logger.warning(
            "[BedrockClient] Unknown model %s - attempting on-demand invocation",
            self.model_id,
        )
        return self.model_id
    
    def _initialize_client(self) -> None:
        """Initialize the Bedrock runtime client."""
        if not BOTO3_AVAILABLE:
            self._init_error = "boto3 is not installed"
            logger.error("[BedrockClient] %s", self._init_error)
            return
        
        try:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
            )
            self._initialized = True
            logger.info(
                "[BedrockClient] Initialized successfully. Target=%s Region=%s",
                self._model_target,
                self.region,
            )
        except NoCredentialsError as e:
            self._init_error = f"AWS credentials not found: {e}"
            logger.error("[BedrockClient] %s", self._init_error)
        except Exception as e:
            self._init_error = f"Failed to initialize: {type(e).__name__}: {e}"
            logger.error("[BedrockClient] %s", self._init_error)
    
    def is_available(self) -> bool:
        """Check if the client is ready for use."""
        return self._initialized and self.client is not None
    
    def get_init_error(self) -> Optional[str]:
        """Return initialization error message if any."""
        return self._init_error
    
    def get_model_target(self) -> str:
        """Return the actual model target being used."""
        return self._model_target
    
    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
    ) -> str:
        """
        Invoke the Bedrock Claude model.
        
        Args:
            system_prompt: System instructions for the model
            user_message: User's message/query
            context: Optional context string for logging
            
        Returns:
            Model's response text
        """
        if not self.is_available():
            logger.error(
                "[BedrockClient] invoke() called but client not available: %s",
                self._init_error,
            )
            return ""
        
        t0 = time.perf_counter()
        
        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": 0.9,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_message}],
                    }
                ],
                "stop_sequences": ["\n\nHuman:", "\n\nUser:"],
            }
            
            # Use the pre-selected model target
            response = self.client.invoke_model(
                modelId=self._model_target,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )
            
            payload = json.loads(response["body"].read())
            content_blocks = payload.get("content", [])
            
            if not content_blocks:
                logger.warning(
                    "[BedrockClient] Empty response from model: %s",
                    json.dumps(payload)[:200],
                )
                return ""
            
            result = content_blocks[0].get("text", "").strip()
            
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "[BedrockClient] invoke OK: ctx=%s model=%s ms=%d len=%d",
                context,
                self._model_target[:50],
                elapsed_ms,
                len(result),
            )
            
            return result
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            logger.error(
                "[BedrockClient] ClientError [%s]: %s (ctx=%s, model=%s)",
                error_code,
                error_msg,
                context,
                self._model_target,
            )
            return ""
            
        except Exception as e:
            logger.error(
                "[BedrockClient] invoke failed: %s: %s (ctx=%s)",
                type(e).__name__,
                e,
                context,
            )
            return ""


# ──────────────────────────────────────────────────────────────────────
# Core service – zh-tw monolingual behaviour
# ──────────────────────────────────────────────────────────────────────
class LLMService:
    """
    Core LLM orchestration service for the Leave AI Assistant.
    
    Now powered by AWS Bedrock (Claude) instead of OpenAI.

    Responsibilities:
      - 接收自然語言問題（主要為繁體中文 zh-tw）。
      - 使用上游檢索層提供的 `intent_context`。
      - 產生安全的 **SELECT-only** T-SQL。
      - 於資料庫錯誤時用 LLM 做查詢修復。
      - 呼叫 `db_service.run_select(...)` 執行最終 SQL。
      - 輸出給主管看的繁體中文說明（explanation_zh）。
    """

    # Keywords that we will block in generated SQL for safety
    _PROHIBITED_RE = re.compile(
        r"(?is)\b(insert|update|delete|merge|drop|alter|create|truncate|exec|execute|grant|revoke)\b"
    )
    _FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)
    _FIRST_SELECT_RE = re.compile(r"(?is)\bwith\b[\s\S]+?\bselect\b|\bselect\b")
    
    # Pattern to detect unbound SQL parameters like @today, @startDate, etc.
    _UNBOUND_PARAM_RE = re.compile(r"@(\w+)")

    def __init__(
        self,
        model_id: Optional[str] = None,
        temperature: float = BEDROCK_TEMPERATURE,
        region: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or BEDROCK_MODEL_ID
        self.temperature = temperature
        self.region = region or AWS_REGION

        # Initialize Bedrock client
        self.bedrock = BedrockClient(
            region=self.region,
            model_id=self.model_id,
            temperature=self.temperature,
        )
        
        self.llm_enabled = self.bedrock.is_available()

        logger.info(
            "[LLMService.__init__] model_id=%s, temp=%.2f, region=%s, llm_enabled=%s",
            self.model_id,
            self.temperature,
            self.region,
            self.llm_enabled,
        )

    # ──────────────────────────────────────────────────────────────────
    # LLM availability check
    # ──────────────────────────────────────────────────────────────────
    def _is_llm_available(self) -> bool:
        if not self.llm_enabled:
            error = self.bedrock.get_init_error() or "Unknown initialization error"
            logger.error("[LLMService] LLM unavailable: %s", error)
            return False
        return True

    # ──────────────────────────────────────────────────────────────────
    # Prompt builders (zh-tw only)
    # ──────────────────────────────────────────────────────────────────
    def _build_sql_generation_prompt(
        self,
        question: str,
        schema: str,
        join_hints: str,
        intent_context: Dict[str, Any],
        table_whitelist_text: str,
    ) -> Tuple[str, str]:
        """
        Build system and user prompts for SQL generation.
        Returns (system_prompt, user_message).
        """
        slots = intent_context.get("slots", {}) or {}
        few_shot_sql = (
            intent_context.get("few_shot_sql")
            or intent_context.get("example_sql")
            or ""
        )
        intent_debug = self._intent_debug_string(intent_context)
        today_info = self._get_today_info()
        
        system_prompt = f"""你是一位專精於人資請假與考勤資料的 T-SQL 專家，負責產生安全的查詢語句。
請只回傳 **一個** Microsoft SQL Server (T-SQL) 查詢，且必須是 **僅限 SELECT**，可以使用 CTE。

（下列內容由上游檢索系統提供，已根據公司實際資料庫 schema 與 recipe 精心設計）

意圖 (intent)：
{intent_debug}

Few-shot 參考 SQL（若有，優先參考其欄位與 JOIN 寫法）：
{few_shot_sql}

【重要】日期處理規則：
- 嚴禁使用 SQL 參數變數（如 @today、@startDate 等），因為本系統不支援參數綁定。
- 若需要「今天」的日期，請使用 CAST(GETDATE() AS DATE)。
- 若需要「本週」，請使用 DATEADD(DAY, -DATEPART(WEEKDAY, GETDATE())+1, CAST(GETDATE() AS DATE)) 作為週一。
- 若需要「本月」，請使用 DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)。
- 建議使用 CAST(column AS DATE) 搭配 BETWEEN 或 >= / < 做日期過濾。

系統提供的日期資訊（可直接參考）：
- 今天日期: {today_info["today_date"]}
- 今天星期: {today_info["today_weekday"]}

業務規則（Leave AI）：
- WORKDATE 為發生日；STARTDATE/ENDDATE 為請假區間。
- 統計「已批准」請假時，請加上 VALIDATED = 1 條件（若適用）。
- 只有在需要顯示部門/單位資訊時再 JOIN 組織表或人員表。

T-SQL 安全規範：
- 嚴禁使用 INSERT/UPDATE/DELETE/MERGE/ALTER/DROP/CREATE/TRUNCATE/EXEC 等指令。
- 只能產生一個查詢語句，不得包含多個批次或 GO。
- 別名必須先在 FROM/JOIN 宣告後再使用。
- GROUP BY 必須包含所有非聚合欄位。
- 不可使用 @ 開頭的變數。

可用資料庫結構 (schema 摘要)：
{schema}

建議 JOIN 關聯說明：
{join_hints}

若有提供 table_whitelist，請只使用其中出現的資料表：
{table_whitelist_text}"""

        user_message = f"""使用者問題：{question}

已抽取的 slots (JSON)：{json.dumps(slots, ensure_ascii=False)}

請只回傳最終 SQL 查詢本體（不要加 markdown、不要加額外說明或註解、不要使用 @ 變數）。"""

        return system_prompt, user_message

    def _build_sql_repair_prompt(
        self,
        failed_sql: str,
        error_summary: str,
        schema: str,
        join_hints: str,
        intent_context: Dict[str, Any],
        table_whitelist_text: str,
    ) -> Tuple[str, str]:
        """
        Build system and user prompts for SQL repair.
        Returns (system_prompt, user_message).
        """
        slots = intent_context.get("slots", {}) or {}
        few_shot_sql = (
            intent_context.get("few_shot_sql")
            or intent_context.get("example_sql")
            or ""
        )
        intent_debug = self._intent_debug_string(intent_context)
        today_info = self._get_today_info()
        
        system_prompt = f"""你要協助修復一段失敗的 Microsoft SQL Server (T-SQL) 查詢。
請輸出一個修正後的 **僅限 SELECT** 的查詢，維持原本意圖，不得新增 DML/DDL 指令。

務必遵守：
- 別名先在 FROM/JOIN 宣告再使用。
- GROUP BY 包含所有非聚合欄位。
- 【重要】不可使用任何 @ 開頭的變數（如 @today、@startDate），請改用 T-SQL 內建日期函數。
- 若錯誤訊息提到「必須宣告純量變數」，表示原 SQL 使用了未定義的 @ 變數，請將其替換為對應的 T-SQL 函數。

日期替換指引：
- @today → CAST(GETDATE() AS DATE)
- @now → GETDATE()
- @startOfWeek → DATEADD(DAY, -DATEPART(WEEKDAY, GETDATE())+1, CAST(GETDATE() AS DATE))
- @startOfMonth → DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)
- @startOfYear → DATEFROMPARTS(YEAR(GETDATE()), 1, 1)

系統提供的日期資訊：
- 今天日期: {today_info["today_date"]}

意圖 (intent)：
{intent_debug}

Few-shot 參考 SQL：
{few_shot_sql}

可用 schema：
{schema}

建議 JOIN 關係：
{join_hints}

允許使用之資料表（若有提供）：
{table_whitelist_text}"""

        user_message = f"""資料庫錯誤訊息：
{error_summary}

原始失敗的 SQL：
{failed_sql}

請只回傳修正後的 SQL，本體即可（不要使用 @ 變數）。"""

        return system_prompt, user_message

    def _build_explanation_prompt(
        self,
        question: str,
        row_count: int,
        columns: List[str],
        aggregates: Dict[str, Any],
        sample_text: str,
    ) -> Tuple[str, str]:
        """
        Build system and user prompts for explanation generation.
        Returns (system_prompt, user_message).
        """
        system_prompt = """你是一位服務公司高階主管的人資資料分析師。
請根據提供的欄位、統計摘要與樣本資料，用繁體中文寫出簡潔的說明。

嚴格規則：
- 僅可使用提供的欄位名稱、聚合統計與樣本資料，不可自行杜撰欄位或數值。
- 若資料不足以回答問題，請在摘要中明確說明。
- 不要輸出 SQL 或程式碼。

輸出格式（Markdown）：
### 摘要
• 2–3 點最重要的數字或結論（需與問題直接相關）。

### 主要觀察
• 2–4 點描述分布、趨勢、異常值或部門/假別等類別的重點。

### 風險與建議
• 1–3 點給主管的具體建議（例如追蹤對象、檢查政策、設定門檻）。

### 資料品質說明
• 1–2 點說明樣本限制（例如資料期間、欄位缺漏、筆數過少）。"""

        cols_joined = ", ".join(columns) if columns else "(none)"
        aggs_json = json.dumps(aggregates or {}, ensure_ascii=False)
        
        user_message = f"""問題：{question}
資料筆數：{row_count}
欄位：{cols_joined}
統計摘要 (JSON)：{aggs_json}
資料樣本（截斷顯示）：
{sample_text}"""

        return system_prompt, user_message

    # ──────────────────────────────────────────────────────────────────
    # Date helpers for SQL generation
    # ──────────────────────────────────────────────────────────────────
    def _get_today_info(self) -> Dict[str, str]:
        """
        Returns today's date information for use in prompts and SQL substitution.
        """
        today = date.today()
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return {
            "today_date": today.isoformat(),
            "today_weekday": weekday_names[today.weekday()],
        }

    def _substitute_date_parameters(self, sql: str) -> str:
        """
        Substitutes common @parameter placeholders with T-SQL expressions.
        This is a safety net in case the LLM still generates @variables.
        """
        if not sql:
            return sql

        today_str = date.today().isoformat()
        
        substitutions = [
            (r"@today\b", f"CAST('{today_str}' AS DATE)"),
            (r"@currentDate\b", f"CAST('{today_str}' AS DATE)"),
            (r"@now\b", "GETDATE()"),
            (r"@startOfWeek\b", "DATEADD(DAY, -DATEPART(WEEKDAY, GETDATE())+1, CAST(GETDATE() AS DATE))"),
            (r"@endOfWeek\b", "DATEADD(DAY, 7-DATEPART(WEEKDAY, GETDATE()), CAST(GETDATE() AS DATE))"),
            (r"@startOfMonth\b", "DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)"),
            (r"@endOfMonth\b", "EOMONTH(GETDATE())"),
            (r"@startOfYear\b", "DATEFROMPARTS(YEAR(GETDATE()), 1, 1)"),
            (r"@endOfYear\b", "DATEFROMPARTS(YEAR(GETDATE()), 12, 31)"),
        ]

        result = sql
        for pattern, replacement in substitutions:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        if result != sql:
            logger.info(
                "SQL_PARAM_SUBSTITUTION: replaced @variables in SQL (original had unbound params)"
            )

        return result

    def _check_for_unbound_parameters(self, sql: str) -> Optional[str]:
        """
        Checks if SQL still contains unbound @parameters after substitution.
        Returns error message if found, None otherwise.
        """
        if not sql:
            return None
            
        matches = self._UNBOUND_PARAM_RE.findall(sql)
        if matches:
            unbound = [m for m in matches if not m.startswith('@')]
            if unbound:
                return f"SQL contains unbound parameters: @{', @'.join(unbound)}"
        return None

    # ──────────────────────────────────────────────────────────────────
    # SQL + DB execution with repair
    # ──────────────────────────────────────────────────────────────────
    def run_query_with_llm_repair(
        self,
        *,
        db_service: Any,
        user_question: str,
        schema: str,
        join_hints: str,
        intent_context: Optional[Dict[str, Any]] = None,
        max_rows: int = 1000,
        query_timeout: int = 10,
        max_attempts: int = 3,
    ) -> Tuple[List[Tuple[Any, ...]], List[str], str, int]:
        """
        Pipeline:
          1) Detect language (僅供 logging / metadata)。
          2) 使用 zh-tw 提示詞產生 SQL（含 recipes/few-shot）。
          3) 執行 SQL；若 DB 錯誤，將錯誤摘要丟回 LLM 修復。
          4) 最多 max_attempts 輪（LLM + DB）。

        Returns (rows, columns, sql, attempts).
        """
        question = (user_question or "").strip()
        detected_lang: Language = detect_query_language(question)
        ctx = intent_context or {}

        if not self._is_llm_available():
            logger.error(
                "RUN_QUERY_LLM_UNAVAILABLE: detected_lang=%s q=%r",
                detected_lang,
                question[:120],
            )
            return [], [], "", 0

        table_whitelist: List[str] = list(
            ctx.get("table_whitelist") or ctx.get("tables") or []
        )
        whitelist_text = ", ".join(table_whitelist) if table_whitelist else "(no restriction)"

        attempts = 0
        sql = ""
        rows: List[Tuple[Any, ...]] = []
        cols: List[str] = []
        last_error_summary = "initial generation (no DB error yet)"

        while attempts < max_attempts:
            attempts += 1

            if attempts == 1:
                raw = self._generate_sql_raw(
                    question=question,
                    schema=schema,
                    join_hints=join_hints,
                    intent_context=ctx,
                    table_whitelist_text=whitelist_text,
                )
            else:
                raw = self._repair_sql_raw(
                    failed_sql=sql,
                    error_summary=last_error_summary,
                    schema=schema,
                    join_hints=join_hints,
                    intent_context=ctx,
                    table_whitelist_text=whitelist_text,
                )

            sql = self._finalize_sql(raw)

            if not sql:
                logger.warning("RUN_QUERY_ATTEMPT_EMPTY_SQL: attempt=%d", attempts)
                continue

            if self._PROHIBITED_RE.search(sql):
                logger.warning(
                    "RUN_QUERY_PROHIBITED_KEYWORD: attempt=%d sql_prefix=%r",
                    attempts,
                    sql[:200],
                )
                sql = ""
                continue

            sql = self._substitute_date_parameters(sql)

            param_error = self._check_for_unbound_parameters(sql)
            if param_error:
                logger.warning(
                    "RUN_QUERY_UNBOUND_PARAMS: attempt=%d error=%s sql_prefix=%r",
                    attempts,
                    param_error,
                    sql[:200],
                )
                last_error_summary = param_error
                continue

            if table_whitelist and not self._tables_respect_whitelist(sql, table_whitelist):
                logger.warning(
                    "RUN_QUERY_WHITELIST_VIOLATION: attempt=%d sql_prefix=%r",
                    attempts,
                    sql[:200],
                )
                sql = ""
                last_error_summary = "Table whitelist violation in generated SQL"
                continue

            try:
                rows, cols = db_service.run_select(
                    sql,
                    params=None,
                    max_rows=max_rows,
                    query_timeout=query_timeout,
                )
                logger.info(
                    "RUN_QUERY_OK: attempts=%d rows=%d cols=%d",
                    attempts,
                    len(rows or []),
                    len(cols or []),
                )
                return rows or [], cols or [], sql, attempts
            except DBServiceSyntaxError as e:
                last_error_summary = self._format_db_error_for_repair(e, "syntax")
                logger.warning(
                    "RUN_QUERY_SYNTAX_ERROR: attempt=%d err=%s sql_prefix=%r",
                    attempts,
                    last_error_summary,
                    sql[:200],
                )
            except DBServiceTableNotFoundError as e:
                last_error_summary = self._format_db_error_for_repair(e, "table_not_found")
                logger.warning(
                    "RUN_QUERY_TABLE_ERROR: attempt=%d err=%s",
                    attempts,
                    last_error_summary,
                )
            except DBServiceColumnNotFoundError as e:
                last_error_summary = self._format_db_error_for_repair(e, "column_not_found")
                logger.warning(
                    "RUN_QUERY_COLUMN_ERROR: attempt=%d err=%s",
                    attempts,
                    last_error_summary,
                )
            except DBServiceTimeoutError as e:
                last_error_summary = self._format_db_error_for_repair(e, "timeout")
                logger.warning(
                    "RUN_QUERY_TIMEOUT: attempt=%d err=%s",
                    attempts,
                    last_error_summary,
                )
            except DBServiceQueryError as e:
                last_error_summary = self._format_db_error_for_repair(e, "general")
                logger.warning(
                    "RUN_QUERY_DB_ERROR: attempt=%d err=%s sql_prefix=%r",
                    attempts,
                    last_error_summary,
                    sql[:200],
                )

        logger.error(
            "RUN_QUERY_EXHAUSTED: max_attempts=%d last_error=%s", max_attempts, last_error_summary
        )
        return [], [], sql, attempts

    def _format_db_error_for_repair(self, error: Exception, category: str) -> str:
        """
        Formats DB error message with helpful context for the repair prompt.
        """
        base_msg = f"{type(error).__name__}: {str(error)}"
        
        hints = {
            "syntax": "（提示：可能是語法錯誤、未宣告的變數、或欄位別名問題）",
            "table_not_found": "（提示：資料表名稱可能拼錯或不存在於 schema 中）",
            "column_not_found": "（提示：欄位名稱可能拼錯或該欄位不存在於指定資料表）",
            "timeout": "（提示：查詢太慢，考慮加上 TOP 限制、減少 JOIN、或加上索引欄位條件）",
            "general": "",
        }
        
        hint = hints.get(category, "")
        
        if "@" in str(error) and "宣告" in str(error):
            hint = "（重要：這是因為使用了 @ 變數但未定義。請將所有 @xxx 變數替換為 T-SQL 日期函數如 CAST(GETDATE() AS DATE)）"
        
        return f"{base_msg} {hint}".strip()

    # ──────────────────────────────────────────────────────────────────
    # Public entrypoint for HTTP controllers
    # ──────────────────────────────────────────────────────────────────
    def answer_question(
        self,
        db_service: Any,
        user_question: str,
        schema: str,
        join_hints: str,
        *,
        intent_context: Optional[Dict[str, Any]] = None,
        table_whitelist: Optional[List[str]] = None,
        max_rows: int = 1000,
        query_timeout: int = 10,
        max_attempts: int = 3,
        allow_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline for HTTP controllers:
          - 使用 run_query_with_llm_repair 產生 SQL + 執行 DB。
          - 計算基本統計。
          - 產出 zh-tw 說明給主管閱讀。
        """
        user_question = (user_question or "").strip()
        if not user_question:
            return {
                "question": user_question,
                "language_detected": "zh-tw",
                "sql": None,
                "rows": [],
                "columns": [],
                "attempts": 0,
                "aggregates": {
                    "row_count": 0,
                    "unique_people": None,
                    "by_leave_type": {},
                    "total_hours": None,
                },
                "explanation_zh": (
                    "### 摘要\n"
                    "• 問題不可為空白。\n\n"
                    "### 資料品質說明\n"
                    "• 請輸入有效的查詢問題。"
                ),
                "success": False,
                "error": "問題不可為空白。",
                "error_category": "validation_error",
            }

        detected_lang: Language = detect_query_language(user_question)
        logger.info("LLM_PIPELINE_START: lang=%s q=%r", detected_lang, user_question[:120])

        if not self._is_llm_available():
            init_error = self.bedrock.get_init_error() or "Unknown error"
            msg = f"LLM backend not available（{init_error}）。"
            logger.error("LLM_PIPELINE_ABORT: %s", msg)
            return {
                "question": user_question,
                "language_detected": detected_lang,
                "sql": None,
                "rows": [],
                "columns": [],
                "attempts": 0,
                "aggregates": {
                    "row_count": 0,
                    "unique_people": None,
                    "by_leave_type": {},
                    "total_hours": None,
                },
                "explanation_zh": (
                    "### 摘要\n"
                    "• 系統目前無法使用 LLM 服務，請稍後再試或聯絡系統管理員。\n\n"
                    "### 資料品質說明\n"
                    "• AWS Bedrock 設定或連線可能有問題。"
                ),
                "success": False,
                "error": msg,
                "error_category": "llm_unavailable",
            }

        ctx = intent_context or {}
        if table_whitelist:
            ctx = dict(ctx)
            ctx["table_whitelist"] = table_whitelist

        rows, cols, sql, attempts_gen = self.run_query_with_llm_repair(
            db_service=db_service,
            user_question=user_question,
            schema=schema,
            join_hints=join_hints,
            intent_context=ctx,
            max_rows=max_rows,
            query_timeout=query_timeout,
            max_attempts=max_attempts,
        )

        if not sql:
            msg = "LLM 無法產生或修復有效 SQL。"
            logger.error("LLM_PIPELINE_SQL_EMPTY: %s", msg)
            return {
                "question": user_question,
                "language_detected": detected_lang,
                "sql": None,
                "rows": [],
                "columns": [],
                "attempts": attempts_gen,
                "aggregates": {
                    "row_count": 0,
                    "unique_people": None,
                    "by_leave_type": {},
                    "total_hours": None,
                },
                "explanation_zh": (
                    "### 摘要\n"
                    "• LLM 未能產生有效的查詢語句。\n\n"
                    "### 資料品質說明\n"
                    "• 請確認問題描述是否明確，或稍後再試。"
                ),
                "success": False,
                "error": msg,
                "error_category": "llm_sql_generation_failed",
            }

        aggregates = self._compute_basic_aggregates(rows, cols)
        sample_text = self._format_sample_rows(rows, cols, max_rows=5)

        explanation_zh = self._generate_explanation(
            question=user_question,
            row_count=len(rows),
            columns=cols,
            aggregates=aggregates,
            sample_text=sample_text,
        )

        return {
            "question": user_question,
            "language_detected": detected_lang,
            "sql": sql,
            "rows": rows,
            "columns": cols,
            "attempts": attempts_gen,
            "aggregates": aggregates,
            "explanation_zh": explanation_zh,
            "intent_context": ctx,
            "success": True,
        }

    # ──────────────────────────────────────────────────────────────────
    # SQL generation + repair (Bedrock calls)
    # ──────────────────────────────────────────────────────────────────
    def _generate_sql_raw(
        self,
        question: str,
        schema: str,
        join_hints: str,
        intent_context: Dict[str, Any],
        table_whitelist_text: str,
    ) -> str:
        """
        Generate SQL using Bedrock.
        """
        if not self._is_llm_available():
            logger.warning("SQL_GEN_RAW: LLM not available.")
            return ""

        system_prompt, user_message = self._build_sql_generation_prompt(
            question=question,
            schema=schema,
            join_hints=join_hints,
            intent_context=intent_context,
            table_whitelist_text=table_whitelist_text,
        )

        return self.bedrock.invoke(
            system_prompt=system_prompt,
            user_message=user_message,
            context="sql_gen",
        )

    def _repair_sql_raw(
        self,
        failed_sql: str,
        error_summary: str,
        schema: str,
        join_hints: str,
        intent_context: Dict[str, Any],
        table_whitelist_text: str,
    ) -> str:
        """
        Repair SQL using Bedrock.
        """
        if not self._is_llm_available():
            logger.warning("SQL_REPAIR_RAW: LLM not available.")
            return failed_sql

        system_prompt, user_message = self._build_sql_repair_prompt(
            failed_sql=failed_sql,
            error_summary=error_summary,
            schema=schema,
            join_hints=join_hints,
            intent_context=intent_context,
            table_whitelist_text=table_whitelist_text,
        )

        return self.bedrock.invoke(
            system_prompt=system_prompt,
            user_message=user_message,
            context="sql_repair",
        )

    # ──────────────────────────────────────────────────────────────────
    # SQL sanitization
    # ──────────────────────────────────────────────────────────────────
    def _extract_sql_from_text(self, text: str) -> str:
        if not text:
            return ""
        m = self._FENCE_RE.search(text)
        sql = m.group(1) if m else text
        sql = sql.strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.I)
        sql = re.sub(r"\s*```$", "", sql, flags=re.I)
        m2 = self._FIRST_SELECT_RE.search(sql)
        if m2:
            sql = sql[m2.start():].strip()
        return sql

    def _ensure_select_only(self, sql: str) -> str:
        if not sql:
            return ""
        s = sql.strip().rstrip(";")
        parts = [
            p.strip()
            for p in re.split(r";\s*(?=WITH\b|SELECT\b|$)", s, flags=re.I)
            if p.strip()
        ]
        first = next(
            (p for p in parts if re.match(r"(?is)^(with\b|select\b)", p)), ""
        )
        if not first:
            return ""
        if self._PROHIBITED_RE.search(first):
            return ""
        return first

    def _finalize_sql(self, text: str) -> str:
        sql = self._extract_sql_from_text(text)
        sql = self._ensure_select_only(sql)
        return sql.strip()

    def _tables_respect_whitelist(self, sql: str, whitelist: List[str]) -> bool:
        """
        Check if SQL only uses tables from the whitelist.
        
        Handles various table name formats:
        - dbo.TableName
        - [dbo].[TableName]
        - [Database].[dbo].[TableName]
        - Database.dbo.TableName
        
        Also handles:
        - CTEs (WITH ... AS) - extracts CTE names and allows them
        - Table aliases (FROM table AS alias)
        """
        if not whitelist:
            return True
        
        if not sql:
            return True
        
        sql_upper = sql.upper()
        
        # Extract CTE names (WITH cte_name AS ...)
        cte_pattern = r'(?i)\bWITH\s+(\w+)\s+AS\s*\('
        cte_names = set(m.lower() for m in re.findall(cte_pattern, sql))
        
        # Also find recursive CTE patterns: ), cte_name AS (
        recursive_cte_pattern = r'(?i)\)\s*,\s*(\w+)\s+AS\s*\('
        cte_names.update(m.lower() for m in re.findall(recursive_cte_pattern, sql))
        
        logger.debug("CTE names found: %s", cte_names)
        
        # Extract table references from SQL
        # Matches: FROM/JOIN followed by table name (handles schema.table and [schema].[table])
        # Updated to capture full qualified names including brackets
        pattern = r'(?i)(?:FROM|JOIN)\s+(\[?[\w]+\]?(?:\.\[?[\w]+\]?){0,2})'
        matches = re.findall(pattern, sql)
        
        if not matches:
            return True  # No tables found, assume OK
        
        def normalize_table_name(name: str) -> str:
            """Normalize table name for comparison."""
            # Remove brackets
            name = re.sub(r'[\[\]]', '', name)
            # Convert to lowercase
            name = name.lower()
            # Extract just the last 2 parts (schema.table)
            parts = name.split('.')
            if len(parts) >= 2:
                return f"{parts[-2]}.{parts[-1]}"
            return parts[-1]
        
        # Normalize whitelist
        whitelist_normalized = set()
        for w in whitelist:
            norm = normalize_table_name(w)
            whitelist_normalized.add(norm)
            # Also add just the table name (without schema)
            parts = norm.split('.')
            if len(parts) >= 2:
                whitelist_normalized.add(parts[-1])
        
        # Check each table in SQL
        for table_ref in matches:
            table_ref_clean = table_ref.strip()
            table_norm = normalize_table_name(table_ref_clean)
            table_name_only = table_norm.split('.')[-1] if '.' in table_norm else table_norm
            
            # Skip if it's a CTE name
            if table_name_only in cte_names:
                logger.debug("WHITELIST_CHECK: skipping CTE %s", table_ref_clean)
                continue
            
            # Check if table matches whitelist
            if table_norm not in whitelist_normalized and table_name_only not in whitelist_normalized:
                logger.warning(
                    "WHITELIST_CHECK_FAIL: table=%s normalized=%s not in whitelist=%s",
                    table_ref_clean, table_norm, whitelist_normalized
                )
                return False
        
        return True

    # ──────────────────────────────────────────────────────────────────
    # Intent debug string for prompts
    # ──────────────────────────────────────────────────────────────────
    def _intent_debug_string(self, intent_context: Dict[str, Any]) -> str:
        if not intent_context:
            return "(no intent context provided)"

        tpl = intent_context.get("template_ref")
        slots = intent_context.get("slots", {})
        tables = intent_context.get("tables", [])
        title = intent_context.get("title") or intent_context.get("display_name")
        score = intent_context.get("score")
        cands = intent_context.get("candidates", [])
        recipe_id = intent_context.get("recipe_id")
        business_prompt = intent_context.get("business_prompt", "")

        lines = [f"template_ref={tpl}", f"slots={json.dumps(slots, ensure_ascii=False)}"]
        if recipe_id:
            lines.append(f"recipe_id={recipe_id}")
        if tables:
            lines.append(f"tables_hint={','.join(tables)}")
        if title:
            lines.append(f"title={title}")
        if score is not None:
            lines.append(f"score={score}")
        if cands:
            lines.append(f"top_candidates={json.dumps(cands[:3], ensure_ascii=False)}")
        if business_prompt:
            bp = business_prompt
            if len(bp) > 400:
                bp = bp[:400] + " …(truncated)"
            lines.append(f"business_prompt={bp}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # Aggregates + explanation
    # ──────────────────────────────────────────────────────────────────
    def _compute_basic_aggregates(
        self,
        rows: List[Tuple[Any, ...]],
        columns: List[str],
    ) -> Dict[str, Any]:
        """
        Very simple aggregates that explanation can use.
        """
        col_index = {name: idx for idx, name in enumerate(columns or [])}
        row_count = len(rows)

        emp_cols = [c for c in columns if c.upper() in ("EMPLOYEEID", "員編")]
        type_cols = [
            c
            for c in columns
            if c.upper() in ("CLASSNAME", "假別名稱", "ATTENDANCETYPE")
        ]
        hours_cols = [c for c in columns if c.upper() in ("HOURS", "請假時數", "總時數")]

        unique_people: Optional[int] = None
        if emp_cols:
            idx = col_index[emp_cols[0]]
            unique_people = len({r[idx] for r in rows})

        by_leave_type: Dict[str, int] = {}
        if type_cols:
            idx = col_index[type_cols[0]]
            counter = Counter(str(r[idx]) for r in rows)
            by_leave_type = dict(counter)

        total_hours: Optional[float] = None
        if hours_cols:
            idx = col_index[hours_cols[0]]
            s = 0.0
            for r in rows:
                try:
                    v = r[idx]
                    if v is not None:
                        s += float(v)
                except Exception:
                    continue
            total_hours = s

        return {
            "row_count": row_count,
            "unique_people": unique_people,
            "by_leave_type": by_leave_type,
            "total_hours": total_hours,
        }

    def _format_sample_rows(
        self,
        rows: List[Tuple[Any, ...]],
        columns: List[str],
        *,
        max_rows: int = 5,
    ) -> str:
        if not rows or not columns:
            return "(no sample)"

        header = " | ".join(columns)
        lines = [header, "-" * len(header)]
        for r in rows[:max_rows]:
            line = " | ".join("" if v is None else str(v) for v in r)
            lines.append(line)
        if len(rows) > max_rows:
            lines.append(f"... ({len(rows) - max_rows} more rows truncated)")
        return "\n".join(lines)

    def _generate_explanation(
        self,
        question: str,
        row_count: int,
        columns: List[str],
        aggregates: Dict[str, Any],
        sample_text: str,
    ) -> str:
        """
        zh-tw explanation generator using Bedrock.
        """
        # Fast path: no data
        if row_count <= 0:
            return (
                "### 摘要\n"
                "• 查詢結果為 0 筆，沒有可供分析的資料。\n\n"
                "### 資料品質說明\n"
                "• 請確認日期區間、請假條件或使用者權限是否正確。"
            )

        if not self._is_llm_available():
            # Simple fallback if LLM not available
            rc = aggregates.get("row_count", row_count)
            up = aggregates.get("unique_people")
            th = aggregates.get("total_hours")
            parts = [f"共 {rc} 筆記錄。"]
            if up is not None:
                parts.append(f"{up} 位不重複人員。")
            if th is not None:
                parts.append(f"總請假時數約為 {th}。")
            return " ".join(parts)

        system_prompt, user_message = self._build_explanation_prompt(
            question=question,
            row_count=row_count,
            columns=columns,
            aggregates=aggregates,
            sample_text=sample_text,
        )

        text = self.bedrock.invoke(
            system_prompt=system_prompt,
            user_message=user_message,
            context="explain_zh-tw",
        )
        
        return text.strip() if text else ""