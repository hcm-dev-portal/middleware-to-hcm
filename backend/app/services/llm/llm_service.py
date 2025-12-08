# backend/app/services/llm/llm_service.py
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
    Try to load .env from a few likely locations:

    1. Project root (…/hcm-ai-portal-v3/.env), derived from this file path.
    2. backend/.env (if you ever put one there).
    3. Default python-dotenv search (current working directory upwards).

    Logs exactly which path (if any) was used.
    """
    here = Path(__file__).resolve()
    candidates: List[Path] = []

    try:
        # hcm-ai-portal-v3
        project_root = here.parents[4]
        candidates.append(project_root / ".env")
    except Exception:
        pass

    # backend/.env (optional)
    try:
        backend_root = here.parents[3]  # …/backend
        candidates.append(backend_root / ".env")
    except Exception:
        pass

    # Try explicit paths first
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

    # Fallback: default behaviour (search from CWD upwards)
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

# We keep the Literal type for compatibility with other modules,
# but the pipeline behaviour is zh-tw–only.
Language = Literal["zh-tw", "en"]

LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

if not LLM_API_KEY:
    logger.warning(
        "[LLMService] OPENAI_API_KEY is not set (LLM calls will be disabled)."
    )
else:
    logger.info(
        "[LLMService] OPENAI_API_KEY is set (length=%d, value masked).",
        len(LLM_API_KEY),
    )

logger.info("[LLMService] OPENAI_MODEL=%s", LLM_MODEL or "(not set)")


# ──────────────────────────────────────────────────────────────────────
# Optional LangChain / OpenAI imports (soft dependency)
# ──────────────────────────────────────────────────────────────────────
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain_core.messages import BaseMessage, HumanMessage
except ImportError:
    ChatOpenAI = None
    ChatPromptTemplate = None
    SystemMessagePromptTemplate = None
    HumanMessagePromptTemplate = None
    BaseMessage = object  # type: ignore
    HumanMessage = object  # type: ignore

    logger.warning(
        "[LLMService] langchain_openai is not installed; LLM integration will be disabled."
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
    # Minimal generic fallbacks so this module can import even before db_service is ready
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

    NOTE: The *behaviour* of LLMService is zh-tw–only; this is used
    purely for logging/metadata so you can observe how people type
    (Chinese vs English), but prompts/explanations are always zh-tw.
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
# Core service – zh-tw monolingual behaviour
# ──────────────────────────────────────────────────────────────────────
class LLMService:
    """
    Core LLM orchestration service for the Leave AI Assistant.

    Responsibilities:
      - 接收自然語言問題（主要為繁體中文 zh-tw）。
      - 使用上游檢索層提供的 `intent_context`：
          * template_ref: str | None
          * slots: dict
          * tables: list[str]
          * few_shot_sql: str (recipes 中的範例 SQL)
      - 產生安全的 **SELECT-only** T-SQL。
      - 於資料庫錯誤時用 LLM 做查詢修復。
      - 呼叫 `db_service.run_select(...)` 執行最終 SQL。
      - 輸出給主管看的繁體中文說明（explanation_zh）。

    注意：本版本行為為「繁體中文單軌」，不再提供英文提示詞。
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
        model_name: str = LLM_MODEL,
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = LLM_API_KEY

        # "Global" feature flag based on env + imports
        self.llm_enabled = bool(self.api_key) and ChatOpenAI is not None
        self.llm: Optional[ChatOpenAI] = None  # type: ignore

        # zh-tw–only prompts
        self.sql_prompt_zh = None
        self.repair_sql_prompt_zh = None
        self.explanation_prompt_zh = None

        logger.info(
            "[LLMService.__init__] model_name=%s, temp=%.2f, llm_enabled=%s",
            self.model_name,
            self.temperature,
            self.llm_enabled,
        )

        self._initialize_llm()
        self._initialize_prompts()

    # ──────────────────────────────────────────────────────────────────
    # LLM init + availability
    # ──────────────────────────────────────────────────────────────────
    def _is_llm_available(self) -> bool:
        if not self.api_key:
            logger.error(
                "[LLMService] LLM unavailable: OPENAI_API_KEY is missing or empty."
            )
            return False
        if ChatOpenAI is None:
            logger.error(
                "[LLMService] LLM unavailable: langchain_openai is not installed."
            )
            return False
        if not self.llm_enabled:
            logger.error(
                "[LLMService] LLM unavailable: llm_enabled flag is False (init failed or disabled)."
            )
            return False
        if self.llm is None:
            logger.error(
                "[LLMService] LLM unavailable: ChatOpenAI client was not initialised."
            )
            return False
        return True

    def _initialize_llm(self) -> None:
        # More granular logging on why it's disabled
        if not self.api_key:
            logger.warning(
                "[LLMService] Skipping LLM init: OPENAI_API_KEY not set in env."
            )
            self.llm_enabled = False
            return
        if ChatOpenAI is None:
            logger.warning(
                "[LLMService] Skipping LLM init: langchain_openai is missing."
            )
            self.llm_enabled = False
            return

        t0 = time.perf_counter()
        try:
            try:
                # Newer langchain_openai signature
                self.llm = ChatOpenAI(  # type: ignore
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=self.api_key,
                )
            except TypeError:
                # Older signature fallback
                self.llm = ChatOpenAI(  # type: ignore
                    model_name=self.model_name,  # type: ignore
                    temperature=self.temperature,
                    openai_api_key=self.api_key,
                )
            self.llm_enabled = True
            logger.info(
                "[LLMService] LLM initialized: model=%s temp=%.2f in %.0fms",
                self.model_name,
                self.temperature,
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            logger.error(
                "[LLMService] LLM init failed: %s: %s", type(e).__name__, e, exc_info=True
            )
            self.llm = None
            self.llm_enabled = False

    # ──────────────────────────────────────────────────────────────────
    # Prompt initialization (zh-tw only)
    # ──────────────────────────────────────────────────────────────────
    def _initialize_prompts(self) -> None:
        if not ChatPromptTemplate:
            logger.warning(
                "[LLMService] Prompts disabled: ChatPromptTemplate not available."
            )
            return

        # --- SQL generation prompt (zh-tw only) ---
        # FIXED: Removed misleading @today instruction; now instructs to use inline T-SQL functions
        self.sql_prompt_zh = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(  # type: ignore
                    "你是一位專精於人資請假與考勤資料的 T-SQL 專家，負責產生安全的查詢語句。\n"
                    "請只回傳 **一個** Microsoft SQL Server (T-SQL) 查詢，且必須是 **僅限 SELECT**，可以使用 CTE。\n\n"
                    "（下列內容由上游檢索系統提供，已根據公司實際資料庫 schema 與 recipe 精心設計）\n\n"
                    "意圖 (intent)：\n{intent_debug}\n\n"
                    "Few-shot 參考 SQL（若有，優先參考其欄位與 JOIN 寫法）：\n{few_shot}\n\n"
                    "【重要】日期處理規則：\n"
                    "- 嚴禁使用 SQL 參數變數（如 @today、@startDate 等），因為本系統不支援參數綁定。\n"
                    "- 若需要「今天」的日期，請使用 CAST(GETDATE() AS DATE)。\n"
                    "- 若需要「本週」，請使用 DATEADD(DAY, -DATEPART(WEEKDAY, GETDATE())+1, CAST(GETDATE() AS DATE)) 作為週一。\n"
                    "- 若需要「本月」，請使用 DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)。\n"
                    "- 建議使用 CAST(column AS DATE) 搭配 BETWEEN 或 >= / < 做日期過濾。\n\n"
                    "系統提供的日期資訊（可直接參考）：\n"
                    "- 今天日期: {today_date}\n"
                    "- 今天星期: {today_weekday}\n\n"
                    "業務規則（Leave AI）：\n"
                    "- WORKDATE 為發生日；STARTDATE/ENDDATE 為請假區間。\n"
                    "- 統計「已批准」請假時，請加上 VALIDATED = 1 條件（若適用）。\n"
                    "- 只有在需要顯示部門/單位資訊時再 JOIN 組織表或人員表。\n\n"
                    "T-SQL 安全規範：\n"
                    "- 嚴禁使用 INSERT/UPDATE/DELETE/MERGE/ALTER/DROP/CREATE/TRUNCATE/EXEC 等指令。\n"
                    "- 只能產生一個查詢語句，不得包含多個批次或 GO。\n"
                    "- 別名必須先在 FROM/JOIN 宣告後再使用。\n"
                    "- GROUP BY 必須包含所有非聚合欄位。\n"
                    "- 不可使用 @ 開頭的變數。\n\n"
                    "可用資料庫結構 (schema 摘要)：\n{schema}\n\n"
                    "建議 JOIN 關聯說明：\n{join_hints}\n\n"
                    "若有提供 table_whitelist，請只使用其中出現的資料表：\n{table_whitelist}\n"
                ),
                HumanMessagePromptTemplate.from_template(  # type: ignore
                    "使用者問題：{query}\n\n"
                    "已抽取的 slots (JSON)：{slots_json}\n\n"
                    "請只回傳最終 SQL 查詢本體（不要加 markdown、不要加額外說明或註解、不要使用 @ 變數）。"
                ),
            ]
        )

        # --- SQL repair prompt (zh-tw only) ---
        # FIXED: Added explicit instruction about @variable errors
        self.repair_sql_prompt_zh = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(  # type: ignore
                    "你要協助修復一段失敗的 Microsoft SQL Server (T-SQL) 查詢。\n"
                    "請輸出一個修正後的 **僅限 SELECT** 的查詢，維持原本意圖，不得新增 DML/DDL 指令。\n"
                    "務必遵守：\n"
                    "- 別名先在 FROM/JOIN 宣告再使用。\n"
                    "- GROUP BY 包含所有非聚合欄位。\n"
                    "- 【重要】不可使用任何 @ 開頭的變數（如 @today、@startDate），請改用 T-SQL 內建日期函數。\n"
                    "- 若錯誤訊息提到「必須宣告純量變數」，表示原 SQL 使用了未定義的 @ 變數，請將其替換為對應的 T-SQL 函數。\n\n"
                    "日期替換指引：\n"
                    "- @today → CAST(GETDATE() AS DATE)\n"
                    "- @now → GETDATE()\n"
                    "- @startOfWeek → DATEADD(DAY, -DATEPART(WEEKDAY, GETDATE())+1, CAST(GETDATE() AS DATE))\n"
                    "- @startOfMonth → DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)\n"
                    "- @startOfYear → DATEFROMPARTS(YEAR(GETDATE()), 1, 1)\n\n"
                    "系統提供的日期資訊：\n"
                    "- 今天日期: {today_date}\n\n"
                    "意圖 (intent)：\n{intent_debug}\n\n"
                    "Few-shot 參考 SQL：\n{few_shot}\n\n"
                    "可用 schema：\n{schema}\n\n"
                    "建議 JOIN 關係：\n{join_hints}\n\n"
                    "允許使用之資料表（若有提供）：\n{table_whitelist}\n"
                ),
                HumanMessagePromptTemplate.from_template(  # type: ignore
                    "資料庫錯誤訊息：\n{error_summary}\n\n"
                    "原始失敗的 SQL：\n{failed_sql}\n\n"
                    "請只回傳修正後的 SQL，本體即可（不要使用 @ 變數）。"
                ),
            ]
        )

        # --- Explanation prompt (zh-tw only) ---
        self.explanation_prompt_zh = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(  # type: ignore
                    "你是一位服務公司高階主管的人資資料分析師。\n"
                    "請根據提供的欄位、統計摘要與樣本資料，用繁體中文寫出簡潔的說明。\n"
                    "嚴格規則：\n"
                    "- 僅可使用提供的欄位名稱、聚合統計與樣本資料，不可自行杜撰欄位或數值。\n"
                    "- 若資料不足以回答問題，請在摘要中明確說明。\n"
                    "- 不要輸出 SQL 或程式碼。\n"
                    "輸出格式（Markdown）：\n"
                    "### 摘要\n"
                    "• 2–3 點最重要的數字或結論（需與問題直接相關）。\n"
                    "### 主要觀察\n"
                    "• 2–4 點描述分布、趨勢、異常值或部門/假別等類別的重點。\n"
                    "### 風險與建議\n"
                    "• 1–3 點給主管的具體建議（例如追蹤對象、檢查政策、設定門檻）。\n"
                    "### 資料品質說明\n"
                    "• 1–2 點說明樣本限制（例如資料期間、欄位缺漏、筆數過少）。\n"
                ),
                HumanMessagePromptTemplate.from_template(  # type: ignore
                    "問題：{question}\n"
                    "資料筆數：{row_count}\n"
                    "欄位：{columns}\n"
                    "統計摘要 (JSON)：{aggregates_json}\n"
                    "資料樣本（截斷顯示）：\n{sample_text}\n"
                ),
            ]
        )

        logger.info("[LLMService] Prompts initialized (zh-tw only).")

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
            "today_date": today.isoformat(),  # e.g., "2025-12-08"
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
        
        # Define substitutions (case-insensitive matching)
        substitutions = [
            # @today variants
            (r"@today\b", f"CAST('{today_str}' AS DATE)"),
            (r"@currentDate\b", f"CAST('{today_str}' AS DATE)"),
            (r"@now\b", "GETDATE()"),
            # Week boundaries
            (r"@startOfWeek\b", "DATEADD(DAY, -DATEPART(WEEKDAY, GETDATE())+1, CAST(GETDATE() AS DATE))"),
            (r"@endOfWeek\b", "DATEADD(DAY, 7-DATEPART(WEEKDAY, GETDATE()), CAST(GETDATE() AS DATE))"),
            # Month boundaries  
            (r"@startOfMonth\b", "DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)"),
            (r"@endOfMonth\b", "EOMONTH(GETDATE())"),
            # Year boundaries
            (r"@startOfYear\b", "DATEFROMPARTS(YEAR(GETDATE()), 1, 1)"),
            (r"@endOfYear\b", "DATEFROMPARTS(YEAR(GETDATE()), 12, 31)"),
        ]

        result = sql
        for pattern, replacement in substitutions:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Log if we made substitutions
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
            # Filter out false positives (e.g., @@ROWCOUNT, @@IDENTITY are valid)
            unbound = [m for m in matches if not m.startswith('@')]
            if unbound:
                return f"SQL contains unbound parameters: @{', @'.join(unbound)}"
        return None

    # ──────────────────────────────────────────────────────────────────
    # New: unified SQL + DB execution with repair
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
        Used by VectorSearchService.run_with_openai.

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

        # Table whitelist: primary is explicit field; fallback to tables hint.
        table_whitelist: List[str] = list(
            ctx.get("table_whitelist") or ctx.get("tables") or []
        )
        whitelist_text = ", ".join(table_whitelist) if table_whitelist else "(no restriction)"

        attempts = 0
        sql = ""
        rows: List[Tuple[Any, ...]] = []
        cols: List[str] = []
        last_error_summary = "initial generation (no DB error yet)"

        # 行為上：一律使用 zh-tw 提示詞（但仍保留 detected_lang 供觀察）
        effective_lang: Language = "zh-tw"

        while attempts < max_attempts:
            attempts += 1

            if attempts == 1:
                raw = self._generate_sql_raw(
                    question=question,
                    schema=schema,
                    join_hints=join_hints,
                    intent_context=ctx,
                    table_whitelist_text=whitelist_text,
                    language=effective_lang,
                )
            else:
                raw = self._repair_sql_raw(
                    failed_sql=sql,
                    error_summary=last_error_summary,
                    schema=schema,
                    join_hints=join_hints,
                    intent_context=ctx,
                    table_whitelist_text=whitelist_text,
                    language=effective_lang,
                )

            sql = self._finalize_sql(raw)

            if not sql:
                logger.warning("RUN_QUERY_ATTEMPT_EMPTY_SQL: attempt=%d", attempts)
                continue

            # Safety guard: prohibited keywords
            if self._PROHIBITED_RE.search(sql):
                logger.warning(
                    "RUN_QUERY_PROHIBITED_KEYWORD: attempt=%d sql_prefix=%r",
                    attempts,
                    sql[:200],
                )
                sql = ""
                continue

            # FIXED: Substitute any remaining @parameters before execution
            sql = self._substitute_date_parameters(sql)

            # Check for any remaining unbound parameters
            param_error = self._check_for_unbound_parameters(sql)
            if param_error:
                logger.warning(
                    "RUN_QUERY_UNBOUND_PARAMS: attempt=%d error=%s sql_prefix=%r",
                    attempts,
                    param_error,
                    sql[:200],
                )
                last_error_summary = param_error
                # Don't clear sql - let the repair prompt try to fix it
                continue

            # String-based whitelist check
            if table_whitelist and not self._tables_respect_whitelist(sql, table_whitelist):
                logger.warning(
                    "RUN_QUERY_WHITELIST_VIOLATION: attempt=%d sql_prefix=%r",
                    attempts,
                    sql[:200],
                )
                sql = ""
                last_error_summary = "Table whitelist violation in generated SQL"
                continue

            # Try executing against DB
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
                # Specific handling for syntax errors (like undeclared variables)
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
        
        # Special handling for @variable errors (common issue)
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
        allow_fallback: bool = False,  # kept for compatibility with caller
    ) -> Dict[str, Any]:
        """
        Full pipeline for HTTP controllers:
          - 使用 run_query_with_llm_repair 產生 SQL + 執行 DB。
          - 計算基本統計。
          - 產出 zh-tw 說明給主管閱讀。

        STRICT BEHAVIOUR:
          - 若 LLM 無法使用 → error, success=False。
          - 若 max_attempts 內仍無有效 SQL → error, success=False。
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

        if allow_fallback:
            logger.debug(
                "[LLMService.answer_question] allow_fallback=True (currently not used)."
            )

        # 0) LLM availability check
        if not self._is_llm_available():
            msg = "LLM backend not available（可能是 API key 或 langchain_openai 未正確設定）。"
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
                    "• LLM API 設定或連線可能有問題。"
                ),
                "success": False,
                "error": msg,
                "error_category": "llm_unavailable",
            }

        # 1) Query + repair pipeline (shared with VectorSearchService)
        ctx = intent_context or {}
        # If controller provides explicit whitelist, respect it (higher priority)
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

        # 2) Aggregates & explanation
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
    # SQL generation + repair (LLM side only, zh-tw behaviour)
    # ──────────────────────────────────────────────────────────────────
    def _generate_sql_raw(
        self,
        question: str,
        schema: str,
        join_hints: str,
        intent_context: Dict[str, Any],
        table_whitelist_text: str,
        language: Language,
    ) -> str:
        """
        language 參數僅保留給 logging/上下游相容；
        實際上永遠使用 zh-tw 提示詞。
        """
        if not (self.llm_enabled and self.llm and ChatPromptTemplate):
            logger.warning("SQL_GEN_RAW: LLM not available.")
            return ""

        prompt = self.sql_prompt_zh
        if not prompt:
            logger.warning("SQL_GEN_RAW: zh-tw prompt missing")
            return ""

        slots = intent_context.get("slots", {}) or {}
        # Support multiple possible keys from recipes/router for examples
        few_shot_sql = (
            intent_context.get("few_shot_sql")
            or intent_context.get("example_sql")
            or ""
        )
        intent_debug = self._intent_debug_string(intent_context)
        
        # Get today's date info for the prompt
        today_info = self._get_today_info()

        try:
            msgs: List[BaseMessage] = prompt.format_messages(  # type: ignore
                query=question,
                schema=schema,
                join_hints=join_hints,
                intent_debug=intent_debug,
                few_shot=few_shot_sql,
                slots_json=json.dumps(slots, ensure_ascii=False),
                table_whitelist=table_whitelist_text,
                today_date=today_info["today_date"],
                today_weekday=today_info["today_weekday"],
            )
            # context 仍帶 language 以方便追蹤，但內容為 zh-tw prompt
            return self._invoke_llm(msgs, context=f"sql_gen_{language}")
        except Exception as e:
            logger.error("SQL_GEN_RAW_FAIL: %s: %s", type(e).__name__, e)
            return ""

    def _repair_sql_raw(
        self,
        failed_sql: str,
        error_summary: str,
        schema: str,
        join_hints: str,
        intent_context: Dict[str, Any],
        table_whitelist_text: str,
        language: Language,
    ) -> str:
        """
        language 參數僅保留給 logging/上下游相容；
        實際上永遠使用 zh-tw 修復提示詞。
        """
        if not (self.llm_enabled and self.llm and ChatPromptTemplate):
            logger.warning("SQL_REPAIR_RAW: LLM not available.")
            return failed_sql

        prompt = self.repair_sql_prompt_zh
        if not prompt:
            logger.warning("SQL_REPAIR_RAW: zh-tw repair prompt missing")
            return failed_sql

        slots = intent_context.get("slots", {}) or {}
        few_shot_sql = (
            intent_context.get("few_shot_sql")
            or intent_context.get("example_sql")
            or ""
        )
        intent_debug = self._intent_debug_string(intent_context)
        
        # Get today's date info for the repair prompt
        today_info = self._get_today_info()

        try:
            msgs: List[BaseMessage] = prompt.format_messages(  # type: ignore
                failed_sql=failed_sql,
                error_summary=error_summary,
                schema=schema,
                join_hints=join_hints,
                intent_debug=intent_debug,
                few_shot=few_shot_sql,
                slots_json=json.dumps(slots, ensure_ascii=False),
                table_whitelist=table_whitelist_text,
                today_date=today_info["today_date"],
            )
            return self._invoke_llm(msgs, context=f"sql_repair_{language}")
        except Exception as e:
            logger.error("SQL_REPAIR_RAW_FAIL: %s: %s", type(e).__name__, e)
            return failed_sql

    # ──────────────────────────────────────────────────────────────────
    # LLM invoke + SQL sanitization
    # ──────────────────────────────────────────────────────────────────
    def _invoke_llm(self, messages: List[BaseMessage], context: str = "") -> str:  # pyright: ignore[reportInvalidTypeForm]
        if not (self.llm_enabled and self.llm):
            logger.warning(
                "LLM_INVOKE_SKIPPED: llm_enabled=%s, llm=%s",
                self.llm_enabled,
                type(self.llm).__name__ if self.llm else None,
            )
            return ""
        t0 = time.perf_counter()
        try:
            user_preview = ""
            for m in reversed(messages):
                if isinstance(m, HumanMessage):  # type: ignore
                    user_preview = str(m.content)[:120]
                    break
            logger.debug("LLM_INVOKE: ctx=%s user=%r", context, user_preview)
            resp = self.llm.invoke(messages)  # type: ignore
            content = str(getattr(resp, "content", "") or "")
            logger.info(
                "LLM_INVOKE_OK: ctx=%s ms=%d len=%d",
                context,
                int((time.perf_counter() - t0) * 1000),
                len(content),
            )
            return content
        except Exception as e:
            logger.error(
                "LLM_INVOKE_FAIL: ctx=%s %s: %s", context, type(e).__name__, e
            )
            return ""

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
        # Split on semicolon and keep first SELECT/CTE
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
        Very simple table whitelist enforcement by regex scanning for FROM/JOIN.
        """
        if not whitelist:
            return True
        toks = re.findall(
            r"(?i)\bfrom\s+([^\s\(\),]+)|\bjoin\s+([^\s\(\),]+)", sql or ""
        )
        raw = [x or y for (x, y) in toks]

        def norm(t: str) -> str:
            # Normalize: strip [ ], ``, quotes and lowercase
            t = t.strip().rstrip(",")
            t = re.sub(r"^[\[\]`\"']+|[\[\]`\"']+$", "", t)
            return t.lower()

        whitelist_norm = {norm(w) for w in whitelist}
        for t in raw:
            if whitelist_norm and norm(t) not in whitelist_norm:
                return False
        return True

    # ──────────────────────────────────────────────────────────────────
    # Intent debug string for prompts – includes recipe hints
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
            # Only show top 2–3 to avoid blowing the context
            lines.append(f"top_candidates={json.dumps(cands[:3], ensure_ascii=False)}")
        if business_prompt:
            # Truncate long recipe description
            bp = business_prompt
            if len(bp) > 400:
                bp = bp[:400] + " …(truncated)"
            lines.append(f"business_prompt={bp}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # Aggregates + zh-tw explanation
    # ──────────────────────────────────────────────────────────────────
    def _compute_basic_aggregates(
        self,
        rows: List[Tuple[Any, ...]],
        columns: List[str],
    ) -> Dict[str, Any]:
        """
        Very simple aggregates that explanation can use:
          - row_count
          - unique_people (by EMPLOYEEID/員編)
          - by_leave_type (count per CLASSNAME/假別名稱/ATTENDANCETYPE)
          - total_hours (sum HOURS/請假時數/總時數)
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
        zh-tw 專用說明產生器。
        """
        # Fast path: no data
        if row_count <= 0:
            return (
                "### 摘要\n"
                "• 查詢結果為 0 筆，沒有可供分析的資料。\n\n"
                "### 資料品質說明\n"
                "• 請確認日期區間、請假條件或使用者權限是否正確。"
            )

        if not (self.llm_enabled and self.llm and ChatPromptTemplate):
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

        prompt = self.explanation_prompt_zh
        if not prompt:
            return ""

        cols_joined = ", ".join(columns) if columns else "(none)"
        aggs_json = json.dumps(aggregates or {}, ensure_ascii=False)

        try:
            msgs: List[BaseMessage] = prompt.format_messages(  # type: ignore
                question=question,
                row_count=row_count,
                columns=cols_joined,
                aggregates_json=aggs_json,
                sample_text=sample_text,
            )
            text = self._invoke_llm(msgs, context="explain_zh-tw").strip()
            return text or ""
        except Exception as e:
            logger.error("EXPLANATION_FAIL: %s: %s", type(e).__name__, e)
            return ""