# backend/app/services/llm/openai_service_unified.py
from __future__ import annotations

import re
import os
import logging
import time
import hashlib
from typing import List, Optional, Dict, Any, Tuple, Literal

logger = logging.getLogger(__name__)

# Optional OpenAI / LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.schema import BaseMessage, HumanMessage
    from langchain.memory import ConversationBufferMemory
except ImportError:
    ChatOpenAI = None
    ChatPromptTemplate = None
    SystemMessagePromptTemplate = None
    HumanMessagePromptTemplate = None
    BaseMessage = None
    HumanMessage = None
    ConversationBufferMemory = None

# Typed DB exceptions
from app.services.db_service import (
    DatabaseQueryError as DBServiceQueryError,
    DatabaseSyntaxError as DBServiceSyntaxError,
    TableNotFoundError as DBServiceTableNotFoundError,
    ColumnNotFoundError as DBServiceColumnNotFoundError,
    DatabaseDataError as DBServiceDataError,
    DatabaseIntegrityError as DBServiceIntegrityError,
    DatabaseOperationalError as DBServiceOperationalError,
    DatabaseTimeoutError as DBServiceTimeoutError,
    DatabaseConnectionError as DBServiceConnectionError,
    PermissionDeniedError as DBServicePermissionDeniedError,
    DeadlockError as DBServiceDeadlockError,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ────────────────────────────────────────────────────────────────────────────────
# Language detection (lightweight & robust for mixed zh/en)
# ────────────────────────────────────────────────────────────────────────────────
def detect_query_language(text: str) -> Literal["zh-tw", "en"]:
    if not text or not text.strip():
        return "en"
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin_num = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isdigit()))
    if chinese_chars >= 2 and chinese_chars >= latin_num:
        return "zh-tw"
    # soft keywords tip it to zh
    if any(k in text for k in ["請假", "考勤", "部門", "員工", "今天", "現在", "統計", "趨勢"]):
        return "zh-tw"
    return "en"


class UnifiedBilingualOpenAIService:
    """
    Bilingual T-SQL generation/repair/explanation with strict SELECT-only guardrails.
    """

    def __init__(self, model_name: str = OPENAI_MODEL, temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.llm_enabled = bool(OPENAI_API_KEY) and ChatOpenAI is not None
        self.memory = None

        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "repair_attempts": 0,
            "successful_repairs": 0,
            "total_tokens_used": 0,
            "avg_generation_time": 0.0,
        }

        self.sql_prompt_en = None
        self.sql_prompt_zh = None
        self.repair_sql_prompt_en = None
        self.repair_sql_prompt_zh = None
        self.explanation_prompt_en = None
        self.explanation_prompt_zh = None

        self._initialize_llm()
        self._initialize_all_prompts()

    # ────────────────────────────────────────────────────────────────────────────
    # LLM init
    # ────────────────────────────────────────────────────────────────────────────
    def _initialize_llm(self):
        if not self.llm_enabled:
            logger.warning("LLM DISABLED: No API key or langchain_openai missing.")
            return
        t0 = time.perf_counter()
        try:
            try:
                self.llm = ChatOpenAI(  # type: ignore
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=OPENAI_API_KEY,  # type: ignore
                )
            except TypeError:
                self.llm = ChatOpenAI(  # type: ignore
                    model_name=self.model_name,  # type: ignore
                    temperature=self.temperature,  # type: ignore
                    openai_api_key=OPENAI_API_KEY,  # type: ignore
                )
            self.memory = ConversationBufferMemory(return_messages=True) if ConversationBufferMemory else None
            logger.info("LLM INITIALIZED: model=%s temp=%.2f memory=%s init_ms=%d",
                        self.model_name, self.temperature, bool(self.memory),
                        int((time.perf_counter() - t0) * 1000))
        except Exception as e:
            logger.error("LLM INIT FAILED: %s: %s", type(e).__name__, e, exc_info=True)
            self.llm = None
            self.llm_enabled = False

    # ────────────────────────────────────────────────────────────────────────────
    # Prompts (EN + ZH)
    # ────────────────────────────────────────────────────────────────────────────
    def _initialize_all_prompts(self):
        if not ChatPromptTemplate:
            logger.warning("PROMPTS DISABLED: ChatPromptTemplate not available.")
            return

        self.sql_prompt_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "You are an expert T-SQL analyst for HR leave & attendance data.\n"
                "Return exactly ONE safe **SELECT-only** Microsoft SQL Server (T-SQL) query.\n\n"
                "DATE HANDLING:\n"
                "- For 'today'/'current' use ACTUAL current date; a data anchor is ONLY background.\n"
                "- Date filters use: CAST(column AS date) = 'YYYY-MM-DD' or BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.\n"
                "- Do NOT use GETDATE(); dates are already rewritten upstream.\n\n"
                "BUSINESS RULES:\n"
                "- VALIDATED = 1 when counting approved leave.\n"
                "- WORKDATE is the occurrence date; STARTDATE/ENDDATE is the request range.\n"
                "- Person info via person dimension when needed.\n\n"
                "T-SQL RULES:\n"
                "- Only SELECT (CTEs allowed). No DML/DDL.\n"
                "- No LIMIT; use TOP (N) with ORDER BY if needed.\n"
                "- Paginate with ORDER BY ... OFFSET ... FETCH.\n"
                "- Only use columns/tables present in schema.\n"
                "- GROUP BY must include all non-aggregates.\n\n"
                "Available schema:\n{schema}\n\nJoin hints:\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "User question: {query}\n\nReturn only the SQL query. No markdown, no comments."
            ),
        ])

        self.sql_prompt_zh = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "你是人資請假/考勤資料的T-SQL專家。\n"
                "請只回傳一個安全的 **僅限SELECT** 的 Microsoft SQL Server (T-SQL) 查詢。\n\n"
                "日期處理：\n"
                "- 「今天/目前」使用真實今天；資料錨點只作為歷史背景。\n"
                "- 日期過濾用 CAST(column AS date) = 'YYYY-MM-DD' 或 BETWEEN。\n"
                "- 不要使用 GETDATE()；日期已在上游處理。\n\n"
                "業務規則：\n"
                "- 統計已批准請假用 VALIDATED = 1。\n"
                "- WORKDATE 是發生日；STARTDATE/ENDDATE 是申請範圍。\n"
                "- 需要姓名/員工編號時再關聯人員維度。\n\n"
                "T-SQL 規範：\n"
                "- 只允許 SELECT（可用 CTE）。禁止 DML/DDL。\n"
                "- 不可使用 LIMIT；若需要，使用 TOP (N) 並搭配 ORDER BY。\n"
                "- 分頁使用 ORDER BY ... OFFSET ... FETCH。\n"
                "- 僅使用提供的資料表/欄位；GROUP BY 包含所有非聚合欄位。\n\n"
                "可用架構：\n{schema}\n\n關聯提示：\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "使用者問題：{query}\n\n請只回傳 SQL 查詢語句，無需註解或 markdown。"
            ),
        ])

        self._initialize_repair_prompts()
        self._initialize_explanation_prompts()
        logger.info("PROMPTS INITIALIZED.")

    def _initialize_repair_prompts(self):
        self.repair_sql_prompt_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "You fix failing Microsoft SQL Server (T-SQL) queries.\n"
                "Output exactly one corrected **SELECT-only** T-SQL statement.\n"
                "Keep original intent; use only schema columns; respect GROUP BY rules; no comments.\n\n"
                "Available schema:\n{schema}\n\nJoin hints:\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "Database error:\n{error_summary}\n\nFailed SQL:\n{failed_sql}\n\nReturn only the corrected SQL."
            ),
        ])
        self.repair_sql_prompt_zh = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "你要修復失敗的 Microsoft SQL Server (T-SQL) 查詢。\n"
                "請輸出一個修正後的 **僅限SELECT** 的 T-SQL 語句，維持原意且僅用架構欄位；不得有註解。\n\n"
                "可用架構：\n{schema}\n\n關聯提示：\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "資料庫錯誤：\n{error_summary}\n\n失敗的SQL：\n{failed_sql}\n\n只回傳修正後的SQL。"
            ),
        ])

    def _initialize_explanation_prompts(self):
        self.explanation_prompt_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "You are a data analyst. Write a brief, business-friendly summary (3–6 bullets or 2–4 sentences). "
                "Include totals, breakdowns, notable patterns, and actionable insights. No SQL."
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "Question: {question}\nRow count: {row_count}\nColumns: {columns}\n"
                "Aggregates (JSON): {aggregates_json}\nSample rows (truncated):\n{sample_text}\n\n"
                "Write the summary in English."
            ),
        ])
        self.explanation_prompt_zh = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "你是資料分析師。請以繁體中文寫出簡短、業務易讀的摘要（3–6點或2–4句）。包含總數、分類、重點趨勢、建議。不要有SQL。"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "問題：{question}\n資料筆數：{row_count}\n欄位：{columns}\n"
                "統計摘要 (JSON)：{aggregates_json}\n資料樣本（截斷）：\n{sample_text}\n\n"
                "用繁體中文撰寫摘要。"
            ),
        ])

    # ────────────────────────────────────────────────────────────────────────────
    # Utilities (extraction, sanitation, fixes)
    # ────────────────────────────────────────────────────────────────────────────
    _FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    _FIRST_SELECT_RE = re.compile(r"(?is)\bwith\b[\s\S]+?\bselect\b|\bselect\b")
    _PROHIBITED_RE = re.compile(
        r"(?is)\b(insert|update|delete|merge|drop|alter|create|truncate|exec|execute|grant|revoke)\b"
    )

    def _extract_sql_from_text(self, text: str) -> str:
        if not text:
            return ""
        m = self._FENCE_RE.search(text)
        sql = m.group(1) if m else text
        sql = sql.strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.I)
        sql = re.sub(r"\s*```$", "", sql)
        # If there’s leading prose, cut to the first WITH/SELECT
        m2 = self._FIRST_SELECT_RE.search(sql)
        if m2:
            sql = sql[m2.start():].strip()
        return sql

    def _normalize_id_quotes(self, sql: str) -> str:
        # Convert MySQL backticks and double-quoted identifiers to bare or [brackets] only where necessary.
        s = re.sub(r"`([^`]+)`", r"[\1]", sql)  # backticks → brackets
        # Leave double quotes alone unless clearly used as identifier wrappers:
        s = re.sub(r'(?<!")"([A-Za-z_][\w]*)"(?!")', r"[\1]", s)
        return s

    def _tsql_limit_fix(self, sql: str) -> str:
        if not sql:
            return sql
        s = sql.strip().rstrip(";")
        m = re.search(r"\blimit\s+(\d+)\s*$", s, flags=re.I)
        if m and re.search(r"^\s*select\b", s, flags=re.I):
            n = m.group(1)
            s = re.sub(r"\blimit\s+\d+\s*$", "", s, flags=re.I).strip()
            s = re.sub(r"(?i)^\s*select", f"SELECT TOP ({n})", s, count=1)
            logger.debug("TSQL_FIX: LIMIT→TOP(%s)", n)
        return s

    def _ensure_select_only(self, sql: str) -> str:
        """Harden to a single SELECT/CTE; strip extra statements and block DDL/DML."""
        if not sql:
            return ""
        s = sql.strip()
        # Kill prohibited keywords
        if self._PROHIBITED_RE.search(s):
            logger.warning("SANITIZE: prohibited keyword detected; returning safe empty SELECT.")
            return "SELECT 1 WHERE 1=0"
        # Split on statement terminators; keep first statement that starts with WITH/SELECT
        parts = [p.strip() for p in re.split(r";\s*(?=WITH\b|SELECT\b|$)", s, flags=re.I)]
        first_valid = next((p for p in parts if re.match(r"(?is)^(with\b|select\b)", p)), "")
        if not first_valid:
            return ""
        return first_valid

    def _finalize_sql(self, sql: str) -> str:
        """Apply all post-processing: fences→sql, normalize, LIMIT fix, SELECT-only guard."""
        s = self._extract_sql_from_text(sql)
        s = self._normalize_id_quotes(s)
        s = self._tsql_limit_fix(s)
        s = self._ensure_select_only(s)
        return s.strip()

    def _create_query_signature(self, query: str, language: str) -> str:
        normalized = re.sub(r"\s+", " ", (query or "").lower().strip())
        return hashlib.md5(f"{language}:{normalized}".encode()).hexdigest()[:12]

    def _invoke_llm(self, messages: List[BaseMessage], context: str = "") -> str:  # type: ignore
        if not (self.llm_enabled and self.llm and messages):
            logger.warning("LLM_INVOKE: unavailable (enabled=%s, llm=%s, msgs=%s)",
                           self.llm_enabled, bool(self.llm), bool(messages))
            return ""
        t0 = time.perf_counter()
        self.generation_stats["total_requests"] += 1
        try:
            user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)  # type: ignore
            prev = (user_msg.content[:120] + "...") if user_msg and len(user_msg.content) > 120 else (user_msg.content if user_msg else "")
            logger.debug("LLM_INVOKE_START: ctx=%s user_preview=%r", context, prev)
            resp = self.llm.invoke(messages)
            content = str(resp.content)
            if self.memory and user_msg:
                self.memory.save_context({"input": user_msg.content}, {"output": content})
            dt = time.perf_counter() - t0
            self.generation_stats["successful_generations"] += 1
            n = self.generation_stats["successful_generations"]
            self.generation_stats["avg_generation_time"] = (
                (self.generation_stats["avg_generation_time"] * (n - 1)) + dt
            ) / n
            logger.info("LLM_OK: ctx=%s time=%.2fs len=%d avg=%.2fs",
                        context, dt, len(content), self.generation_stats["avg_generation_time"])
            return content
        except Exception as e:
            dt = time.perf_counter() - t0
            self.generation_stats["failed_generations"] += 1
            logger.error("LLM_FAIL: ctx=%s time=%.2fs %s: %s", context, dt, type(e).__name__, e, exc_info=True)
            return ""

    # ────────────────────────────────────────────────────────────────────────────
    # Error classification
    # ────────────────────────────────────────────────────────────────────────────
    def _is_repairable_error(self, e: DBServiceQueryError) -> bool:
        repairable = (
            DBServiceSyntaxError,
            DBServiceTableNotFoundError,
            DBServiceColumnNotFoundError,
            DBServiceDataError,
            DBServiceIntegrityError,
            DBServiceOperationalError,
            DBServiceQueryError,
        )
        non_repairable = (
            DBServiceTimeoutError,
            DBServiceConnectionError,
            DBServicePermissionDeniedError,
            # Deadlock intentionally excluded; app may retry outside.
        )
        ok = isinstance(e, repairable) and not isinstance(e, non_repairable)
        logger.debug("ERROR_CLASS: %s repairable=%s", type(e).__name__, ok)
        return ok

    # ────────────────────────────────────────────────────────────────────────────
    # SQL generation & repair
    # ────────────────────────────────────────────────────────────────────────────
    def generate_sql(self, query: str, schema: str, join_hints: str,
                     language: Optional[Literal["zh-tw", "en"]] = None) -> str:
        t0 = time.perf_counter()
        if not self.llm_enabled:
            logger.warning("SQL_GEN: LLM disabled → fallback stub.")
            return "SELECT 1 WHERE 1=0"

        language = language or detect_query_language(query)
        sig = self._create_query_signature(query, language)
        logger.info("SQL_GEN_START: sig=%s lang=%s q='%s'", sig, language, query[:120])

        try:
            prompt = self.sql_prompt_zh if language == "zh-tw" else self.sql_prompt_en
            if not prompt:
                logger.error("SQL_GEN: prompt missing for lang=%s", language)
                return "SELECT 1 WHERE 1=0"

            messages = prompt.format_messages(query=query, schema=schema, join_hints=join_hints)
            raw = self._invoke_llm(messages, f"sql_gen_{'zh' if language=='zh-tw' else 'en'}")
            if not raw:
                logger.warning("SQL_GEN_EMPTY: sig=%s", sig)
                return "SELECT 1 WHERE 1=0"

            final_sql = self._finalize_sql(raw)
            dt = time.perf_counter() - t0
            logger.info("SQL_GEN_OK: sig=%s time=%.2fs len=%d", sig, dt, len(final_sql))
            logger.debug("SQL_GEN_SQL: sig=%s\n%s", sig, final_sql)
            return final_sql or "SELECT 1 WHERE 1=0"
        except Exception as e:
            dt = time.perf_counter() - t0
            logger.error("SQL_GEN_FAIL: sig=%s time=%.2fs %s: %s", sig, dt, type(e).__name__, e, exc_info=True)
            return "SELECT 1 WHERE 1=0"

    def generate_sql_with_repair(
        self,
        question: str,
        schema: str,
        join_hints: str,
        *,
        language: Optional[Literal["zh-tw", "en"]] = None,
        failed_sql: Optional[str] = None,
        error_summary: Optional[str] = None,
        max_attempts: int = 3,
    ) -> Tuple[str, int]:
        t0 = time.perf_counter()
        language = language or detect_query_language(question)
        sig = self._create_query_signature(question, language)
        attempts = 0
        sql = ""

        logger.info("SQL_REPAIR_START: sig=%s lang=%s max_attempts=%d has_failed=%s",
                    sig, language, max_attempts, bool(failed_sql))

        while attempts < max_attempts:
            attempts += 1
            a0 = time.perf_counter()

            if attempts == 1 and not failed_sql:
                logger.debug("SQL_REPAIR_ATTEMPT: sig=%s attempt=%d fresh-gen", sig, attempts)
                sql = self.generate_sql(question, schema, join_hints, language)
            else:
                self.generation_stats["repair_attempts"] += 1
                if not self.llm_enabled:
                    logger.warning("SQL_REPAIR_ABORT: LLM disabled.")
                    break
                repair_prompt = self.repair_sql_prompt_zh if language == "zh-tw" else self.repair_sql_prompt_en
                if not repair_prompt:
                    logger.warning("SQL_REPAIR_ABORT: missing repair prompt for %s", language)
                    break
                messages = repair_prompt.format_messages(
                    failed_sql=failed_sql or sql,
                    error_summary=error_summary or "(no error message)",
                    schema=schema,
                    join_hints=join_hints,
                )
                raw = self._invoke_llm(messages, f"sql_repair_{'zh' if language=='zh-tw' else 'en'}")
                sql = self._finalize_sql(raw)
                if sql and sql != "SELECT 1 WHERE 1=0":
                    self.generation_stats["successful_repairs"] += 1

            logger.debug("SQL_REPAIR_ATTEMPT_DONE: sig=%s attempt=%d time=%.2fs len=%d",
                         sig, attempts, time.perf_counter() - a0, len(sql))

            if sql.strip():
                logger.info("SQL_REPAIR_OK: sig=%s attempts=%d total=%.2fs",
                            sig, attempts, time.perf_counter() - t0)
                return sql, attempts

        logger.warning("SQL_REPAIR_FAIL: sig=%s attempts=%d total=%.2fs",
                       sig, attempts or 1, time.perf_counter() - t0)
        return ("SELECT 1 WHERE 1=0", attempts or 1)

    # ────────────────────────────────────────────────────────────────────────────
    # Execution + repair loop
    # ────────────────────────────────────────────────────────────────────────────
    def run_query_with_llm_repair(
        self,
        db_service,
        user_question: str,
        schema: str,
        join_hints: str,
        *,
        params: Optional[Tuple[Any, ...]] = None,
        max_rows: int = 1000,
        query_timeout: Optional[int] = 10,
        max_attempts: int = 3,
    ) -> Tuple[List[Tuple], List[str], str, int]:
        t0 = time.perf_counter()
        language = detect_query_language(user_question)
        sig = self._create_query_signature(user_question, language)

        logger.info("QUERY_START: sig=%s lang=%s rows<=%d timeout=%s attempts<=%d",
                    sig, language, max_rows, query_timeout, max_attempts)

        # First SQL (one try)
        sql, attempts = self.generate_sql_with_repair(
            question=user_question,
            schema=schema,
            join_hints=join_hints,
            language=language,
            max_attempts=1,
        )

        # Execute
        try:
            a0 = time.perf_counter()
            rows, cols = db_service.run_select(sql, params=params, max_rows=max_rows, query_timeout=query_timeout)
            logger.info("QUERY_OK: sig=%s attempts=%d rows=%d cols=%d exec=%.2fs total=%.2fs",
                        sig, attempts, len(rows), len(cols),
                        time.perf_counter() - a0, time.perf_counter() - t0)
            logger.info("QUERY_SQL_OK: sig=%s sql=%s", sig, sql[:300])
            return rows, cols, sql, attempts
        except DBServiceQueryError as e:
            logger.warning("QUERY_FAIL: sig=%s attempt=1 err=%s: %s", sig, type(e).__name__, str(e)[:240])
            if not self._is_repairable_error(e):
                logger.error("QUERY_ABORT: non-repairable %s", type(e).__name__)
                raise

            error_details = self._build_error_summary(e, language)
            last_sql = sql

            while attempts < max_attempts:
                attempts += 1
                logger.debug("QUERY_REPAIR: sig=%s attempt=%d", sig, attempts)
                sql, _ = self.generate_sql_with_repair(
                    question=user_question,
                    schema=schema,
                    join_hints=join_hints,
                    language=language,
                    failed_sql=last_sql,
                    error_summary=error_details,
                    max_attempts=1,
                )
                try:
                    a0 = time.perf_counter()
                    rows, cols = db_service.run_select(sql, params=params, max_rows=max_rows, query_timeout=query_timeout)
                    logger.info("QUERY_REPAIR_OK: sig=%s attempts=%d rows=%d exec=%.2fs total=%.2fs",
                                sig, attempts, len(rows), time.perf_counter() - a0, time.perf_counter() - t0)
                    logger.info("QUERY_SQL_REPAIRED: sig=%s sql=%s", sig, sql[:300])
                    return rows, cols, sql, attempts
                except DBServiceQueryError as e2:
                    logger.warning("QUERY_REPAIR_FAIL: sig=%s attempt=%d %s: %s",
                                   sig, attempts, type(e2).__name__, str(e2)[:240])
                    if not self._is_repairable_error(e2):
                        logger.error("QUERY_REPAIR_ABORT: non-repairable at attempt %d", attempts)
                        raise
                    last_sql = sql
                    error_details = self._build_error_summary(e2, language)

            logger.error("QUERY_EXHAUSTED: sig=%s attempts=%d total=%.2fs", sig, attempts, time.perf_counter() - t0)
            raise

    def _build_error_summary(self, e: DBServiceQueryError, language: Literal["zh-tw", "en"]) -> str:
        parts = []
        if getattr(e, "category", None):
            parts.append(f"category={e.category}")
        if getattr(e, "db_code", None) is not None:
            parts.append(f"db_code={e.db_code}")
        if getattr(e, "sqlstate", None):
            parts.append(f"sqlstate={e.sqlstate}")
        meta = "; ".join(parts) if parts else ""
        msg = f"{type(e).__name__}: {str(e)}"
        if language == "zh-tw":
            msg = f"錯誤類型: {type(e).__name__}: {str(e)}"
        return f"{msg} ({meta})" if meta else msg

    # ────────────────────────────────────────────────────────────────────────────
    # Explanations
    # ────────────────────────────────────────────────────────────────────────────
    def generate_explanation(self, question: str, row_count: int, columns: List[str],
                             aggregates: Dict[str, Any], sample_text: str) -> str:
        return self._generate_explanation_internal(question, row_count, columns, aggregates, sample_text, "en")

    def generate_explanation_chinese(self, question: str, row_count: int, columns: List[str],
                                     aggregates: Dict[str, Any], sample_text: str) -> str:
        return self._generate_explanation_internal(question, row_count, columns, aggregates, sample_text, "zh-tw")

    def _generate_explanation_internal(self, question: str, row_count: int, columns: List[str],
                                       aggregates: Dict[str, Any], sample_text: str,
                                       language: Literal["zh-tw", "en"]) -> str:
        if not self.llm_enabled:
            return self._fallback_explanation(aggregates, language)

        prompt = self.explanation_prompt_zh if language == "zh-tw" else self.explanation_prompt_en
        if not prompt:
            return self._fallback_explanation(aggregates, language)

        import json as _json
        msgs = prompt.format_messages(
            question=question,
            row_count=row_count,
            columns=", ".join(columns) if columns else "(none)",
            aggregates_json=_json.dumps(aggregates, ensure_ascii=False),
            sample_text=sample_text,
        )
        resp = self._invoke_llm(msgs, f"explain_{'zh' if language=='zh-tw' else 'en'}")
        return (resp or "").strip() or self._fallback_explanation(aggregates, language)

    def _fallback_explanation(self, aggregates: Dict[str, Any], language: Literal["zh-tw", "en"] = "en") -> str:
        rc = int(aggregates.get("row_count", 0) or 0)
        up = aggregates.get("unique_people")
        bt = aggregates.get("by_leave_type") or {}
        th = aggregates.get("total_hours")

        if language == "zh-tw":
            parts = [f"{rc} 筆記錄。"]
            if up is not None: parts.append(f"{up} 位不重複人員。")
            if bt:
                total = sum(bt.values())
                if total:
                    top = sorted(bt.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    parts.append("主要請假類型：" + "、".join(f"{k}（{v}，{round(v/total*100,1)}%）" for k, v in top))
            if th: parts.append(f"總時數：{th}")
            return " ".join(parts)
        else:
            parts = [f"{rc} records."]
            if up is not None: parts.append(f"{up} unique people.")
            if bt:
                total = sum(bt.values())
                if total:
                    top = sorted(bt.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    parts.append("Top leave types: " + ", ".join(f"{k} ({v}, {round(v/total*100,1)}%)" for k, v in top))
            if th: parts.append(f"Total hours: {th}")
            return " ".join(parts)

    # ────────────────────────────────────────────────────────────────────────────
    # Convenience & stats
    # ────────────────────────────────────────────────────────────────────────────
    def run_query_with_llm_repair_language_aware(
        self,
        db_service,
        user_question: str,
        original_language: Literal["zh-tw", "en"],
        schema: str,
        join_hints: str,
        *,
        max_rows: int = 1000,
        query_timeout: int = 10,
        max_attempts: int = 3,
        **kwargs,
    ) -> Tuple[List, List[str], str, int]:
        logger.debug("LANGUAGE_AWARE_QUERY: original_lang=%s → main path", original_language)
        return self.run_query_with_llm_repair(
            db_service=db_service,
            user_question=user_question,
            schema=schema,
            join_hints=join_hints,
            params=None,
            max_rows=max_rows,
            query_timeout=query_timeout,
            max_attempts=max_attempts,
        )

    def get_service_stats(self) -> Dict[str, Any]:
        total = self.generation_stats["total_requests"]
        succ = self.generation_stats["successful_generations"]
        repair_attempts = self.generation_stats["repair_attempts"]
        return {
            "service_enabled": self.llm_enabled,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "total_requests": total,
            "successful_generations": succ,
            "failed_generations": self.generation_stats["failed_generations"],
            "success_rate_percent": round((succ / max(total, 1)) * 100, 2),
            "repair_attempts": repair_attempts,
            "successful_repairs": self.generation_stats["successful_repairs"],
            "repair_rate_percent": round((repair_attempts / max(total, 1)) * 100, 2),
            "repair_success_rate_percent": round((self.generation_stats["successful_repairs"] / max(repair_attempts, 1)) * 100, 2),
            "avg_generation_time_seconds": round(self.generation_stats["avg_generation_time"], 3),
            "has_memory": bool(self.memory),
            "prompts_initialized": all([
                self.sql_prompt_en, self.sql_prompt_zh,
                self.repair_sql_prompt_en, self.repair_sql_prompt_zh,
                self.explanation_prompt_en, self.explanation_prompt_zh,
            ]),
        }

    def reset_stats(self):
        logger.info("SERVICE_STATS_RESET")
        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "repair_attempts": 0,
            "successful_repairs": 0,
            "total_tokens_used": 0,
            "avg_generation_time": 0.0,
        }

    def _simple_completion(self, system_prompt: str, user_prompt: str) -> str:
        if not self.llm_enabled or not ChatPromptTemplate:
            return ""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt) if SystemMessagePromptTemplate else "",
                HumanMessagePromptTemplate.from_template("{u}")  # type: ignore
            ])
            msgs = prompt.format_messages(u=user_prompt)
            return self._invoke_llm(msgs, "simple_completion") or ""
        except Exception as e:
            logger.error("SIMPLE_COMPLETION_FAIL: %s: %s", type(e).__name__, e)
            return ""


# Backward compatibility aliases
OpenAIService = UnifiedBilingualOpenAIService
LanguageAwareOpenAIService = UnifiedBilingualOpenAIService
