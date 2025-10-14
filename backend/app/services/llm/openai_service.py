# backend/app/services/llm/openai_service_unified.py
from __future__ import annotations

import re
import os
import logging
import time
import hashlib
from typing import List, Optional, Dict, Any, Tuple, Literal
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Optional OpenAI / LangChain imports
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# DB exceptions (for repair classification)
# ────────────────────────────────────────────────────────────────────────────────
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

# Use ORG_TABLE name from leave_vector (resolves env override, e.g. [eHRAntung_DB].[dbo].[ORGStdStruct])
try:
    from app.services.leave_vector import ORG_TABLE as _ORG_TABLE
except Exception:
    _ORG_TABLE = "dbo.ORGStdStruct"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ENABLE_LLM_SUMMARY = os.getenv("LLM_SUMMARY", "1") == "1"
SQL_CACHE_SIZE = int(os.getenv("SQL_GEN_CACHE_SIZE", "64"))

# ────────────────────────────────────────────────────────────────────────────────
# Language detection
# ────────────────────────────────────────────────────────────────────────────────
def detect_query_language(text: str) -> Literal["zh-tw", "en"]:
    if not text or not text.strip():
        return "en"
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin_num = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isdigit()))
    if chinese_chars >= 2 and chinese_chars >= latin_num:
        return "zh-tw"
    if any(k in text for k in ["請假", "考勤", "部門", "員工", "今天", "現在", "統計", "趨勢"]):
        return "zh-tw"
    return "en"


# ────────────────────────────────────────────────────────────────────────────────
# Unified service
# ────────────────────────────────────────────────────────────────────────────────
class UnifiedBilingualOpenAIService:
    """
    Bilingual T-SQL generation/repair/explanation with strict SELECT-only guardrails.
    Optimizations:
      • Small LRU cache for SQL generations.
      • Safer SQL post-processing (joins, LIMIT→TOP, identifier normalization).
      • Deterministic, rule-based summarizer with optional LLM polish.
    """

    # -------------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------------
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
            "avg_generation_time": 0.0,
        }

        # Small in-process LRU cache for generated SQL
        self._sql_cache: "OrderedDict[str, str]" = OrderedDict()

        self.sql_prompt_en = None
        self.sql_prompt_zh = None
        self.repair_sql_prompt_en = None
        self.repair_sql_prompt_zh = None
        self.explanation_prompt_en = None
        self.explanation_prompt_zh = None

        self._initialize_llm()
        self._initialize_all_prompts()

    # -------------------------------------------------------------------------
    # LLM init
    # -------------------------------------------------------------------------
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
            logger.info(
                "LLM INITIALIZED: model=%s temp=%.2f memory=%s init_ms=%d",
                self.model_name,
                self.temperature,
                bool(self.memory),
                int((time.perf_counter() - t0) * 1000),
            )
        except Exception as e:
            logger.error("LLM INIT FAILED: %s: %s", type(e).__name__, e, exc_info=True)
            self.llm = None
            self.llm_enabled = False

    # -------------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------------
    def _initialize_all_prompts(self):
        if not ChatPromptTemplate:
            logger.warning("PROMPTS DISABLED: ChatPromptTemplate not available.")
            return

        tail_en = (
            "\n\nFollow the context STRICTLY. If any rule here conflicts with the context above, the context WINS.\n"
            "Hard guardrails:\n"
            "• Use CTEs (WITH ...) and fully-qualified table names.\n"
            "• Join to person dimension (dbo.PSNACCOUNT) for names/employee_id when needed; cast PERSONID as NVARCHAR(100).\n"
            "• For approved leave, include VALIDATED = 1 when applicable.\n"
            "• Do NOT invent columns; use only provided schema.\n"
            "• Prefer deterministic ORDER BY when returning ranked results.\n"
            "• Only SELECT; no DDL/DML/EXEC.\n"
        )
        tail_zh = (
            "\n\n請嚴格遵守上方提供的上下文內容，若與下列規則衝突，以上下文為準。\n"
            "硬性規範：\n"
            "• 使用 CTE（WITH ...）與完整資料表名稱。\n"
            "• 需要姓名/員編時，JOIN 至人員維度（dbo.PSNACCOUNT），且 PERSONID 請 CAST 成 NVARCHAR(100)。\n"
            "• 涉及已批准請假時加入 VALIDATED = 1（如適用）。\n"
            "• 不可臆造欄位；僅使用提供之結構。\n"
            "• 回傳排名結果時，請採用可重現的 ORDER BY。\n"
            "• 僅允許 SELECT；不可使用 DDL/DML/EXEC。\n"
        )

        self.sql_prompt_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "You are an expert T-SQL analyst for HR leave & attendance.\n"
                "You will be given a **context block** with business guardrails, examples, and schema.\n"
                "Return exactly ONE safe **SELECT-only** T-SQL query for SQL Server.\n"
                "Use provided date logic and anchoring if present.\n\n"
                "Context:\n{schema}\n\nJoin hints:\n{join_hints}" + tail_en
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "User question:\n{query}\n\nReturn only the SQL query. No markdown, no comments."
            ),
        ])

        self.sql_prompt_zh = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "你是請假/考勤領域的 T-SQL 專家。"
                "以下提供**上下文**（含業務規範、範例與資料庫結構）。\n"
                "請只回傳一個安全的 **僅限 SELECT** 的 SQL Server 查詢。\n"
                "日期邏輯與年份錨定請遵循上下文（如有）。\n\n"
                "上下文：\n{schema}\n\n關聯提示：\n{join_hints}" + tail_zh
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "使用者問題：\n{query}\n\n請只回傳 SQL 查詢語句，無需註解或 markdown。"
            ),
        ])

        # Repair prompts
        self.repair_sql_prompt_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "You fix failing SQL Server (T-SQL) queries.\n"
                "Output exactly one corrected **SELECT-only** T-SQL statement.\n"
                "Keep original intent; use only schema columns; respect GROUP BY rules; no comments.\n\n"
                "Context:\n{schema}\n\nJoin hints:\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "Database error:\n{error_summary}\n\nFailed SQL:\n{failed_sql}\n\nReturn only the corrected SQL."
            ),
        ])
        self.repair_sql_prompt_zh = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "你要修復失敗的 SQL Server (T-SQL) 查詢。\n"
                "請輸出一個修正後的 **僅限 SELECT** 的 T-SQL 語句，維持原意且僅用結構欄位；不得有註解。\n\n"
                "上下文：\n{schema}\n\n關聯提示：\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "資料庫錯誤：\n{error_summary}\n\n失敗的 SQL：\n{failed_sql}\n\n只回傳修正後的 SQL。"
            ),
        ])

        # Explanation prompts – used only to POLISH deterministic facts
        self.explanation_prompt_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "You are a business analyst. Rewrite the bullet points into a concise summary "
                "(3–6 bullets or 2–4 sentences). Keep to the facts provided; do not invent new numbers. "
                "If dates are present, state the period clearly. No SQL."
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "Facts:\n{facts}\n\nWrite the summary in English."
            ),
        ])
        self.explanation_prompt_zh = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "你是商業分析師。請將提供的重點整理成簡潔摘要（3–6 點或 2–4 句）。"
                "僅能使用提供的事實，不得臆造數字。若含日期，請清楚說明期間。不要有 SQL。"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "重點：\n{facts}\n\n請用繁體中文撰寫摘要。"
            ),
        ])

        logger.info("PROMPTS INITIALIZED.")

    # -------------------------------------------------------------------------
    # LLM invocation
    # -------------------------------------------------------------------------
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
            ) / max(n, 1)
            logger.info("LLM_OK: ctx=%s time=%.2fs len=%d avg=%.2fs",
                        context, dt, len(content), self.generation_stats["avg_generation_time"])
            return content
        except Exception as e:
            dt = time.perf_counter() - t0
            self.generation_stats["failed_generations"] += 1
            logger.error("LLM_FAIL: ctx=%s time=%.2fs %s: %s", context, dt, type(e).__name__, e, exc_info=True)
            return ""

    # -------------------------------------------------------------------------
    # SQL helpers (sanitize, postprocess, cache)
    # -------------------------------------------------------------------------
    _FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    _FIRST_SELECT_RE = re.compile(r"(?is)\bwith\b[\s\S]+?\bselect\b|\bselect\b")
    _PROHIBITED_RE = re.compile(
        r"(?is)\b(insert|update|delete|merge|drop|alter|create|truncate|exec|execute|grant|revoke)\b"
    )
    _P_PERSON_COLS = re.compile(r'\b(TRUENAME|EMPLOYEEID|BRANCHID)\b', re.IGNORECASE)
    _P_JOIN_EXISTS = re.compile(r'\bJOIN\s+(?:\[?eHRAntung_DB\]\.)?\[?dbo\]?\.\[?PSNACCOUNT\]?\s+(?:AS\s+)?p\b', re.IGNORECASE)
    _P_FROM_FACT = re.compile(
        r'\bFROM\s+(?:\[?eHRAntung_DB\]\.)?\[?dbo\]?\.\[?(ATDLEAVEDATA|ATDLEAVECANCELDATA)\]?\s+([a-zA-Z]\w*)\b',
        re.IGNORECASE
    )
    _ORG_COLS = re.compile(r'\b(UNITID|UNITNAME|UNITDISPLAYNAME|UNITCODE|department_name)\b', re.IGNORECASE)
    _ORG_JOIN_EXISTS = re.compile(r'\bJOIN\s+.*?\borg\b', re.IGNORECASE)

    def _qualify_person_columns(self, sql: str) -> str:
        s = re.sub(r'(?<!\.)\bTRUENAME\b', 'p.TRUENAME', sql, flags=re.IGNORECASE)
        s = re.sub(r'(?<!\.)\bEMPLOYEEID\b', 'p.EMPLOYEEID', s, flags=re.IGNORECASE)
        s = re.sub(r'(?<!\.)\bBRANCHID\b', 'p.BRANCHID', s, flags=re.IGNORECASE)
        return s

    def _inject_person_join_if_needed(self, sql: str) -> str:
        if not self._P_PERSON_COLS.search(sql):
            return sql
        fixed = self._qualify_person_columns(sql)
        if self._P_JOIN_EXISTS.search(fixed):
            return fixed
        m = self._P_FROM_FACT.search(fixed)
        fact_alias = (m.group(2) if m else 'l').strip()
        insert_join = f"\nLEFT JOIN dbo.PSNACCOUNT p ON p.PERSONID = {fact_alias}.PERSONID\n"
        fixed = re.sub(r'(\bFROM\b[^\n]*\n)', r'\1' + insert_join, fixed, count=1, flags=re.IGNORECASE)
        return fixed

    def _inject_org_join_if_needed(self, sql: str) -> str:
        if not self._ORG_COLS.search(sql):
            return sql
        if self._ORG_JOIN_EXISTS.search(sql):
            return sql
        s = self._inject_person_join_if_needed(sql)  # ensure p.BRANCHID
        insert_join = f"\nLEFT JOIN {_ORG_TABLE} org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
        s = re.sub(
            r'(\bFROM\b[\s\S]*?)(\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|$)',
            r'\1' + insert_join + r'\2',
            s, count=1, flags=re.IGNORECASE
        )
        return s

    def _extract_sql_from_text(self, text: str) -> str:
        if not text:
            return ""
        m = self._FENCE_RE.search(text)
        sql = m.group(1) if m else text
        sql = sql.strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.I)
        sql = re.sub(r"\s*```$", "", sql)
        m2 = self._FIRST_SELECT_RE.search(sql)
        if m2:
            sql = sql[m2.start():].strip()
        return sql

    def _normalize_id_quotes(self, sql: str) -> str:
        s = re.sub(r"`([^`]+)`", r"[\1]", sql)  # backticks → brackets
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
        if not sql:
            return ""
        s = sql.strip()
        if self._PROHIBITED_RE.search(s):
            logger.warning("SANITIZE: prohibited keyword detected; returning safe empty SELECT.")
            return "SELECT 1 WHERE 1=0"
        parts = [p.strip() for p in re.split(r";\s*(?=WITH\b|SELECT\b|$)", s, flags=re.I)]
        first_valid = next((p for p in parts if re.match(r"(?is)^(with\b|select\b)", p)), "")
        return first_valid or ""

    def _finalize_sql(self, sql: str) -> str:
        s = self._extract_sql_from_text(sql)
        s = self._normalize_id_quotes(s)
        s = self._tsql_limit_fix(s)
        s = self._ensure_select_only(s)
        return s.strip()

    def _postprocess_sql(self, sql: str) -> str:
        if not sql:
            return sql
        s = sql
        s = self._inject_person_join_if_needed(s)
        s = self._inject_org_join_if_needed(s)
        return s

    def _create_query_signature(self, query: str, language: str) -> str:
        normalized = re.sub(r"\s+", " ", (query or "").lower().strip())
        return hashlib.md5(f"{language}:{normalized}".encode()).hexdigest()[:12]

    def _sql_cache_key(self, query: str, schema: str, join_hints: str, language: str) -> str:
        base = f"{language}|{schema}|{join_hints}|{query}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]

    def _sql_cache_get(self, key: str) -> Optional[str]:
        if not SQL_CACHE_SIZE:
            return None
        val = self._sql_cache.get(key)
        if val is not None:
            # Refresh LRU
            self._sql_cache.move_to_end(key, last=True)
            logger.debug("SQL_CACHE_HIT: key=%s", key[:12])
        return val

    def _sql_cache_put(self, key: str, value: str) -> None:
        if not SQL_CACHE_SIZE:
            return
        self._sql_cache[key] = value
        self._sql_cache.move_to_end(key, last=True)
        while len(self._sql_cache) > SQL_CACHE_SIZE:
            evicted_key, _ = self._sql_cache.popitem(last=False)
            logger.debug("SQL_CACHE_EVICT: key=%s", evicted_key[:12])

    # -------------------------------------------------------------------------
    # Error classification
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # SQL generation & repair (with caching)
    # -------------------------------------------------------------------------
    def generate_sql(self, query: str, schema: str, join_hints: str,
                     language: Optional[Literal["zh-tw", "en"]] = None) -> str:
        t0 = time.perf_counter()
        if not self.llm_enabled:
            logger.warning("SQL_GEN: LLM disabled → fallback stub.")
            return "SELECT 1 WHERE 1=0"

        language = language or detect_query_language(query)
        sig = self._create_query_signature(query, language)
        cache_key = self._sql_cache_key(query, schema, join_hints, language)

        # LRU cache
        cached = self._sql_cache_get(cache_key)
        if cached:
            logger.info("SQL_GEN_CACHE_OK: sig=%s len=%d", sig, len(cached))
            return cached

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
            final_sql = self._postprocess_sql(final_sql)

            dt = time.perf_counter() - t0
            logger.info("SQL_GEN_OK: sig=%s time=%.2fs len=%d", sig, dt, len(final_sql))
            logger.debug("SQL_GEN_SQL: sig=%s\n%s", sig, final_sql)

            if final_sql and final_sql != "SELECT 1 WHERE 1=0":
                self._sql_cache_put(cache_key, final_sql)

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
                sql = self._postprocess_sql(sql)
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

    # -------------------------------------------------------------------------
    # Execution + repair loop
    # -------------------------------------------------------------------------
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

        sql, attempts = self.generate_sql_with_repair(
            question=user_question,
            schema=schema,
            join_hints=join_hints,
            language=language,
            max_attempts=1,
        )

        sql = self._postprocess_sql(sql)

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
                sql = self._postprocess_sql(sql)

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

    # -------------------------------------------------------------------------
    # EXPLANATIONS / SUMMARIES (rule-based core + optional LLM polish)
    # -------------------------------------------------------------------------
    def generate_explanation(self, question: str, row_count: int, columns: List[str],
                             aggregates: Dict[str, Any], sample_text: str) -> str:
        return self._generate_summary(question, row_count, columns, aggregates, sample_text, "en")

    def generate_explanation_chinese(self, question: str, row_count: int, columns: List[str],
                                     aggregates: Dict[str, Any], sample_text: str) -> str:
        return self._generate_summary(question, row_count, columns, aggregates, sample_text, "zh-tw")

    def _generate_summary(self, question: str, row_count: int, columns: List[str],
                          aggregates: Dict[str, Any], sample_text: str,
                          language: Literal["zh-tw", "en"]) -> str:
        """
        New summary path:
          1) Build deterministic bullet 'facts' from aggregates (no hallucinations).
          2) If LLM summary enabled & available, ask it to polish into 3–6 bullets / 2–4 sentences.
          3) Else render a crisp, localized template from the facts.
        """
        facts = self._facts_from_aggregates(question, row_count, columns, aggregates, sample_text, language)
        if not facts:
            # Absolute fallback—very short line
            return self._fallback_explanation(aggregates, language)

        # Optional LLM polish — constrained to provided facts
        if ENABLE_LLM_SUMMARY and self.llm_enabled and (self.explanation_prompt_zh if language == "zh-tw" else self.explanation_prompt_en):
            try:
                prompt = self.explanation_prompt_zh if language == "zh-tw" else self.explanation_prompt_en
                msgs = prompt.format_messages(facts="\n".join(f"- {f}" for f in facts))
                resp = self._invoke_llm(msgs, f"explain_{'zh' if language=='zh-tw' else 'en'}")
                clean = (resp or "").strip()
                # Trim runaway outputs
                if clean.count("\n") > 12:
                    clean = "\n".join(clean.splitlines()[:12])
                if clean:
                    return clean
            except Exception as e:
                logger.warning("SUMMARY_LLM_POLISH_FAIL: %s", e)

        # Deterministic textualization
        if language == "zh-tw":
            return "；".join(facts[:6])
        return "; ".join(facts[:6])

    # -------------------------------------------------------------------------
    # Deterministic facts builder
    # -------------------------------------------------------------------------
    def _facts_from_aggregates(
        self,
        question: str,
        row_count: int,
        columns: List[str],
        ag: Dict[str, Any],
        sample_text: str,
        lang: Literal["zh-tw", "en"],
    ) -> List[str]:
        facts: List[str] = []
        # Normalize aggregates dict to avoid KeyErrors
        ag = ag or {}
        safe_cols = ", ".join(columns) if columns else "(none)"

        # 1) Basic size
        if lang == "zh-tw":
            facts.append(f"資料筆數：{int(row_count or ag.get('row_count') or 0)}；欄位：{safe_cols}")
        else:
            facts.append(f"Rows: {int(row_count or ag.get('row_count') or 0)}; columns: {safe_cols}")

        # 2) Time window if present
        eff = ag.get("effective_as_of") or ag.get("as_of")
        rng = ag.get("effective_range") or ag.get("range")
        if rng and isinstance(rng, dict) and rng.get("start") and rng.get("end"):
            if lang == "zh-tw":
                facts.append(f"期間：{rng.get('start')} ～ {rng.get('end')}")
            else:
                facts.append(f"Period: {rng.get('start')} to {rng.get('end')}")
        elif eff:
            if lang == "zh-tw":
                facts.append(f"基準日：{eff}")
            else:
                facts.append(f"As of: {eff}")

        # 3) Totals
        total_hours = ag.get("total_hours") or ag.get("hours_total") or ag.get("sum_hours")
        unique_people = ag.get("unique_people") or ag.get("distinct_people") or ag.get("people_count")
        if total_hours is not None:
            if lang == "zh-tw":
                facts.append(f"總時數：{total_hours}")
            else:
                facts.append(f"Total hours: {total_hours}")
        if unique_people is not None:
            if lang == "zh-tw":
                facts.append(f"不重複人員數：{unique_people}")
            else:
                facts.append(f"Unique people: {unique_people}")

        # 4) Categorical breakdowns — be robust to different keys
        for key in ("by_leave_type", "by_attendance_type", "by_department", "by_dept", "by_unit"):
            data = ag.get(key)
            if isinstance(data, dict) and data:
                top = sorted(data.items(), key=lambda kv: (kv[1] if isinstance(kv[1], (int, float)) else 0), reverse=True)
                top = [kv for kv in top if kv[1]]  # drop zeros/None
                total = sum(v for _, v in top)
                if total:
                    head = top[:3]
                    if lang == "zh-tw":
                        facts.append("主要分類（{}）：{}".format(
                            key.replace("_", ""),
                            "、".join(f"{k}：{v}（{round(v/total*100,1)}%）" for k, v in head)
                        ))
                    else:
                        facts.append("Top categories ({}): {}".format(
                            key.replace("_", ""),
                            ", ".join(f"{k}: {v} ({round(v/total*100,1)}%)" for k, v in head)
                        ))

        # 5) People ranking if provided
        for key in ("top_people", "top_employees"):
            arr = ag.get(key)
            if isinstance(arr, list) and arr:
                head = arr[:3]
                # Try to accept dicts like {"name":..., "hours":...}
                def _fmt_person(p: Any) -> Optional[str]:
                    if isinstance(p, dict):
                        name = p.get("name") or p.get("TRUENAME") or p.get("person_name") or p.get("employee")
                        hours = p.get("hours") or p.get("total_hours") or p.get("sum")
                        if name and hours is not None:
                            return f"{name} {hours}"
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        return f"{p[0]} {p[1]}"
                    return None
                items = [x for x in ( _fmt_person(p) for p in head ) if x]
                if items:
                    facts.append(("Top people: " if lang=="en" else "時數最高人員：") + (", ".join(items) if lang=="en" else "、".join(items)))

        # 6) Trend detection if series provided
        for key in ("trend", "time_series", "by_date"):
            series = ag.get(key)
            if isinstance(series, list) and series:
                # Accept formats: [{"date": "...", "value": n}, {"date": "...", "hours": n}, ...]
                def _get_val(d: Dict[str, Any]) -> Optional[float]:
                    for f in ("value", "hours", "count", "people", "on_leave"):
                        if f in d:
                            try:
                                return float(d[f])
                            except Exception:
                                pass
                    return None
                vals = [ _get_val(x) for x in series if isinstance(x, dict) ]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 2:
                    first, last = vals[0], vals[-1]
                    change = None
                    try:
                        change = ((last - first) / (first if first else 1.0)) * 100.0
                    except Exception:
                        change = None
                    if change is not None:
                        if lang == "zh-tw":
                            facts.append(f"趨勢：期末相較期初 {('增加' if change>=0 else '減少')} {abs(round(change,1))}%")
                        else:
                            facts.append(f"Trend: last vs first {('up' if change>=0 else 'down')} {abs(round(change,1))}%")

        # 7) Guard: if we only have size line, add a safe fallback
        if len(facts) <= 1:
            facts.append(self._fallback_explanation(ag, lang))

        # Remove any duplicates / empties
        uniq = []
        seen = set()
        for f in facts:
            s = (f or "").strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        return uniq[:8]

    # -------------------------------------------------------------------------
    # Lightweight fallback summary (if aggregates too sparse)
    # -------------------------------------------------------------------------
    def _fallback_explanation(self, aggregates: Dict[str, Any], language: Literal["zh-tw", "en"] = "en") -> str:
        rc = int((aggregates or {}).get("row_count", 0) or 0)
        up = (aggregates or {}).get("unique_people")
        bt = (aggregates or {}).get("by_leave_type") or {}
        th = (aggregates or {}).get("total_hours")
        if language == "zh-tw":
            parts = [f"{rc} 筆記錄。"]
            if up is not None: parts.append(f"{up} 位不重複人員。")
            if bt:
                total = sum(v for v in bt.values() if isinstance(v, (int, float)))
                if total:
                    top = sorted(bt.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    parts.append("主要請假類型：" + "、".join(f"{k}（{v}，{round(v/total*100,1)}%）" for k, v in top))
            if th: parts.append(f"總時數：{th}")
            return " ".join(parts)
        else:
            parts = [f"{rc} records."]
            if up is not None: parts.append(f"{up} unique people.")
            if bt:
                total = sum(v for v in bt.values() if isinstance(v, (int, float)))
                if total:
                    top = sorted(bt.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    parts.append("Top leave types: " + ", ".join(f"{k} ({v}, {round(v/total*100,1)}%)" for k, v in top))
            if th: parts.append(f"Total hours: {th}")
            return " ".join(parts)

    # -------------------------------------------------------------------------
    # Convenience & stats
    # -------------------------------------------------------------------------
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
            "sql_cache_size": SQL_CACHE_SIZE,
            "sql_cache_items": len(self._sql_cache),
        }

    def reset_stats(self):
        logger.info("SERVICE_STATS_RESET")
        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "repair_attempts": 0,
            "successful_repairs": 0,
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
