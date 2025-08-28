# backend/app/services/nlp_service.py
from __future__ import annotations

import os, re, time, logging, json
from typing import Dict, Any, List, Optional, Tuple

# Optional AWS
import boto3
from botocore.exceptions import ClientError

# Optional OpenAI
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.schema import BaseMessage, HumanMessage
    from langchain.memory import ConversationBufferMemory
except Exception:
    ChatOpenAI = None
    ChatPromptTemplate = None
    SystemMessagePromptTemplate = None
    HumanMessagePromptTemplate = None
    BaseMessage = None
    HumanMessage = None
    ConversationBufferMemory = None

from app.services.leave_vector import build_leave_index
from app.services.db_service import SQLServerDatabaseService, DatabaseQueryError
from app.services.person_resolver import PersonResolver

logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_REGION", "ap-southeast-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)

class NLPService:
    """
    Pipeline:
      - detect/translate (AWS optional)
      - retrieve relevant leave tables (runtime index)
      - generate ONE SELECT (LLM optional)
      - execute safely (SELECT-only)
      - enrich PERSONID -> names
    """

    def __init__(self, db_service: SQLServerDatabaseService, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1, **_):
        self.db_service = db_service

        # AWS (optional)
        self.comprehend_client = None
        self.translate_client = None
        try:
            if AWS_ACCESS_KEY and AWS_SECRET_KEY:
                self.comprehend_client = boto3.client(
                    "comprehend", region_name=REGION,
                    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY
                )
                self.translate_client = boto3.client(
                    "translate", region_name=REGION,
                    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY
                )
                logger.info(f"AWS clients initialized for region={REGION}")
            else:
                logger.info("AWS credentials not set; defaulting to English.")
        except Exception as e:
            logger.warning(f"AWS init failed: {e}")

        # LLM (optional)
        self.llm = None
        self.llm_enabled = bool(OPENAI_API_KEY) and ChatOpenAI is not None
        if self.llm_enabled:
            try:
                try:
                    self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=OPENAI_API_KEY)
                except TypeError:
                    self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY)
                logger.info(f"LLM initialized model={model_name}")
            except Exception as e:
                logger.warning(f"LLM init failed: {e}")
                self.llm = None
                self.llm_enabled = False
        else:
            logger.info("LLM disabled or langchain_openai not installed.")

        self.memory = ConversationBufferMemory(return_messages=True) if ConversationBufferMemory else None

        # Initialize SQL prompt (this was missing!)
        if ChatPromptTemplate:
            self.sql_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are an expert SQL analyst for HR leave & attendance data. "
                    "Generate ONE safe SELECT query based on the user's question. "
                    "Rules:\n"
                    "- Only SELECT statements allowed\n"
                    "- Use provided table schemas exactly as shown\n"
                    "- Join tables using the hints provided when needed\n"
                    "- Use appropriate WHERE clauses for date filtering\n"
                    "- Include PERSONID when querying people data\n"
                    "- Return only the SQL query, no explanations\n"
                    "- Use CAST(date_column AS date) for date comparisons\n"
                    "- For 'today' queries, use CAST(GETDATE() AS date)\n\n"
                    "Available schemas:\n{schema}\n\n"
                    "Join hints:\n{join_hints}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "Generate a SQL SELECT query for: {query}"
                ),
            ])
        else:
            self.sql_prompt = None

        # Leave index (runtime-built)
        try:
            self.vector = build_leave_index()
            logger.info("Leave vector index ready.")
        except Exception as e:
            self.vector = None
            logger.warning(f"Leave vector unavailable: {e}")

        # People resolver
        self.person_resolver = PersonResolver(db_service=self.db_service)

        # Result explanation prompt (English; we translate after)
        if ChatPromptTemplate:
            self.explanation_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a data analyst for HR leave & attendance. "
                    "Write a short, business-friendly summary of the results. "
                    "Prefer 3–6 bullet points or 2–4 sentences. "
                    "Include the total people/rows, any breakdowns (e.g., by leave type), notable patterns, "
                    "and anything actionable. Do not include SQL or code; be concise."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Question: {question}\n"
                    "Row count: {row_count}\n"
                    "Columns: {columns}\n"
                    "Aggregates (JSON): {aggregates_json}\n"
                    "Sample rows (truncated):\n{sample_text}\n\n"
                    "Write the summary in English."
                ),
            ])
        else:
            self.explanation_prompt = None

        self.last_language: Optional[str] = None
        self.last_confidence: Optional[float] = None

    # ---- health for /api/health
    def vector_status(self) -> Dict[str, Any]:
        try:
            return self.vector.health_check() if self.vector else {"ready": False, "reason": "no index"}
        except Exception as e:
            return {"ready": False, "error": str(e)}

    # ---- language helpers
    @staticmethod
    def _normalize_lang(code: str) -> str:
        if not code: return "en-US"
        c = code.lower()
        if c.startswith("en"): return "en-US"
        if c in ("zh-tw","zh-hant"): return "zh-TW"
        if c in ("zh","zh-cn","zh-hans"): return "zh-CN"
        return "en-US"

    def detect_language(self, text: str) -> Tuple[str, float]:
        if not self.comprehend_client:
            self.last_language, self.last_confidence = "en-US", 1.0
            return "en-US", 1.0
        try:
            resp = self.comprehend_client.detect_dominant_language(Text=text)
            if resp.get("Languages"):
                top = resp["Languages"][0]
                lang = self._normalize_lang(top["LanguageCode"])
                conf = float(top.get("Score", 0.0))
                self.last_language, self.last_confidence = lang, conf
                return lang, conf
        except ClientError as e:
            logger.error(f"AWS Comprehend error: {e}")
        except Exception as e:
            logger.error(f"Comprehend failure: {e}", exc_info=True)
        self.last_language, self.last_confidence = "en-US", 0.0
        return "en-US", 0.0

    def translate_to_english(self, text: str, src_lang: str) -> str:
        if src_lang == "en-US" or not self.translate_client:
            return text
        code = "zh-TW" if src_lang == "zh-TW" else "zh"
        try:
            return self.translate_client.translate_text(Text=text, SourceLanguageCode=code, TargetLanguageCode="en")["TranslatedText"]
        except Exception:
            return text

    def translate_from_english(self, text: str, tgt_lang: str) -> str:
        if tgt_lang == "en-US" or not self.translate_client:
            return text
        code = "zh-TW" if tgt_lang == "zh-TW" else "zh"
        try:
            return self.translate_client.translate_text(Text=text, SourceLanguageCode="en", TargetLanguageCode=code)["TranslatedText"]
        except Exception:
            return text

    # ---- retrieval helpers
    def _find_relevant_tables(self, english_query: str, schema_filter: Optional[str] = None, rid: Optional[str] = None) -> List[Tuple[str, float]]:
        try:
            hits = self.vector.search_relevant_tables(english_query, top_k=5) if self.vector else []
            if schema_filter:
                hits = [(t, s) for (t, s) in hits if t.lower().startswith(schema_filter.lower() + ".")]
            return hits
        except Exception as e:
            logger.warning(f"rid={rid} vector search failed: {e}")
            return []

    def _join_hints_for(self, tables: List[str]) -> str:
        try:
            hints = self.vector.join_hints(tables) if self.vector else []
            return "\n".join(hints) if hints else "None"
        except Exception:
            return "None"

    def _schema_context_for(self, tables: List[str], max_cols: int = 12) -> str:
        if not tables:
            return "No relevant tables found"
        # get actual columns from DB for top 3 tables + PSNACCOUNT_D if not present
        pick = list(dict.fromkeys(tables[:3] + (["dbo.PSNACCOUNT_D"] if "dbo.PSNACCOUNT_D" not in tables[:3] else [])))
        return self.db_service.get_compact_schema_for(pick, max_columns_per_table=max_cols)
    
    def _col_index(self, columns: List[str], *candidates: str) -> Optional[int]:
        """Case-insensitive column finder."""
        if not columns:
            return None
        lookup = {c.lower(): i for i, c in enumerate(columns)}
        for name in candidates:
            i = lookup.get(name.lower())
            if i is not None:
                return i
        return None

    def _compute_aggregates(self, rows: List[Tuple], columns: List[str]) -> Dict[str, Any]:
        """Lightweight, deterministic stats used by the LLM (and as fallback if LLM is off)."""
        agg: Dict[str, Any] = {}
        if not rows:
            return {"row_count": 0}

        idx_person  = self._col_index(columns, "PERSONID")
        idx_empid   = self._col_index(columns, "EMPLOYEEID")
        idx_name    = self._col_index(columns, "TRUENAME", "Name")
        idx_type    = self._col_index(columns, "ATTENDANCETYPE", "LEAVETYPE")
        idx_hours   = self._col_index(columns, "HOURS")
        idx_start_d = self._col_index(columns, "STARTDATE", "WORKDATE", "StartDate", "WORKDATES")
        idx_end_d   = self._col_index(columns, "ENDDATE",   "EndDate",   "WORKDATEE")

        # uniques
        people: set = set()
        if idx_person is not None:
            people = {str(r[idx_person]) for r in rows if len(r) > idx_person and r[idx_person] is not None}
        elif idx_empid is not None:
            people = {str(r[idx_empid]) for r in rows if len(r) > idx_empid and r[idx_empid] is not None}

        # counts by leave type
        type_counts: Dict[str, int] = {}
        if idx_type is not None:
            for r in rows:
                if len(r) <= idx_type: 
                    continue
                v = r[idx_type]
                key = str(v).strip() if v is not None else "(unknown)"
                type_counts[key] = type_counts.get(key, 0) + 1

        # total hours (if present)
        total_hours = 0.0
        if idx_hours is not None:
            for r in rows:
                if len(r) <= idx_hours: 
                    continue
                v = r[idx_hours]
                try:
                    total_hours += float(v) if v is not None else 0.0
                except Exception:
                    pass

        # names sample
        names_sample: List[str] = []
        if idx_name is not None:
            for r in rows[:15]:
                if len(r) > idx_name and r[idx_name]:
                    names_sample.append(str(r[idx_name]))
            names_sample = [n for n in names_sample if n][:10]

        # date window (stringify; rows may be datetime.date/datetime)
        def _as_date_str(v):
            try:
                return str(v.date()) if hasattr(v, "date") else str(v)
            except Exception:
                return str(v)

        start_dates: List[str] = []
        end_dates: List[str]   = []
        if idx_start_d is not None:
            start_dates = [_as_date_str(r[idx_start_d]) for r in rows if len(r) > idx_start_d and r[idx_start_d] is not None]
        if idx_end_d is not None:
            end_dates = [_as_date_str(r[idx_end_d]) for r in rows if len(r) > idx_end_d and r[idx_end_d] is not None]

        agg.update({
            "row_count": len(rows),
            "unique_people": len(people) if people else None,
            "by_leave_type": type_counts if type_counts else None,
            "total_hours": round(total_hours, 2) if total_hours else None,
            "names_sample": names_sample if names_sample else None,
            "min_start_date": min(start_dates) if start_dates else None,
            "max_end_date": max(end_dates) if end_dates else None,
        })
        return agg

    def _format_sample_text(self, rows: List[Tuple], columns: List[str], max_rows: int = 20, max_chars: int = 1800) -> str:
        """Tiny, readable sample to give the LLM some shape without dumping everything."""
        if not rows or not columns:
            return "No data sample."
        # choose a small, human-friendly subset of columns to print first if present
        preferred = ["TRUENAME", "Name", "EMPLOYEEID", "PERSONID", "ATTENDANCETYPE", "LEAVETYPE", "HOURS", "STARTDATE", "ENDDATE", "WORKDATE"]
        keep_idx: List[int] = []
        seen = set()
        for p in preferred:
            i = self._col_index(columns, p)
            if i is not None and i not in seen:
                keep_idx.append(i); seen.add(i)
        # ensure at least first 5 columns if none matched
        if not keep_idx:
            keep_idx = list(range(min(5, len(columns))))

        hdr = [columns[i] for i in keep_idx]
        lines = [" | ".join(hdr)]
        for r in rows[:max_rows]:
            vals = []
            for i in keep_idx:
                try:
                    v = r[i] if i < len(r) else None
                    s = "" if v is None else str(v)
                    vals.append(s)
                except Exception:
                    vals.append("")
            lines.append(" | ".join(vals))
            if sum(len(x) for x in lines) > max_chars:
                lines.append("... (truncated)")
                break
        return "\n".join(lines)

    def _generate_explanation_text(self, question: str, rows: List[Tuple], columns: List[str]) -> str:
        """Use LLM if available; otherwise produce a deterministic summary."""
        aggregates = self._compute_aggregates(rows, columns)
        sample = self._format_sample_text(rows, columns)

        # If LLM disabled, deterministic text:
        if not (self.llm_enabled and self.explanation_prompt):
            parts = []
            parts.append(f"{aggregates.get('row_count', 0)} records.")
            if aggregates.get("unique_people") is not None:
                parts.append(f"{aggregates['unique_people']} unique people.")
            if aggregates.get("by_leave_type"):
                # take top 3
                by_type = aggregates["by_leave_type"]
                top_types = sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)[:3]
                parts.append("Top leave types: " + ", ".join(f"{k} ({v})" for k, v in top_types))
            if aggregates.get("total_hours"):
                parts.append(f"Total hours: {aggregates['total_hours']}")
            return " ".join(parts)

        try:
            msg = self.explanation_prompt.format_messages(
                question=question,
                row_count=len(rows),
                columns=", ".join(columns) if columns else "(none)",
                aggregates_json=json.dumps(aggregates, ensure_ascii=False),
                sample_text=sample,
            )
            resp = self.llm.invoke(msg)
            return (str(resp.content) or "").strip()
        except Exception as e:
            logger.error(f"Explanation LLM failed: {e}", exc_info=True)
            # fallback deterministic
            parts = []
            parts.append(f"{aggregates.get('row_count', 0)} records.")
            if aggregates.get("unique_people") is not None:
                parts.append(f"{aggregates['unique_people']} unique people.")
            if aggregates.get("by_leave_type"):
                # take top 3
                by_type = aggregates["by_leave_type"]
                top_types = sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)[:3]
                parts.append("Top leave types: " + ", ".join(f"{k} ({v})" for k, v in top_types))
            if aggregates.get("total_hours"):
                parts.append(f"Total hours: {aggregates['total_hours']}")
            return " ".join(parts)

    # ---- LLM helper
    def _run_openai_chain(self, messages: List[BaseMessage]) -> str:  # type: ignore
        if not (self.llm_enabled and self.llm and messages):
            return "SELECT 1 WHERE 1=0"
        try:
            resp = self.llm.invoke(messages)
            content = str(resp.content)
            if self.memory:
                last_user = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
                if last_user:
                    self.memory.save_context({"input": last_user.content}, {"output": content})
            return content
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return "SELECT 1 WHERE 1=0"

    # ---- SQL guard + exec
    def _guard_sql(self, sql: str) -> str:
        s = (sql or "").strip()
        if not s: return "SELECT 1 WHERE 1=0"
        low = re.sub(r"\s+", " ", s).lower()
        allowed_start = low.startswith("select ") or low.startswith("with ")
        forbidden = re.search(r"\b(insert|update|delete|merge|drop|alter|truncate|create|grant|revoke|exec|execute|into)\b", low)
        if not allowed_start or forbidden:
            return "SELECT 1 WHERE 1=0"
        # single statement only
        body = s[:-1] if s.endswith(";") else s
        if ";" in body:
            return "SELECT 1 WHERE 1=0"
        return s

    def _execute_sql(self, sql: str, rid: Optional[str]) -> Tuple[List[Tuple], List[str], Optional[str]]:
        if not sql or sql.strip().lower() == "select 1 where 1=0":
            return [], [], None
        try:
            rows, cols = self.db_service.run_select(sql, max_rows=200)
            logger.info(f"rid={rid} SQL ok rows={len(rows)} cols={len(cols)}")
            return rows, cols, None
        except DatabaseQueryError as e:
            logger.error(f"rid={rid} SQL failed: {e}")
            return [], [], str(e)
        except Exception as e:
            logger.error(f"rid={rid} SQL unexpected: {e}", exc_info=True)
            return [], [], str(e)

    # ---- deterministic fallback for common leave ask
    def _fallback_sql(self, english: str) -> Optional[str]:
        q = english.lower()
        if "leave" in q and any(w in q for w in ("today", "now", "currently")):
            # "Who's on leave today?"
            return (
                "SELECT DISTINCT "
                "  P.TRUENAME AS Name, P.EMPLOYEEID, L.PERSONID, "
                "  CAST(L.STARTDATE AS date) AS StartDate, L.STARTTIME, "
                "  CAST(L.ENDDATE AS date)   AS EndDate,   L.ENDTIME, "
                "  L.HOURS, L.ATTENDANCETYPE "
                "FROM dbo.ATDLEAVEDATA AS L "
                "LEFT JOIN dbo.PSNACCOUNT_D AS P "
                "  ON P.PERSONID = L.PERSONID "
                "WHERE CAST(GETDATE() AS date) "
                "      BETWEEN CAST(L.STARTDATE AS date) AND CAST(L.ENDDATE AS date)"
            )
        return None

    # ---- enrichment (non-destructive: return mapping)
    def _detect_personid_idxs(self, columns: List[str]) -> List[int]:
        hits = []
        for i, c in enumerate(columns or []):
            if "personid" in (c or "").lower().replace("_",""):
                hits.append(i)
        return hits

    def _enrich_people(self, rows: List[Tuple], columns: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
        idxs = self._detect_personid_idxs(columns)
        if not rows or not idxs:
            return {}
        ids = set()
        for r in rows:
            for i in idxs:
                if i < len(r) and r[i]:
                    ids.add(str(r[i]).strip())
        try:
            return self.person_resolver.resolve_many(list(ids))
        except Exception as e:
            logger.warning(f"Person resolve failed: {e}")
            return {}

    # ---- main
    def process_complete_query(self, user_input: str, schema_name: Optional[str] = "dbo", rid: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            lang, conf = self.detect_language(user_input)
            english = self.translate_to_english(user_input, lang)
            logger.info(f"rid={rid} query={user_input!r} lang={lang} conf={conf:.2f}")

            # retrieval
            rel = self._find_relevant_tables(english, schema_filter=schema_name, rid=rid)
            rel_tables = [t for (t, _) in rel]
            join_hints = self._join_hints_for(rel_tables)
            schema_ctx = self._schema_context_for(rel_tables)

            # SQL: try LLM, then fallback template if needed
            if self.sql_prompt and self.llm_enabled and rel_tables:
                prompt = self.sql_prompt.format_messages(query=english, schema=schema_ctx, join_hints=join_hints)
                sql_raw = (self._run_openai_chain(prompt) or "").strip()
            else:
                sql_raw = ""

            sql = self._guard_sql(sql_raw)
            if sql.lower() == "select 1 where 1=0":
                alt = self._fallback_sql(english)
                if alt:
                    sql = alt

            # execute
            rows, columns, execution_error = self._execute_sql(sql, rid)

            # enrich
            resolved_people = self._enrich_people(rows, columns)

            # explanation
            if execution_error:
                explanation_en = f"Query execution failed: {execution_error}"
            else:
                explanation_en = self._generate_explanation_text(english, rows, columns)

            localized = self.translate_from_english(explanation_en, lang)

            resp = {
                "original_text": user_input,
                "detected_language": lang,
                "language_confidence": conf,
                "english_text": english,
                "intent": "generic",
                "schema": schema_name,
                "relevant_tables": [{"table": t, "score": round(s, 3)} for (t, s) in rel],
                "generated_sql": sql or "SELECT 1 WHERE 1=0",
                "execution_successful": execution_error is None,
                "execution_error": execution_error,
                "columns": columns,
                "results": [list(r) for r in rows],
                "row_count": len(rows),
                "resolved_people": resolved_people,
                "columns_enriched": columns,
                "results_enriched": [list(r) for r in rows],  # unchanged rows; mapping is separate
                "explanation_english": explanation_en,
                "explanation_localized": localized,
                "summary": localized,
                "success": execution_error is None,
            }
            logger.info(f"rid={rid} pipeline ok ms={_ms(t0)}")
            return resp

        except Exception as e:
            logger.error(f"rid={rid} pipeline failed after {_ms(t0)}ms: {type(e).__name__}: {e}", exc_info=True)
            msg = "An error occurred while processing your query."
            return {
                "original_text": user_input,
                "detected_language": self.last_language or "en-US",
                "language_confidence": self.last_confidence or 0.0,
                "execution_successful": False,
                "execution_error": str(e),
                "summary": msg,
                "explanation_localized": msg,
                "success": False,
            }