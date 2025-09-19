# backend/app/services/db_service.py
# -*- coding: utf-8 -*-

import os
import re
import time
import uuid
import pyodbc
import logging
import contextvars
from typing import Any, Dict, List, Optional, Tuple, Literal

# -----------------------------------------------------------------------------
# Logging setup with Unicode support
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Request-scoped correlation id (set by FastAPI middleware; falls back to '-')
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("rid", default="-")

def set_request_id(rid: Optional[str] = None) -> str:
    """Allow middleware or tests to set the correlation id for downstream logs."""
    rid = rid or str(uuid.uuid4())
    _request_id_var.set(rid)
    return rid

def rid() -> str:
    """Current correlation id."""
    return _request_id_var.get()

def _mask(s: str, keep_last: int = 2) -> str:
    if not s:
        return ""
    return "*" * max(0, len(s) - keep_last) + s[-keep_last:]

def _safe_unicode_repr(text: str, max_length: int = 200) -> str:
    """Safely represent Unicode text for logging, handling Chinese characters."""
    if text is None:
        return ""
    try:
        s = str(text)
        if len(s) > max_length:
            s = s[:max_length] + "..."
        return repr(s) if any(ord(c) > 127 for c in s) else s
    except Exception:
        return f"<text-repr-error:{len(str(text))}chars>"

DB_LOG_SQL = os.getenv("DB_LOG_SQL", "0") == "1"
DB_LOG_PARAMS = os.getenv("DB_LOG_PARAMS", "0") == "1"
DB_LOG_UNICODE = os.getenv("DB_LOG_UNICODE", "1") == "1"

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class DatabaseQueryError(Exception):
    """Base custom exception for database failures; carries SQL + diagnostics."""
    def __init__(
        self,
        message: str,
        sql: str = "",
        *,
        sqlstate: Optional[str] = None,
        db_code: Optional[int] = None,
        category: Optional[str] = None,
    ):
        super().__init__(message)
        self.sql = sql
        self.sqlstate = sqlstate
        self.db_code = db_code
        self.category = category

class DatabaseConnectionError(DatabaseQueryError): ...
class DatabaseTimeoutError(DatabaseQueryError): ...
class DatabaseSyntaxError(DatabaseQueryError): ...
class PermissionDeniedError(DatabaseQueryError): ...
class TableNotFoundError(DatabaseQueryError): ...
class ColumnNotFoundError(DatabaseQueryError): ...
class DeadlockError(DatabaseQueryError): ...
class DatabaseOperationalError(DatabaseQueryError): ...
class DatabaseDataError(DatabaseQueryError): ...
class DatabaseIntegrityError(DatabaseQueryError): ...

# -----------------------------------------------------------------------------
# SQL Validation (Unicode-aware)
# -----------------------------------------------------------------------------
_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_LINE_COMMENT_RE = re.compile(r"(--|#).*?$", re.M)
_BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/", re.M)
_SELECT_PREFIX_RE = re.compile(r"^\s*select\b", re.I)
_CTE_PREFIX_RE = re.compile(r"^\s*with\b", re.I)

_BLOCKLIST_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|MERGE|EXEC|EXECUTE|TRUNCATE|GRANT|REVOKE|DENY|BACKUP|RESTORE|"
    r"xp_[A-Za-z_]+|sp_[A-Za-z_]+|OPENROWSET|OPENQUERY|BULK\s+INSERT|INTO)\b",
    re.I | re.UNICODE,
)

# Chinese comment patterns (for better comment stripping)
_CHINESE_COMMENT_PATTERNS = [
    re.compile(r"--\s*[\u4e00-\u9fff].*?$", re.M),
    re.compile(r"/\*[\s\S]*?[\u4e00-\u9fff][\s\S]*?\*/", re.M),
]

# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------
_NATIVE_CODE_RE = re.compile(r"\((\d{3,5})\)")

def _extract_odbc_error(e: Exception) -> Tuple[Optional[str], Optional[int], str]:
    sqlstate: Optional[str] = None
    native: Optional[int] = None
    msg = str(e)

    if isinstance(e, pyodbc.Error):
        try:
            if len(e.args) == 2 and isinstance(e.args[0], str):
                sqlstate = e.args[0] or None
                msg = e.args[1] or msg
            elif len(e.args) >= 1 and isinstance(e.args[0], str):
                msg = e.args[0]
        except Exception:
            pass

    m = _NATIVE_CODE_RE.search(msg)
    if m:
        try:
            native = int(m.group(1))
        except Exception:
            native = None
    return sqlstate, native, msg

def _classify_exception(e: Exception, *, sql: str = "") -> DatabaseQueryError:
    sqlstate, native, msg = _extract_odbc_error(e)
    text = (msg or "").lower()

    if sqlstate in {"08001", "08004", "08S01"} or "communication link failure" in text or "could not open a connection" in text:
        return DatabaseConnectionError(
            f"Database connection failed: {msg}", sql, sqlstate=sqlstate, db_code=native, category="connection"
        )

    if sqlstate in {"28000"} or "login failed" in text or "permission denied" in text or "is denied" in text:
        return PermissionDeniedError(
            f"Authentication/authorization error: {msg}", sql, sqlstate=sqlstate, db_code=native, category="authz"
        )

    if sqlstate in {"HYT00", "HYT01"} or "timeout expired" in text or "lock request time out" in text:
        return DatabaseTimeoutError(
            f"Database timeout: {msg}", sql, sqlstate=sqlstate, db_code=native, category="timeout"
        )

    if native == 1205 or "deadlock" in text:
        return DeadlockError(
            f"Transaction deadlock: {msg}", sql, sqlstate=sqlstate, db_code=native, category="deadlock"
        )

    if sqlstate in {"42S02", "S0002"} or "invalid object name" in text or "could not find object" in text:
        return TableNotFoundError(
            f"Table or view not found: {msg}", sql, sqlstate=sqlstate, db_code=native, category="not_found"
        )
    if sqlstate in {"42S22", "S0022"} or "invalid column name" in text or "unknown column" in text:
        return ColumnNotFoundError(
            f"Column not found: {msg}", sql, sqlstate=sqlstate, db_code=native, category="not_found"
        )

    if native in {4060} or "cannot open database" in text:
        return DatabaseConnectionError(
            f"Cannot open database: {msg}", sql, sqlstate=sqlstate, db_code=native, category="db_unavailable"
        )

    if sqlstate == "42000" or "incorrect syntax" in text or "parse" in text:
        return DatabaseSyntaxError(
            f"SQL syntax or access violation: {msg}", sql, sqlstate=sqlstate, db_code=native, category="syntax"
        )

    if "divide by zero" in text or "arithmetic overflow" in text or "string or binary data would be truncated" in text:
        return DatabaseDataError(
            f"Data error: {msg}", sql, sqlstate=sqlstate, db_code=native, category="data"
        )
    if "conversion failed" in text or "cannot convert" in text or "data type" in text:
        return DatabaseDataError(
            f"Data type conversion error: {msg}", sql, sqlstate=sqlstate, db_code=native, category="datatype"
        )
    if sqlstate in {"23000"}:
        return DatabaseIntegrityError(
            f"Integrity constraint violation: {msg}", sql, sqlstate=sqlstate, db_code=native, category="integrity"
        )

    if "encoding" in text or "character set" in text or "collation" in text:
        return DatabaseDataError(
            f"Character encoding/collation error: {msg}", sql, sqlstate=sqlstate, db_code=native, category="encoding"
        )

    if native == 130 or "aggregate may not appear" in text:
        return DatabaseSyntaxError(
            f"Invalid aggregate usage: {msg}", sql, sqlstate=sqlstate, db_code=native, category="semantic"
        )

    return DatabaseOperationalError(
        f"Database operation failed: {msg}", sql, sqlstate=sqlstate, db_code=native, category="operational"
    )

# -----------------------------------------------------------------------------
# Service (Unicode-optimized)
# -----------------------------------------------------------------------------
class LanguageAwareSQLServerDatabaseService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "server":   os.getenv("DB_SERVER",   "FIDO2"),
                "database": os.getenv("DB_NAME",     "eHRAntung_DB"),
                "username": os.getenv("DB_USER",     "dbuser"),
                "password": os.getenv("DB_PASSWORD", "MG@5678"),
                "driver":   os.getenv("ODBC_DRIVER", "ODBC Driver 17 for SQL Server"),
                "important_tables": {},
                # Language-aware / Unicode
                "default_collation": os.getenv("DB_COLLATION", "Chinese_Taiwan_Stroke_CI_AS"),
                "enable_unicode_logging": True,
                "query_hints": {
                    "chinese_text_search": "OPTION (RECOMPILE)",
                    "large_result_sets": "OPTION (FAST 100)",
                }
            }
        self.config = config
        self.connection_string = self._build_connection_string()
        self.important_tables: Dict[str, List[str]] = config.get("important_tables", {})
        self.query_hints = config.get("query_hints", {})

        logger.info(
            "DB init rid=%s driver=%s server=%s database=%s user=%s collation=%s",
            rid(),
            self.config.get("driver"),
            self.config.get("server"),
            self.config.get("database"),
            _mask(self.config.get("username")),
            self.config.get("default_collation"),
        )

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------
    def _build_connection_string(self) -> str:
        # Note: AutoTranslate=no to avoid OEM/ANSI conversions
        conn_str = (
            f"DRIVER={{{self.config['driver']}}};"
            f"SERVER={self.config['server']};"
            f"DATABASE={self.config['database']};"
            f"UID={self.config['username']};"
            f"PWD={self.config['password']};"
            "Trusted_Connection=no;"
            "Encrypt=no;"
            "AutoTranslate=no;"
        )
        return conn_str

    def _log_conn_target(self, connect_timeout: Optional[int]) -> None:
        logger.debug(
            "DB connect rid=%s driver=%s server=%s database=%s user=%s timeout=%s unicode_logging=%s",
            rid(),
            self.config.get("driver"),
            self.config.get("server"),
            self.config.get("database"),
            self.config.get("username"),
            connect_timeout,
            self.config.get("enable_unicode_logging", True),
        )

    def get_connection(self, connect_timeout: Optional[int] = None):
        start = time.perf_counter()
        self._log_conn_target(connect_timeout)
        try:
            kwargs: Dict[str, Any] = {}
            if connect_timeout is not None:
                kwargs["timeout"] = int(connect_timeout)

            conn = pyodbc.connect(self.connection_string, **kwargs)

            # Correct Unicode settings for MS ODBC 17/18:
            # - Wide types (WCHAR/WMETADATA) are UTF-16LE from driver
            # - SQL_CHAR safe as UTF-8
            try:
                conn.setdecoding(pyodbc.SQL_CHAR,      encoding='utf-8',    ctype=pyodbc.SQL_CHAR)
                conn.setdecoding(pyodbc.SQL_WCHAR,     encoding='utf-16le', ctype=pyodbc.SQL_WCHAR)
                conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-16le', ctype=pyodbc.SQL_WMETADATA)
                conn.setencoding(encoding='utf-8')  # Python -> SQL parameters
            except Exception as e:
                logger.warning("Unicode configuration failed: %s", e)

            dur_ms = int((time.perf_counter() - start) * 1000)

            dbms_name = dbms_ver = "unknown"
            try:
                dbms_name = conn.getinfo(pyodbc.SQL_DBMS_NAME)
                dbms_ver  = conn.getinfo(pyodbc.SQL_DBMS_VER)
            except Exception:
                pass

            logger.info(
                "DB connect OK rid=%s in %dms dbms=%s ver=%s unicode_ready=true",
                rid(), dur_ms, dbms_name, dbms_ver
            )
            return conn
        except Exception as e:
            dur_ms = int((time.perf_counter() - start) * 1000)
            sqlstate, native, msg = _extract_odbc_error(e)
            logger.error(
                "DB connect FAIL rid=%s in %dms sqlstate=%s code=%s msg=%s",
                rid(), dur_ms, sqlstate, native, msg
            )
            raise _classify_exception(e)

    def test_connection(self, login_timeout: int = 3) -> bool:
        start = time.perf_counter()
        logger.debug("DB ping rid=%s", rid())
        try:
            with self.get_connection(connect_timeout=login_timeout) as conn:
                cur = conn.cursor()
                conn.timeout = max(login_timeout, 1)
                cur.execute("SELECT N'測試' AS test_unicode, 1 AS test_basic")
                result = cur.fetchone()
                if result and len(result) >= 2:
                    unicode_val, basic_val = result[0], result[1]
                    if basic_val != 1 or unicode_val != '測試':
                        logger.warning("Unicode/basic connectivity mismatch: %r / %r", unicode_val, basic_val)
            dur_ms = int((time.perf_counter() - start) * 1000)
            logger.info("DB ping OK rid=%s in %dms unicode_tested=true", rid(), dur_ms)
            return True
        except Exception as e:
            dur_ms = int((time.perf_counter() - start) * 1000)
            sqlstate, native, msg = _extract_odbc_error(e)
            logger.error("DB ping FAIL rid=%s in %dms sqlstate=%s code=%s msg=%s",
                         rid(), dur_ms, sqlstate, native, msg)
            return False

    # -------------------------------------------------------------------------
    # Query Execution (language-aware)
    # -------------------------------------------------------------------------
    def run_select(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
        max_rows: int = 1000,
        *,
        query_timeout: Optional[int] = None,
        max_retries_on_deadlock: int = 1,
        deadlock_backoff_ms: int = 200,
        language_hint: Optional[Literal["zh-tw", "en"]] = None,
        enable_query_hints: bool = True,
    ) -> Tuple[List[Tuple], List[str]]:
        """Execute a SELECT/CTE query safely with language-aware optimizations."""
        try:
            sanitized = self._sanitize_sql(query, language_hint=language_hint)
        except DatabaseQueryError as e:
            logger.warning("SQL rejected rid=%s reason=%s", rid(), str(e))
            raise

        if enable_query_hints and language_hint == "zh-tw":
            sanitized = self._add_chinese_query_hints(sanitized)

        if DB_LOG_SQL:
            _preview = _safe_unicode_repr(sanitized, 1200) if DB_LOG_UNICODE else (sanitized if len(sanitized) < 1200 else sanitized[:1200] + " ...[truncated]")
            logger.debug("SQL rid=%s lang_hint=%s:\n%s", rid(), language_hint, _preview)

        if DB_LOG_PARAMS and params:
            safe_params = tuple(_safe_unicode_repr(p, 100) if isinstance(p, str) else p for p in params) if DB_LOG_UNICODE else params
            logger.debug("SQL params rid=%s: %r", rid(), safe_params)

        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                with self.get_connection() as conn:
                    t1 = time.perf_counter()
                    cur = conn.cursor()
                    if query_timeout is not None:
                        conn.timeout = int(max(0, query_timeout))

                    if params:
                        cur.execute(sanitized, params)
                    else:
                        cur.execute(sanitized)

                    t2 = time.perf_counter()
                    rows = cur.fetchmany(max_rows)
                    columns = [d[0] for d in cur.description] if cur.description else []
                    rows_list = [tuple(r) for r in rows]
                    t3 = time.perf_counter()

                logger.info(
                    "SQL ok rid=%s rows=%d cols=%d connect=%dms exec=%dms fetch=%dms total=%dms lang=%s",
                    rid(),
                    len(rows_list),
                    len(columns),
                    int((t1 - t0) * 1000),
                    int((t2 - t1) * 1000),
                    int((t3 - t2) * 1000),
                    int((t3 - t0) * 1000),
                    language_hint or "unknown",
                )

                if len(rows_list) >= max_rows:
                    logger.debug("SQL rid=%s note=truncated_to_max_rows max_rows=%d", rid(), max_rows)

                return rows_list, columns

            except Exception as raw_exc:
                ex = _classify_exception(raw_exc, sql=sanitized)
                logger.error(
                    "SQL exec FAIL rid=%s cat=%s sqlstate=%s code=%s msg=%s attempt=%d lang=%s",
                    rid(), getattr(ex, "category", None), ex.sqlstate, ex.db_code,
                    _safe_unicode_repr(str(ex), 500), attempt, language_hint or "unknown"
                )
                if isinstance(ex, DeadlockError) and attempt <= max_retries_on_deadlock:
                    time.sleep(deadlock_backoff_ms / 1000.0)
                    continue
                raise ex

    def _add_chinese_query_hints(self, sql: str) -> str:
        if not self.query_hints:
            return sql
        sql_lower = sql.lower()
        needs = any(p in sql_lower for p in (' like ', 'contains', 'freetext', 'charindex', 'patindex'))
        if needs and "chinese_text_search" in self.query_hints and "option" not in sql_lower:
            sql = sql.rstrip(';') + f" {self.query_hints['chinese_text_search']}"
        return sql

    # -------------------------------------------------------------------------
    # Schema helpers
    # -------------------------------------------------------------------------
    def get_schema_tables(self, schema_name: str) -> List[str]:
        try:
            sql = """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """
            rows, _ = self.run_select(sql, params=(schema_name,), language_hint="en")
            return [r[0] for r in rows]
        except Exception as e:
            logger.error("Schema tables FAIL rid=%s schema=%s err=%s", rid(), schema_name, repr(e))
            return []

    def get_table_columns_enhanced(self, schema_name: str, table_name: str,
                                   include_unicode_info: bool = True) -> List[Dict[str, Any]]:
        try:
            if include_unicode_info:
                sql = """
                    SELECT 
                        c.COLUMN_NAME, 
                        c.DATA_TYPE, 
                        c.IS_NULLABLE,
                        c.CHARACTER_MAXIMUM_LENGTH,
                        c.COLLATION_NAME,
                        CASE 
                            WHEN c.DATA_TYPE IN ('nvarchar', 'nchar', 'ntext') THEN 'unicode'
                            WHEN c.DATA_TYPE IN ('varchar', 'char', 'text') THEN 'ansi'
                            ELSE 'non_text'
                        END as text_type
                    FROM INFORMATION_SCHEMA.COLUMNS c
                    WHERE c.TABLE_SCHEMA = ? AND c.TABLE_NAME = ?
                    ORDER BY c.ORDINAL_POSITION
                """
                rows, _ = self.run_select(sql, params=(schema_name, table_name), language_hint="en")
                return [
                    {
                        "name": r[0],
                        "type": r[1],
                        "nullable": (r[2] == "YES"),
                        "max_length": r[3],
                        "collation": r[4],
                        "text_type": r[5],
                        "supports_chinese": r[5] == 'unicode' or (r[4] and 'chinese' in str(r[4]).lower())
                    }
                    for r in rows
                ]
            else:
                return self.get_table_columns(schema_name, table_name)
        except Exception as e:
            logger.error(
                "Enhanced table columns FAIL rid=%s schema=%s table=%s err=%s",
                rid(), schema_name, table_name, repr(e)
            )
            return self.get_table_columns(schema_name, table_name)

    def get_table_columns(self, schema_name: str, table_name: str) -> List[Dict[str, Any]]:
        try:
            sql = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """
            rows, _ = self.run_select(sql, params=(schema_name, table_name), language_hint="en")
            return [{"name": c, "type": t, "nullable": (n == "YES")} for (c, t, n) in rows]
        except Exception as e:
            logger.error(
                "Table columns FAIL rid=%s schema=%s table=%s err=%s",
                rid(), schema_name, table_name, repr(e)
            )
            return []

    def get_compact_schema_for(
        self,
        table_fullnames: List[str],
        max_columns_per_table: int = 12,
        include_unicode_info: bool = False
    ) -> str:
        lines: List[str] = []
        for full in table_fullnames:
            if "." not in full:
                continue
            schema, table = full.split(".", 1)
            if include_unicode_info:
                cols = self.get_table_columns_enhanced(schema, table, True)[:max_columns_per_table]
            else:
                cols = self.get_table_columns(schema, table)[:max_columns_per_table]

            if not cols:
                continue

            colbits = []
            for c in cols:
                entry = f"{c['name']}:{c['type']}"
                if c.get("nullable"):
                    entry += " NULL"
                if include_unicode_info and c.get("supports_chinese"):
                    entry += " 中文"
                colbits.append(entry)
            lines.append(f"{schema}.{table}({', '.join(colbits)})")
        return "\n".join(lines) if lines else "No relevant tables found"

    # -------------------------------------------------------------------------
    # Sanitization
    # -------------------------------------------------------------------------
    def _sanitize_sql(self, query: str, language_hint: Optional[str] = None) -> str:
        if not isinstance(query, str):
            raise DatabaseQueryError("Query must be a string")

        original_len = len(query)

        m = _CODE_FENCE_RE.search(query)
        if m:
            query = m.group(1)

        if language_hint == "zh-tw":
            for pattern in _CHINESE_COMMENT_PATTERNS:
                query = pattern.sub("", query)

        query = _LINE_COMMENT_RE.sub("", query)
        query = _BLOCK_COMMENT_RE.sub("", query)
        query = query.strip()
        if not query:
            raise DatabaseQueryError("Empty query after cleaning")

        DECLARE_BLOCK_RE = re.compile(
            r"""^\s*(?:(?:(?:DECLARE|SET)[\s\S]*?;\s*)+)\s*""",
            re.IGNORECASE | re.VERBOSE,
        )
        leading_block = DECLARE_BLOCK_RE.match(query)
        declare_txt = ""
        if leading_block:
            declare_txt = leading_block.group(0)
            query_main = query[leading_block.end():].lstrip()
        else:
            query_main = query

        if not query_main:
            raise DatabaseQueryError("Only DECLARE/SET block found; missing main SELECT/WITH")

        OPTION_TAIL_RE = re.compile(r"\s+OPTION\s*\([^)]*\)\s*;?\s*$", re.IGNORECASE)
        option_tail = OPTION_TAIL_RE.search(query_main)
        query_core = OPTION_TAIL_RE.sub("", query_main).strip()
        if not query_core:
            raise DatabaseQueryError("Query body empty after removing OPTION() tail")

        if not (_SELECT_PREFIX_RE.match(query_core) or _CTE_PREFIX_RE.match(query_core)):
            raise DatabaseQueryError("Only SELECT and WITH queries are allowed")

        body = query_core[:-1] if query_core.endswith(";") else query_core
        if ";" in body:
            raise DatabaseQueryError("Multiple statements are not allowed")

        bl_hit = _BLOCKLIST_RE.search(query)
        if bl_hit:
            tok = bl_hit.group(0)
            raise DatabaseQueryError(f"Disallowed SQL token detected: {tok}")

        sanitized = f"{declare_txt}{query_core}"
        if option_tail:
            sanitized += f" {option_tail.group(0).strip()}"

        if DB_LOG_SQL:
            logger.debug(
                "Sanitized SQL rid=%s original_len=%d final_len=%d has_declare=%s has_option=%s lang_hint=%s",
                rid(), original_len, len(sanitized), bool(declare_txt), bool(option_tail), language_hint
            )
        return sanitized

    # -------------------------------------------------------------------------
    # Backward compatibility helpers
    # -------------------------------------------------------------------------
    def get_schema_prompt(
        self,
        schema: str = "dbo",
        tables: Optional[List[str]] = None,
        max_columns_per_table: int = 12
    ) -> str:
        if tables:
            fulls = [t if "." in t else f"{schema}.{t}" for t in tables]
            return self.get_compact_schema_for(fulls, max_columns_per_table)

        if self.important_tables.get(schema):
            table_list = [f"{schema}.{t}" for t in self.important_tables[schema]]
        else:
            discovered = self.get_schema_tables(schema)[:50]
            table_list = [f"{schema}.{t}" for t in discovered]

        return self.get_compact_schema_for(table_list, max_columns_per_table)

    def get_health_status(self) -> Dict[str, Any]:
        try:
            connected = self.test_connection()
            return {
                "database_connected": connected,
                "connection_string_configured": bool(self.connection_string),
                "important_tables_configured": len(self.important_tables) > 0,
                "unicode_support_enabled": self.config.get("enable_unicode_logging", True),
                "default_collation": self.config.get("default_collation"),
                "language_aware_features": True,
                "query_hints_available": len(self.query_hints) > 0,
            }
        except Exception as e:
            logger.error("Health status FAIL rid=%s err=%s", rid(), repr(e))
            return {"database_connected": False, "error": str(e), "language_aware_features": False}

    # -------------------------------------------------------------------------
    # Language-specific helpers
    # -------------------------------------------------------------------------
    def get_chinese_text_columns(self, schema_name: str, table_name: str) -> List[str]:
        try:
            cols = self.get_table_columns_enhanced(schema_name, table_name, True)
            return [c["name"] for c in cols if c.get("supports_chinese", False)]
        except Exception as e:
            logger.warning("Failed to get Chinese text columns: %s", e)
            return []

    def optimize_query_for_language(self, query: str, language: Literal["zh-tw", "en"]) -> str:
        if language == "zh-tw":
            query = self._add_collation_hints(query)
            query = self._add_chinese_query_hints(query)
        return query

    def _add_collation_hints(self, query: str) -> str:
        collation = self.config.get("default_collation")
        if not collation or "chinese" not in str(collation).lower():
            return query
        patterns = [
            (r'(\w+\s*=\s*N?\'[^\']*[\u4e00-\u9fff][^\']*\')', rf'\1 COLLATE {collation}'),
            (r'(\w+\s+LIKE\s+N?\'[^\']*[\u4e00-\u9fff][^\']*\')', rf'\1 COLLATE {collation}'),
        ]
        for pattern, replacement in patterns:
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        return query

    def test_unicode_support(self) -> Dict[str, Any]:
        test_results = {
            "unicode_storage": False,
            "unicode_retrieval": False,
            "chinese_collation": False,
            "error": None
        }
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                test_data = '測試中文數據'
                cur.execute("SELECT ? AS test_value", (test_data,))
                result = cur.fetchone()
                if result and result[0] == test_data:
                    test_results["unicode_storage"] = True
                    test_results["unicode_retrieval"] = True

                collation = self.config.get("default_collation")
                if collation:
                    try:
                        cur.execute(f"SELECT N'測試' COLLATE {collation} AS test_collation")
                        result = cur.fetchone()
                        if result and result[0] == '測試':
                            test_results["chinese_collation"] = True
                    except Exception:
                        pass
        except Exception as e:
            test_results["error"] = str(e)
            logger.warning("Unicode support test failed: %s", e)
        return test_results

    # -------------------------------------------------------------------------
    # Diagnostics (optional)
    # -------------------------------------------------------------------------
    def probe_text_sample(self, schema="dbo", table="BIPSNACCOUNTSP", col="TRUENAME", top=5):
        sql = f"""
          SELECT TOP ({int(top)})
            CAST({col} AS NVARCHAR(200)) AS val,
            SQL_VARIANT_PROPERTY({col}, 'BaseType') AS base_type,
            DATALENGTH({col}) AS bytes
          FROM {schema}.{table}
          WHERE {col} IS NOT NULL AND LEN({col}) > 0
        """
        rows, _ = self.run_select(sql, language_hint="zh-tw")
        for r in rows:
            logger.info("probe %s.%s.%s -> val=%s base=%s bytes=%s",
                        schema, table, col, _safe_unicode_repr(r[0], 50), r[1], r[2])
        return rows


# Backward compatibility alias
SQLServerDatabaseService = LanguageAwareSQLServerDatabaseService

# -----------------------------------------------------------------------------
# CLI Smoke Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=os.getenv("DB_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    set_request_id("cli")

    print("== DB smoke test: Unicode support ==")
    try:
        svc = LanguageAwareSQLServerDatabaseService()
        print(
            f"Server={svc.config.get('server')} "
            f"Database={svc.config.get('database')} "
            f"Driver={svc.config.get('driver')} "
            f"Collation={svc.config.get('default_collation')}"
        )

        print("Testing basic connection...", end=" ")
        if not svc.test_connection():
            print("FAILED")
            sys.exit(1)
        print("OK")

        print("Testing Unicode support...", end=" ")
        unicode_test = svc.test_unicode_support()
        if unicode_test.get("unicode_storage") and unicode_test.get("unicode_retrieval"):
            print("OK")
            if unicode_test.get("chinese_collation"):
                print("  Chinese collation: OK")
            else:
                print("  Chinese collation: Not configured (optional)")
        else:
            print("PARTIAL - some Unicode features may not work")
            print(f"  Error: {unicode_test.get('error', 'Unknown')}")

        # Optional: probe sample names
        try:
            print("\nProbing sample TRUENAME values...")
            svc.probe_text_sample()
        except Exception as e:
            print(f"Probe failed: {e}")

        print("\nDatabase service ready for bilingual operations.")

    except DatabaseQueryError as e:
        print(f"[ERROR] {e} | sqlstate={e.sqlstate} code={e.db_code} cat={e.category}")
        sys.exit(2)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(3)
