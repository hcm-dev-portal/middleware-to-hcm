# backend/app/services/db_service.py

import os
import re
import time
import uuid
import pyodbc
import logging
import contextvars
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Logging setup
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

DB_LOG_SQL = os.getenv("DB_LOG_SQL", "0") == "1"
DB_LOG_PARAMS = os.getenv("DB_LOG_PARAMS", "0") == "1"

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class DatabaseQueryError(Exception):
    """Custom exception for database query failures that carries the SQL"""
    def __init__(self, message: str, sql: str = ""):
        super().__init__(message)
        self.sql = sql

# -----------------------------------------------------------------------------
# SQL Validation Patterns
# -----------------------------------------------------------------------------
_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_LINE_COMMENT_RE = re.compile(r"(--|#).*?$", re.M)
_BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/", re.M)
_SELECT_PREFIX_RE = re.compile(r"^\s*select\b", re.I)
_CTE_PREFIX_RE = re.compile(r"^\s*with\b", re.I)
_BLOCKLIST_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|MERGE|EXEC|EXECUTE|TRUNCATE|GRANT|REVOKE|DENY|BACKUP|RESTORE|"
    r"xp_[A-Za-z_]+|sp_[A-Za-z_]+|OPENROWSET|OPENQUERY|BULK\s+INSERT|INTO)\b",
    re.I,
)

# -----------------------------------------------------------------------------
# Service
# -----------------------------------------------------------------------------
class SQLServerDatabaseService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "server":   os.getenv("DB_SERVER",   "FIDO2"),
                "database": os.getenv("DB_NAME",     "eHRAntung_DB"),
                "username": os.getenv("DB_USER",     "dbuser"),
                "password": os.getenv("DB_PASSWORD", "MG@5678"),
                "driver":   os.getenv("ODBC_DRIVER", "ODBC Driver 17 for SQL Server"),
                "important_tables": {}
            }
        self.config = config
        self.connection_string = self._build_connection_string()
        self.important_tables: Dict[str, List[str]] = config.get("important_tables", {})

        # Log sanitized configuration once at init
        logger.info(
            "DB init rid=%s driver=%s server=%s database=%s user=%s",
            rid(),
            self.config.get("driver"),
            self.config.get("server"),
            self.config.get("database"),
            self.config.get("username"),
        )

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------
    def _build_connection_string(self) -> str:
        return (
            f"DRIVER={{{self.config['driver']}}};"
            f"SERVER={self.config['server']};"
            f"DATABASE={self.config['database']};"
            f"UID={self.config['username']};"
            f"PWD={self.config['password']};"
            "Trusted_Connection=no;"
            "Encrypt=no;"
        )

    def _log_conn_target(self, connect_timeout: Optional[int]) -> None:
        logger.debug(
            "DB connect rid=%s driver=%s server=%s database=%s user=%s timeout=%s",
            rid(),
            self.config.get("driver"),
            self.config.get("server"),
            self.config.get("database"),
            self.config.get("username"),
            connect_timeout,
        )

    def get_connection(self, connect_timeout: Optional[int] = None):
        start = time.perf_counter()
        self._log_conn_target(connect_timeout)
        try:
            kwargs: Dict[str, Any] = {}
            if connect_timeout is not None:
                # pyodbc uses 'timeout' for login/connect timeout (seconds)
                kwargs["timeout"] = int(connect_timeout)
            conn = pyodbc.connect(self.connection_string, **kwargs)
            dur_ms = int((time.perf_counter() - start) * 1000)

            # Try to print DBMS info for debugging (if available)
            dbms_name = dbms_ver = "unknown"
            try:
                dbms_name = conn.getinfo(pyodbc.SQL_DBMS_NAME)  # type: ignore[attr-defined]
                dbms_ver  = conn.getinfo(pyodbc.SQL_DBMS_VER)   # type: ignore[attr-defined]
            except Exception:
                pass

            logger.info(
                "DB connect OK rid=%s in %dms dbms=%s ver=%s",
                rid(), dur_ms, dbms_name, dbms_ver
            )
            return conn
        except Exception as e:
            dur_ms = int((time.perf_counter() - start) * 1000)
            logger.error("DB connect FAIL rid=%s in %dms err=%s", rid(), dur_ms, repr(e))
            raise DatabaseQueryError(f"Failed to connect to database: {e}")

    def test_connection(self, login_timeout: int = 3) -> bool:
        start = time.perf_counter()
        logger.debug("DB ping rid=%s", rid())
        try:
            with self.get_connection(connect_timeout=login_timeout) as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
            dur_ms = int((time.perf_counter() - start) * 1000)
            logger.info("DB ping OK rid=%s in %dms", rid(), dur_ms)
            return True
        except Exception as e:
            dur_ms = int((time.perf_counter() - start) * 1000)
            logger.error("DB ping FAIL rid=%s in %dms err=%s", rid(), dur_ms, repr(e))
            return False

    # -------------------------------------------------------------------------
    # Query Execution (Select-only)
    # -------------------------------------------------------------------------
    def run_select(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
        max_rows: int = 1000,
    ) -> Tuple[List[Tuple], List[str]]:
        """Execute a SELECT/CTE query safely and return up to max_rows"""
        # Sanitization (with diagnostics)
        try:
            sanitized = self._sanitize_sql(query)
        except DatabaseQueryError as e:
            logger.warning("SQL rejected rid=%s reason=%s", rid(), str(e))
            raise

        # Optional SQL / params logging
        if DB_LOG_SQL:
            _preview = sanitized if len(sanitized) < 1200 else sanitized[:1200] + " ...[truncated]"
            logger.debug("SQL rid=%s:\n%s", rid(), _preview)
        if DB_LOG_PARAMS and params:
            logger.debug("SQL params rid=%s: %r", rid(), params)

        # Connect / execute / fetch timings
        t0 = time.perf_counter()
        with self.get_connection() as conn:
            t1 = time.perf_counter()
            cur = conn.cursor()
            try:
                if params:
                    cur.execute(sanitized, params)
                else:
                    cur.execute(sanitized)
            except Exception as e:
                logger.error("SQL exec FAIL rid=%s err=%s", rid(), repr(e))
                raise DatabaseQueryError(f"Query execution failed: {e}", sanitized)

            t2 = time.perf_counter()
            rows = cur.fetchmany(max_rows)
            t3 = time.perf_counter()

        columns = [d[0] for d in cur.description] if cur.description else []
        rows_list = [tuple(r) for r in rows]

        logger.info(
            "SQL ok rid=%s rows=%d cols=%d connect=%dms exec=%dms fetch=%dms total=%dms",
            rid(),
            len(rows_list),
            len(columns),
            int((t1 - t0) * 1000),
            int((t2 - t1) * 1000),
            int((t3 - t2) * 1000),
            int((t3 - t0) * 1000),
        )

        if len(rows_list) >= max_rows:
            logger.debug("SQL rid=%s note=truncated_to_max_rows max_rows=%d", rid(), max_rows)

        return rows_list, columns

    # -------------------------------------------------------------------------
    # Introspection helpers
    # -------------------------------------------------------------------------
    def get_schema_tables(self, schema_name: str) -> List[str]:
        try:
            sql = """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """
            rows, _ = self.run_select(sql, params=(schema_name,))
            return [r[0] for r in rows]
        except Exception as e:
            logger.error("Schema tables FAIL rid=%s schema=%s err=%s", rid(), schema_name, repr(e))
            return []

    def get_table_columns(self, schema_name: str, table_name: str) -> List[Dict[str, Any]]:
        try:
            sql = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """
            rows, _ = self.run_select(sql, params=(schema_name, table_name))
            return [{"name": c, "type": t, "nullable": (n == "YES")} for (c, t, n) in rows]
        except Exception as e:
            logger.error(
                "Table columns FAIL rid=%s schema=%s table=%s err=%s",
                rid(), schema_name, table_name, repr(e)
            )
            return []

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

    def get_compact_schema_for(
        self,
        table_fullnames: List[str],
        max_columns_per_table: int = 12
    ) -> str:
        lines: List[str] = []
        for full in table_fullnames:
            if "." not in full:
                continue
            schema, table = full.split(".", 1)
            cols = self.get_table_columns(schema, table)[:max_columns_per_table]
            if not cols:
                continue
            colbits = []
            for c in cols:
                entry = f"{c['name']}:{c['type']}"
                if c.get("nullable"):
                    entry += " NULL"
                colbits.append(entry)
            lines.append(f"{schema}.{table}({', '.join(colbits)})")
        return "\n".join(lines) if lines else "No relevant tables found"

    def get_health_status(self) -> Dict[str, Any]:
        try:
            connected = self.test_connection()
            return {
                "database_connected": connected,
                "connection_string_configured": bool(self.connection_string),
                "important_tables_configured": len(self.important_tables) > 0,
            }
        except Exception as e:
            logger.error("Health status FAIL rid=%s err=%s", rid(), repr(e))
            return {"database_connected": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # Sanitization (with diagnostics)
    # -------------------------------------------------------------------------
    def _sanitize_sql(self, query: str) -> str:
        if not isinstance(query, str):
            raise DatabaseQueryError("Query must be a string")

        original_len = len(query)

        m = _CODE_FENCE_RE.search(query)
        if m:
            query = m.group(1)

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
                "Sanitized SQL rid=%s original_len=%d final_len=%d has_declare=%s has_option=%s",
                rid(), original_len, len(sanitized), bool(declare_txt), bool(option_tail)
            )
        return sanitized


# -----------------------------------------------------------------------------
# CLI Smoke Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Database connection test
    """
    import sys

    logging.basicConfig(
        level=os.getenv("DB_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | rid=%(rid)s | %(message)s"
        .replace("%(rid)s", "%(message)s")  # fallback if LogRecord has no 'rid'
    )

    # Ensure a RID for CLI
    set_request_id("cli")

    print("== DB smoke test: list tables in eHRAntung_DB ==")
    try:
        svc = SQLServerDatabaseService()
        print(
            f"Server={svc.config.get('server')} "
            f"Database={svc.config.get('database')} "
            f"Driver={svc.config.get('driver')}"
        )

        print("Testing connection...", end=" ")
        if not svc.test_connection():
            print("FAILED")
            sys.exit(1)
        print("OK")

        sql = """
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM eHRAntung_DB.INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        rows, _ = svc.run_select(sql)

        if not rows:
            print("\nNo base tables found in eHRAntung_DB.")
            sys.exit(0)

        print(f"\nTables in eHRAntung_DB (count: {len(rows)}):")
        current_schema = None
        for schema, table in rows:
            if schema != current_schema:
                current_schema = schema
                print(f"\n[{schema}]")
            print(f" - {table}")

    except DatabaseQueryError as e:
        print(f"[ERROR] {e} | SQL={getattr(e, 'sql', '')}")
        sys.exit(2)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(3)
