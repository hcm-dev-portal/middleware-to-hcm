# backend/app/services/db_service.py

import re
import pyodbc
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DatabaseQueryError(Exception):
    """Custom exception for database query failures that carries the SQL"""
    def __init__(self, message: str, sql: str = ""):
        super().__init__(message)
        self.sql = sql

# --- SQL Validation Patterns ---
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

class SQLServerDatabaseService:
    """
    Lean, read-only SQL Server service.
    - Preserves your exact connection details
    - SELECT/CTE-only queries
    - Parameterized introspection
    - Hard row cap via fetchmany()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "server": "FIDO2",
                "database": "eHRAntung_DB",
                "username": "dbuser",
                "password": "MG@5678",
                "driver": "ODBC Driver 17 for SQL Server",

                # Key tables based on the actual eHRAntung_DB schema
                "important_tables": {
                    "dbo": [
                        # Core HR/Personnel tables
                        "PSNACCOUNT", "PSNBASEINFO", "PSNADJUST", "PSNCONTACTINFO",
                        "PSNEDUCATION", "PSNWORKEXPERIENCE", "PSNFAMILYINFO",
                        
                        # Organization structure
                        "ORGSTDSTRUCT", "ORGSTDUNITATTR", "ORGDIMENSIONSTRUCT",
                        "ORGLEGAENTITY", "ORGDIMENSIONUNIT",
                        
                        # Attendance & Time management
                        "ATDEMPLOYEECALENDAR", "ATDOVERTIME", "ATDLEAVEDATA", "ATDTIMECARDDATA",
                        "ATDATTENDANCEDATA", "ATDLATEEARLY", "ATDNOTIMECARD", "ATDGENERAL",
                        
                        # Payroll & Benefits
                        "PAYPAYROLLSUBJECT", "PAYSTAFFBASEINFO", "PAYINSUREDATASUM",
                        "PAYINSUREDATADETAIL", "PAYINCOMETAX", "PAYCALCULATEMAINLOGIC",
                        
                        # Taiwan-specific payroll
                        "PAY_TW_BASEINFO", "PAY_TW_INSURANCEGRADE", "PAY_TW_PACKAGESTAFF",
                        "PAY_TW_TAXINFO", "PAY_TW_INSURECALRESULT",
                        
                        # Job codes & Competencies
                        "CPCJOBCODE", "CPCJOBCODETODEPT", "CPCJOBCODETOPOSITION",
                        "CPCJOBBOOK", "CPCCOMPETENCE", "CPCPOSIITEM",
                        
                        # Learning & Training
                        "LMS_COURSE", "LMS_CLASS", "LMS_PARTICIPANT", "LMS_SCORE",
                        "LMS_TEACHER", "LMS_TRAINPLAN", "LMS_BUDGET_TOTAL",
                        
                        # Workflow forms (key business processes)
                        "FLOWER_LEAVE_APPLY_FORM_MAIN", "FLOWER_OVERTIME_FORM_MAIN",
                        "FLOWER_RECRUIT_FORM_MAIN", "FLOWER_ADJUST_APPLY_MAIN",
                        
                        # Business Intelligence & Reporting
                        "BIEMPLOYEETURNOVERFACT", "BIMANAGEMENTINDICATOR", "BIPARAMETER",
                        
                        # System configuration
                        "SYSPUBLICCODEITEM", "SYSPUBLICCODETYPE", "SYSSETTING"
                    ]
                }
            }
        self.config = config
        self.connection_string = self._build_connection_string()
        self.important_tables: Dict[str, List[str]] = config.get("important_tables", {})

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

    def get_connection(self):
        try:
            return pyodbc.connect(self.connection_string, timeout=30)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise DatabaseQueryError(f"Failed to connect to database: {e}")

    def test_connection(self) -> bool:
        try: 
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
                return True 
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    # ---- Query Execution (Select-only) ----
    def run_select(
        self, 
        query: str, 
        params: Optional[Tuple[Any, ...]] = None,
        max_rows: int = 1000, 
    ) -> Tuple[List[Tuple], List[str]]:
        """Execute a SELECT/CTE query safely and return up to max_rows"""
        try: 
            sql = self._sanitize_sql(query)

            with self.get_connection() as conn:
                cur = conn.cursor()
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)

                columns = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchmany(max_rows)
                rows_list = [tuple(r) for r in rows]
                logger.info(f"Query executed: {len(rows_list)} row(s)")
                return rows_list, columns

        except DatabaseQueryError:
            raise 
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseQueryError(f"Query execution failed: {e}", query)

    # --- Introspection ---
    def get_schema_tables(self, schema_name: str) -> List[str]:
        """List base tables under a schema."""
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
            logger.error(f"Error getting schema tables: {e}")
            return []

    def get_table_columns(self, schema_name: str, table_name: str) -> List[Dict[str, Any]]:
        """List columns of a table with basic metadata."""
        try:
            sql = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """
            rows, _ = self.run_select(sql, params=(schema_name, table_name))
            return [
                {"name": c, "type": t, "nullable": (n == "YES")}
                for (c, t, n) in rows
            ]
        except Exception as e:
            logger.error(f"Error getting table columns: {e}")
            return []

    def get_schema_prompt(self, max_columns_per_table: int = 12) -> str:
        """
        Build a compact text schema (English) for the NLP layer to consume.
        """
        lines: List[str] = ["DATABASE SCHEMA (Key Tables Only):"]
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                for schema_name, tables in self.important_tables.items():
                    lines.append(f"\n-- Schema: {schema_name} --")
                    for table in tables:
                        try:
                            cur.execute(
                                """
                                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                                FROM INFORMATION_SCHEMA.COLUMNS
                                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                                ORDER BY ORDINAL_POSITION
                                """,
                                (schema_name, table),
                            )
                            cols = []
                            for (col, typ, nullable) in cur.fetchall()[:max_columns_per_table]:
                                entry = f"{col}:{typ}"
                                if nullable == "YES":
                                    entry += " NULL"
                                cols.append(entry)
                            if cols:
                                lines.append(f"{schema_name}.{table}({', '.join(cols)})")
                        except Exception as inner:
                            logger.warning(f"Could not get columns for {schema_name}.{table}: {inner}")

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error building schema prompt: {e}")
            return "Error retrieving schema information"

    def get_health_status(self) -> Dict[str, Any]:
        try:
            connected = self.test_connection()
            return {
                "database_connected": connected,
                "connection_string_configured": bool(self.connection_string),
                "important_tables_configured": len(self.important_tables) > 0,
            }
        except Exception as e:
            return {"database_connected": False, "error": str(e)}

    # --- Internal sanitization ----
    def _sanitize_sql(self, query: str) -> str:
        """Allow only single-statement SELECT/CTE queries; strip comments/fences."""
        if not isinstance(query, str):
            raise DatabaseQueryError("Query must be a string")

        # Fenced code
        m = _CODE_FENCE_RE.search(query)
        if m:
            query = m.group(1)

        # Remove comments
        query = _LINE_COMMENT_RE.sub("", query)
        query = _BLOCK_COMMENT_RE.sub("", query)

        # Trim & collapse whitespace edges
        query = query.strip()

        if not query:
            raise DatabaseQueryError("Empty query after cleaning")

        # Only SELECT or WITH
        if not (_SELECT_PREFIX_RE.match(query) or _CTE_PREFIX_RE.match(query)):
            raise DatabaseQueryError("Only SELECT and WITH queries are allowed")

        # Disallow multiple statements (only allow an optional trailing ';')
        body = query[:-1] if query.endswith(";") else query
        if ";" in body:
            raise DatabaseQueryError("Multiple statements are not allowed")

        # Blocklist dangerous tokens
        if _BLOCKLIST_RE.search(query):
            raise DatabaseQueryError("Disallowed SQL token detected")

        return query


if __name__ == "__main__":
    """
    Database connection test
    """
    import sys

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