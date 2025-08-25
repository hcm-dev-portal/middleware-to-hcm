from typing import Any, Dict, List
from fastapi import HTTPException

from app.services.db_service import SQLServerDatabaseService, DatabaseQueryError
from app.services.nlp_service import AWSNLPService


class DBController:
    def __init__(self, db_service: SQLServerDatabaseService, nlp_service: AWSNLPService):
        self.db = db_service
        self.nlp = nlp_service

    async def get_health_status(self) -> Dict[str, Any]:
        return self.db.get_health_status()

    # ------- internal helpers -------

    def _schema_text_for(self, schema: str, max_cols: int = 12) -> str:
        """Schema-scoped compact prompt; fall back to 'important tables' if empty."""
        tables = self.db.get_schema_tables(schema) or []
        if not tables:
            return self.db.get_schema_prompt(max_columns_per_table=max_cols)

        lines: List[str] = [f"DATABASE SCHEMA (Schema: {schema})"]
        for t in tables:
            cols = self.db.get_table_columns(schema, t)[:max_cols]
            if not cols:
                continue
            colbits = []
            for c in cols:
                entry = f"{c['name']}:{c['type']}"
                if c.get("nullable"):
                    entry += " NULL"
                colbits.append(entry)
            lines.append(f"{schema}.{t}({', '.join(colbits)})")
        return "\n".join(lines)

    @staticmethod
    def _summary_en(rows: int, cols: int) -> str:
        return f"Retrieved {rows} rows and {cols} columns."

    # ------- public -------

    async def test_query(self, schema: str, question: str) -> Dict[str, Any]:
        if not question or not isinstance(question, str):
            raise HTTPException(status_code=400, detail="question is required")

        # Allow raw SELECT/CTE for dev
        if question.strip().lower().startswith(("select", "with")):
            try:
                data, headers = self.db.run_select(question)
                return {
                    "success": True,
                    "executed_as": "raw_sql",
                    "query_language": "en-US",
                    "row_count": len(data),
                    "columns": headers,
                    "rows": data,
                    "summary": self._summary_en(len(data), len(headers)),
                }
            except DatabaseQueryError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Natural language → SQL
        try:
            nlp_info = self.nlp.process_user_query(question)  # detect + translate
            user_lang = nlp_info["detected_language"]  # en-US | zh-TW | zh-CN

            schema_txt = self._schema_text_for(schema)
            sql = self.nlp.generate_sql(nlp_info["english_text"], schema_txt)

            # Guardrail: ensure LLM actually produced a SELECT/CTE
            if not sql.lower().startswith(("select", "with")):
                raise HTTPException(status_code=400, detail="Model did not produce a valid SELECT statement.")

            data, headers = self.db.run_select(sql)

            # Short summary back to user language
            summary_en = self._summary_en(len(data), len(headers))
            summary_localized = self.nlp.translate_from_english(summary_en, user_lang)

            return {
                "success": True,
                "executed_as": "nl_sql",
                "query_language": user_lang,
                "generated_sql": sql,
                "row_count": len(data),
                "columns": headers,  # keep DB names intact
                "rows": data,
                "summary": summary_localized,
            }

        except DatabaseQueryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed: {e}")
