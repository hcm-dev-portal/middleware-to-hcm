# backend/app/api/router.py
from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.db_service import SQLServerDatabaseService
from app.services.metadata_harvester import harvest_metadata, save_metadata, load_metadata
from app.services.indexer import build_index, save_index, load_index
from app.services.retriever import intent_from_question, parse_table_ref, search_tables, make_sql_for_intent
from app.core.paths import METADATA_PATH, INDEX_PATH

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton
db = SQLServerDatabaseService()

class RebuildBody(BaseModel):
    schemas: Optional[List[str]] = Field(default=None, description="e.g. ['dbo', 'gbpm']")

class QueryBody(BaseModel):
    question: str
    schema_name: Optional[str] = Field(default="dbo", description="Target schema filter")

@router.get("/api/health")
async def health() -> Dict[str, Any]:
    """Enhanced health check"""
    meta_exists = METADATA_PATH.exists()
    idx_exists = INDEX_PATH.exists()
    
    # Test database connection
    db_connected = db.test_connection()
    
    return {
        "database_connection": db_connected,
        "metadata_present": meta_exists,
        "index_present": idx_exists,
        "ready_for_queries": db_connected and idx_exists,
        "metadata_path": str(METADATA_PATH),
        "index_path": str(INDEX_PATH)
    }

@router.post("/api/schema/rebuild")
async def schema_rebuild(body: Optional[RebuildBody] = None) -> Dict[str, Any]:
    """Rebuild schema metadata and search index"""
    schemas = (body.schemas if body and body.schemas else ["dbo"])
    logger.info(f"Rebuilding schema metadata & index for schemas={schemas}")
    
    try:
        # 1) Harvest metadata
        logger.info("Harvesting metadata from database...")
        metadata = harvest_metadata(db, schemas=schemas, max_columns_per_table=200)
        meta_path = save_metadata(metadata)
        logger.info(f"Metadata saved to: {meta_path}")
        
        # 2) Build search index
        logger.info("Building search index...")
        index = build_index(metadata)
        idx_path = save_index(index)
        logger.info(f"Index saved to: {idx_path}")

        # Count tables
        total_tables = sum(len(v) for v in metadata.get("schemas", {}).values())
        logger.info(f"Successfully indexed {total_tables} tables")

        return {
            "ok": True,
            "metadata_path": str(meta_path),
            "index_path": str(idx_path),
            "database": metadata.get("database"),
            "schemas": schemas,
            "tables_indexed": total_tables,
            "message": f"Successfully rebuilt schema index for {total_tables} tables"
        }
    except Exception as e:
        logger.error(f"Schema rebuild failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Schema rebuild failed: {str(e)}")

@router.post("/api/query")
async def run_query(req: Request) -> Dict[str, Any]:
    """
    Enhanced query endpoint with basic retrieval functionality.
    """
    question: Optional[str] = None
    schema_name: str = "dbo"

    # Try JSON first
    try:
        data = await req.json()
        if isinstance(data, dict):
            question = data.get("question") or data.get("query")
            schema_name = data.get("schema_name") or data.get("schema") or "dbo"
    except Exception:
        pass

    # Try form data if JSON failed
    if not question:
        try:
            form = await req.form()
            question = form.get("question")
            schema_name = form.get("schema_name") or form.get("schema") or "dbo"
        except Exception:
            pass

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' parameter")

    # Ensure index exists
    index = load_index()
    if not index:
        logger.warning("No schema index found. /api/schema/rebuild must run first.")
        raise HTTPException(
            status_code=400, 
            detail="Schema index not found. Please call /api/schema/rebuild first to initialize the system."
        )

    try:
        logger.info(f"Processing query: {question}")
        
        # Basic intent analysis
        intent = intent_from_question(question)
        parsed = parse_table_ref(question)
        table = parsed[0] if parsed else None
        schema_guess = (parsed[1] if parsed else None) or schema_name or "dbo"

        logger.info(f"Intent: {intent}, Table: {table}, Schema: {schema_guess}")

        # Try to generate SQL for simple intents
        sql = make_sql_for_intent(intent, schema_guess, table)
        logger.info(f"Generated SQL: {sql}")

        # Execute SQL if available
        rows, columns = [], []
        execution_error = None
        if sql:
            try:
                logger.info(f"Executing SQL: {sql}")
                rows, columns = db.run_select(sql)
                logger.info(f"Query executed successfully: {len(rows)} rows returned")
            except Exception as e:
                execution_error = str(e)
                logger.error(f"SQL execution failed: {e}")
        else:
            logger.info("No SQL generated - this may be a complex query needing AI processing")

        # Get table suggestions
        candidates = search_tables(index, question, schema_filter=schema_name, top_k=10)
        ret_list = [{"table": t, "score": round(s, 3)} for t, s in candidates]
        logger.info(f"Found {len(candidates)} relevant tables")

        response = {
            "message": "OK",
            "intent": intent,
            "schema": schema_guess,
            "question": question,
            "query": sql or None,
            "columns": columns,
            "results": [list(r) for r in rows] if rows else [],
            "retrieval": ret_list,
            "execution_successful": execution_error is None,
            "execution_error": execution_error,
            "row_count": len(rows)
        }

        # Add simple summary
        if execution_error:
            response["summary"] = f"Query execution failed: {execution_error}"
        elif rows:
            response["summary"] = f"Found {len(rows)} records"
        else:
            response["summary"] = "No records found matching your query"

        logger.info(f"Returning response with {len(rows)} rows and {len(ret_list)} table suggestions")
        return response
            
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.get("/api/debug/tables")
async def debug_tables(schema: str = "dbo") -> Dict[str, Any]:
    """Debug endpoint to list actual tables in the database"""
    try:
        tables = db.get_schema_tables(schema)
        return {
            "schema": schema,
            "table_count": len(tables),
            "tables": tables[:50],  # First 50 tables
            "has_more": len(tables) > 50
        }
    except Exception as e:
        logger.error(f"Failed to get tables: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tables: {str(e)}")

@router.get("/api/debug/table-info/{schema}/{table}")
async def debug_table_info(schema: str, table: str) -> Dict[str, Any]:
    """Debug endpoint to get table column information"""
    try:
        columns = db.get_table_columns(schema, table)
        return {
            "schema": schema,
            "table": table,
            "column_count": len(columns),
            "columns": columns
        }
    except Exception as e:
        logger.error(f"Failed to get table info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get table info: {str(e)}")

@router.get("/api/debug/index")
async def debug_index() -> Dict[str, Any]:
    """Debug endpoint to check index status"""
    try:
        index = load_index()
        if not index:
            return {"status": "no_index", "message": "No index file found"}
        
        return {
            "status": "loaded",
            "database": index.get("database"),
            "table_count": len(index.get("tables", {})),
            "token_count": len(index.get("by_token", {})),
            "sample_tables": list(index.get("tables", {}).keys())[:10]
        }
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        return {"status": "error", "error": str(e)}