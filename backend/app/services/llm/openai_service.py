# ================================================================================
# backend/app/services/llm/openai_service.py
from __future__ import annotations

import re
import os
import logging
from typing import List, Optional, Dict, Any


logger = logging.getLogger(__name__)

# Optional OpenAI imports
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class OpenAIService:
    """Handles OpenAI LLM operations for SQL generation and explanations."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.llm_enabled = bool(OPENAI_API_KEY) and ChatOpenAI is not None
        self.memory = None
        
        self._initialize_llm()
        self._initialize_prompts()
    
    def _initialize_llm(self):
        """Initialize OpenAI LLM client."""
        if not self.llm_enabled:
            logger.info("LLM disabled or langchain_openai not installed.")
            return
        
        try:
            # Handle both parameter styles across langchain versions
            try:
                self.llm = ChatOpenAI(
                    model=self.model_name, 
                    temperature=self.temperature, 
                    api_key=OPENAI_API_KEY
                )
            except TypeError:
                self.llm = ChatOpenAI(
                    model_name=self.model_name, 
                    temperature=self.temperature, 
                    openai_api_key=OPENAI_API_KEY
                )
            
            self.memory = ConversationBufferMemory(return_messages=True) if ConversationBufferMemory else None
            logger.info("LLM initialized model=%s", self.model_name)
            
        except Exception as e:
            logger.warning("LLM init failed: %s", e)
            self.llm = None
            self.llm_enabled = False
    
    def _initialize_prompts(self):
        """Initialize prompt templates."""
        if not ChatPromptTemplate:
            self.sql_prompt = None
            self.explanation_prompt = None
            return
        
        # SQL generation prompt
        self.sql_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an expert SQL analyst for HR leave & attendance data. "
                "Generate ONE safe SELECT query based on the user's question. "

                "HR data rules:\n"
                "- If the user asks for names or employee IDs, join the people tables:\n"
                "  LEFT JOIN dbo.PSNACCOUNT   AS P  ON P.PERSONID  = L.PERSONID\n"
                "  LEFT JOIN dbo.PSNACCOUNT_D AS PD ON PD.PERSONID = L.PERSONID AND P.PERSONID IS NULL\n"
                "  and SELECT COALESCE(P.TRUENAME,  PD.TRUENAME)  AS Name,\n"
                "             COALESCE(P.EMPLOYEEID,PD.EMPLOYEEID) AS EmployeeID\n"
                "- Use CAST(date_column AS date) for date filters.\n"
                "- For “today/this week/current”, use CAST(GETDATE() AS date).\n"
                "- When counting people per day, prefer COUNT(DISTINCT PERSONID) unless asked otherwise.\n"
                "- Only SELECT statements. Use CTEs if you need multiple steps.\n"
                "- Do NOT invent columns that aren’t in the provided schema.\n"

                "Target database: Microsoft SQL Server (T-SQL). STRICT DIALECT:\n"
                "- NEVER use LIMIT. For top-N use TOP (N) with ORDER BY.\n"
                "- Pagination: ORDER BY <col> OFFSET <n> ROWS FETCH NEXT <m> ROWS ONLY.\n"
                "- Use [square brackets] if you must quote identifiers.\n"

                "Available schemas:\n{schema}\n\n"
                "Join hints:\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(
                "Generate a SQL SELECT query for: {query}"
            ),
        ])
        
        # Result explanation prompt
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
    
    def _tsql_limit_fix(self, sql: str) -> str:
        if not sql:
            return sql
        # strip trailing semicolon to make regex simpler
        s = sql.strip()
        s_wo_sc = s[:-1] if s.endswith(";") else s

        # simple "LIMIT N" at the end → TOP (N)
        m = re.search(r"\blimit\s+(\d+)\s*$", s_wo_sc, flags=re.I)
        if m and re.search(r"^\s*select\b", s_wo_sc, flags=re.I):
            n = m.group(1)
            no_limit = re.sub(r"\blimit\s+\d+\s*$", "", s_wo_sc, flags=re.I).strip()
            return re.sub(r"(?i)^\s*select", f"SELECT TOP ({n})", no_limit, count=1)
        return s

    def generate_sql(self, query: str, schema: str, join_hints: str) -> str:
        if not (self.llm_enabled and self.sql_prompt):
            return "SELECT 1 WHERE 1=0"
        try:
            messages = self.sql_prompt.format_messages(
                query=query, schema=schema, join_hints=join_hints
            )
            raw = self._invoke_llm(messages) or ""
            return self._tsql_limit_fix(raw)
        except Exception as e:
            logger.error("SQL generation failed: %s", e, exc_info=True)
            return "SELECT 1 WHERE 1=0"
    
    def generate_explanation(self, question: str, row_count: int, columns: List[str], 
                           aggregates: Dict[str, Any], sample_text: str) -> str:
        """Generate explanation of query results."""
        if not (self.llm_enabled and self.explanation_prompt):
            return self._fallback_explanation(aggregates)
        
        try:
            import json
            messages = self.explanation_prompt.format_messages(
                question=question,
                row_count=row_count,
                columns=", ".join(columns) if columns else "(none)",
                aggregates_json=json.dumps(aggregates, ensure_ascii=False),
                sample_text=sample_text,
            )
            response = self._invoke_llm(messages)
            return response.strip() if response else self._fallback_explanation(aggregates)
        except Exception as e:
            logger.error("Explanation generation failed: %s", e, exc_info=True)
            return self._fallback_explanation(aggregates)
    
    def _invoke_llm(self, messages: List[BaseMessage]) -> str: # type: ignore
        """Invoke LLM with messages and handle memory."""
        if not (self.llm_enabled and self.llm and messages):
            return ""
        
        try:
            resp = self.llm.invoke(messages)
            content = str(resp.content)
            
            if self.memory:
                last_user = next(
                    (m for m in reversed(messages) if isinstance(m, HumanMessage)), 
                    None
                )
                if last_user:
                    self.memory.save_context(
                        {"input": last_user.content}, 
                        {"output": content}
                    )
            
            return content
        except Exception as e:
            logger.error("LLM invocation failed: %s", e, exc_info=True)
            return ""
    
    # openai_service.py
    def _fallback_explanation(self, aggregates: Dict[str, Any]) -> str:
        parts = []
        rc = int(aggregates.get("row_count", 0) or 0)
        parts.append(f"{rc} records.")
        up = aggregates.get("unique_people")
        if up is not None:
            parts.append(f"{up} unique people.")

        bt = aggregates.get("by_leave_type") or {}
        if bt:
            total = sum(bt.values())
            top = sorted(bt.items(), key=lambda kv: kv[1], reverse=True)[:3]
            pretty = ", ".join(f"{k} ({v}, {round((v/total)*100,1)}%)" for k, v in top if total)
            parts.append(f"Top leave types: {pretty}")

        th = aggregates.get("total_hours")
        if th:
            parts.append(f"Total hours: {th}")

        return " ".join(parts)

