# backend/app/services/integrated_nlp_service.py
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import boto3
from botocore.exceptions import ClientError
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

from app.services.db_service import SQLServerDatabaseService, DatabaseQueryError
from app.services.indexer import load_index
from app.services.retriever import search_tables, intent_from_question, parse_table_ref, make_sql_for_intent

logger = logging.getLogger(__name__)

# Environment variables
REGION = os.getenv("AWS_REGION", "ap-southeast-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class NLPService:
    """
    Fully integrated NLP service that handles the complete workflow:
    1. Language detection (AWS Comprehend)
    2. Translation to English (AWS Translate)
    3. Table/schema retrieval and SQL generation (OpenAI + local index)
    4. Database query execution
    5. Result explanation generation (OpenAI)
    6. Translation back to original language (AWS Translate)
    """

    ALLOWED_LANGS = {"en-US", "zh-TW", "zh-CN"}

    def __init__(
        self,
        db_service: SQLServerDatabaseService,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        region_name: str = "ap-southeast-1"
    ):
        self.db_service = db_service
        
        # Initialize AWS clients
        try:
            self.comprehend_client = boto3.client(
                "comprehend",
                region_name=REGION or region_name,
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
            )
            self.translate_client = boto3.client(
                "translate",
                region_name=REGION or region_name,
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
            )
            logger.info(f"AWS clients initialized for region: {REGION or region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise RuntimeError(f"AWS initialization failed: {e}")

        # Initialize OpenAI
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
        self.memory = ConversationBufferMemory(return_messages=True)

        # Language tracking
        self.last_language: Optional[str] = None
        self.last_confidence: Optional[float] = None

        # SQL generation prompt
        self.sql_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are a SQL expert converting natural language queries into SQL Server SELECT statements.

STRICT RULES:
- Use ONLY the tables/columns in the provided schema
- Generate exactly ONE SELECT statement (or WITH...SELECT if needed)
- No DDL/DML operations, no comments, no explanations, no markdown fences
- If the schema doesn't support the question, respond with: SELECT 1 WHERE 1=0
- Use proper SQL Server syntax and functions
- Consider the suggested relevant tables for context

AVAILABLE SCHEMA:
{schema}

RELEVANT TABLES FOUND:
{relevant_tables}

Generate a SQL query for the following request:
"""),
            HumanMessagePromptTemplate.from_template("{query}")
        ])

        # Result explanation prompt
        self.explanation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are a data analyst explaining database query results to business users.

Create a clear, concise summary of the query results in 1-2 sentences.
Focus on:
- What data was found
- Key numbers or patterns
- Business-relevant insights

Be natural and conversational. Don't mention technical SQL details.
"""),
            HumanMessagePromptTemplate.from_template("""
Original Question: {question}
SQL Query: {sql}
Results ({row_count} rows):
{results_summary}

Provide a clear business summary:""")
        ])

    @staticmethod
    def _normalize_lang(code: str, raw_text: str) -> str:
        """Map Comprehend output to {en-US, zh-TW, zh-CN}"""
        if not code:
            return "en-US"
        c = code.lower()
        if c.startswith("en"):
            return "en-US"
        if c in ("zh-tw", "zh-hant"):
            return "zh-TW"
        if c in ("zh", "zh-cn", "zh-hans"):
            return "zh-CN"
        return "en-US"

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language using AWS Comprehend"""
        try:
            resp = self.comprehend_client.detect_dominant_language(Text=text)
            if resp.get("Languages"):
                top = resp["Languages"][0]
                normalized = self._normalize_lang(top["LanguageCode"], text)
                self.last_language, self.last_confidence = normalized, top["Score"]
                return normalized, top["Score"]
            self.last_language, self.last_confidence = "en-US", 0.0
            return "en-US", 0.0
        except ClientError as e:
            logger.error(f"AWS Comprehend error: {e}")
            self.last_language, self.last_confidence = "en-US", 0.0
            return "en-US", 0.0
        except Exception as e:
            logger.error(f"Comprehend failure: {e}", exc_info=True)
            self.last_language, self.last_confidence = "en-US", 0.0
            return "en-US", 0.0

    def translate_to_english(self, text: str, src_lang: str) -> str:
        """Translate from zh-TW/zh-CN -> en; English returns as-is"""
        if src_lang == "en-US":
            return text
        source_code = "zh-TW" if src_lang == "zh-TW" else "zh"
        try:
            resp = self.translate_client.translate_text(
                Text=text, SourceLanguageCode=source_code, TargetLanguageCode="en"
            )
            return resp["TranslatedText"]
        except ClientError as e:
            logger.error(f"AWS Translate error: {e}")
            return text
        except Exception as e:
            logger.error(f"Translate failure: {e}", exc_info=True)
            return text

    def translate_from_english(self, text: str, tgt_lang: str) -> str:
        """Translate summaries back to zh-TW/zh-CN or keep en-US"""
        if tgt_lang == "en-US":
            return text
        target_code = "zh-TW" if tgt_lang == "zh-TW" else "zh"
        try:
            resp = self.translate_client.translate_text(
                Text=text, SourceLanguageCode="en", TargetLanguageCode=target_code
            )
            return resp["TranslatedText"]
        except ClientError as e:
            logger.error(f"AWS Translate error: {e}")
            return text
        except Exception as e:
            logger.error(f"Translate failure: {e}", exc_info=True)
            return text

    def _run_openai_chain(self, messages: List[BaseMessage]) -> str:
        """Run LLM with memory context"""
        past = self.memory.load_memory_variables({}).get("history", [])
        full_context = past + messages
        try:
            response = self.llm.invoke(full_context)
            content = str(response.content)
            # Save context
            last_user = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            if last_user:
                self.memory.save_context({"input": last_user.content}, {"output": content})
            return content
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get response from OpenAI: {e}")

    def _find_relevant_tables(self, english_query: str, schema_filter: Optional[str] = None) -> List[Tuple[str, float]]:
        """Find relevant tables using the search index"""
        try:
            index = load_index()
            if not index:
                logger.warning("No search index available")
                return []
            return search_tables(index, english_query, schema_filter=schema_filter, top_k=5)
        except Exception as e:
            logger.error(f"Error finding relevant tables: {e}")
            return []

    def _get_schema_context(self, relevant_tables: List[Tuple[str, float]]) -> str:
        """Get detailed schema information for relevant tables"""
        try:
            schema_parts = []
            for table_name, score in relevant_tables[:3]:  # Top 3 tables
                if "." in table_name:
                    schema, table = table_name.split(".", 1)
                    columns = self.db_service.get_table_columns(schema, table)
                    if columns:
                        col_info = ", ".join([f"{c['name']}:{c['type']}" for c in columns[:10]])  # Max 10 columns
                        schema_parts.append(f"{table_name}({col_info})")
            return "\n".join(schema_parts) if schema_parts else "No relevant tables found"
        except Exception as e:
            logger.error(f"Error getting schema context: {e}")
            return "Schema information unavailable"

    def _generate_sql(self, english_query: str, schema_context: str, relevant_tables: List[Tuple[str, float]]) -> str:
        """Generate SQL using OpenAI with schema context"""
        relevant_tables_text = "\n".join([f"- {table} (relevance: {score:.2f})" for table, score in relevant_tables])
        
        prompt = self.sql_prompt.format_messages(
            query=english_query,
            schema=schema_context,
            relevant_tables=relevant_tables_text
        )
        sql = self._run_openai_chain(prompt)
        return sql.strip()

    def _format_results_for_explanation(self, rows: List[Tuple], columns: List[str], max_rows: int = 5) -> str:
        """Format query results for explanation generation"""
        if not rows:
            return "No data returned"
        
        # Show first few rows as examples
        result_lines = []
        result_lines.append(f"Columns: {', '.join(columns)}")
        result_lines.append(f"Sample data (showing up to {max_rows} rows):")
        
        for i, row in enumerate(rows[:max_rows]):
            row_data = ", ".join([str(val) if val is not None else "NULL" for val in row])
            result_lines.append(f"Row {i+1}: {row_data}")
        
        if len(rows) > max_rows:
            result_lines.append(f"... and {len(rows) - max_rows} more rows")
            
        return "\n".join(result_lines)

    def _generate_explanation(self, original_question: str, sql: str, rows: List[Tuple], columns: List[str]) -> str:
        """Generate natural language explanation of results"""
        try:
            results_summary = self._format_results_for_explanation(rows, columns)
            
            prompt = self.explanation_prompt.format_messages(
                question=original_question,
                sql=sql,
                row_count=len(rows),
                results_summary=results_summary
            )
            
            explanation = self._run_openai_chain(prompt)
            return explanation.strip()
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Fallback to simple summary
            if rows:
                return f"Found {len(rows)} records matching your query."
            else:
                return "No records found matching your criteria."

    def process_complete_query(self, user_input: str, schema_name: Optional[str] = "dbo") -> Dict[str, Any]:
        """
        Complete end-to-end processing:
        1. Detect language & translate
        2. Find relevant tables
        3. Generate SQL
        4. Execute query
        5. Generate explanation
        6. Translate back
        """
        try:
            # Step 1: Language detection and translation
            detected_lang, confidence = self.detect_language(user_input)
            english_query = self.translate_to_english(user_input, detected_lang)
            
            logger.info(f"Processing query: {user_input} (detected: {detected_lang}, confidence: {confidence:.2f})")
            
            # Step 2: Intent analysis and table retrieval
            intent = intent_from_question(english_query)
            parsed_table = parse_table_ref(english_query)
            
            # Step 3: Find relevant tables
            relevant_tables = self._find_relevant_tables(english_query, schema_name)
            
            # Step 4: Handle different intents
            if intent in ["show_tables", "describe_table"] and parsed_table:
                # Use predefined SQL for specific intents
                sql = make_sql_for_intent(intent, schema_name or "dbo", parsed_table[0] if parsed_table else None)
            else:
                # Generate SQL using AI for complex queries
                schema_context = self._get_schema_context(relevant_tables)
                sql = self._generate_sql(english_query, schema_context, relevant_tables)
            
            # Step 5: Execute query
            rows, columns = [], []
            execution_error = None
            
            if sql and sql.strip() != "SELECT 1 WHERE 1=0":
                try:
                    rows, columns = self.db_service.run_select(sql, max_rows=100)
                    logger.info(f"Query executed successfully: {len(rows)} rows returned")
                except DatabaseQueryError as e:
                    execution_error = str(e)
                    logger.error(f"Query execution failed: {e}")
            
            # Step 6: Generate explanation
            if execution_error:
                explanation_en = f"Query execution failed: {execution_error}"
            else:
                explanation_en = self._generate_explanation(english_query, sql, rows, columns)
            
            # Step 7: Translate explanation back to original language
            explanation_localized = self.translate_from_english(explanation_en, detected_lang)
            
            # Prepare response
            response = {
                "original_text": user_input,
                "detected_language": detected_lang,
                "language_confidence": confidence,
                "english_text": english_query,
                "intent": intent,
                "schema": schema_name,
                "relevant_tables": [{"table": t, "score": round(s, 3)} for t, s in relevant_tables],
                "generated_sql": sql,
                "execution_successful": execution_error is None,
                "execution_error": execution_error,
                "columns": columns,
                "results": [list(row) for row in rows] if rows else [],
                "row_count": len(rows),
                "explanation_english": explanation_en,
                "explanation_localized": explanation_localized,
                "summary": explanation_localized  # For frontend compatibility
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Complete query processing failed: {e}", exc_info=True)
            error_msg = "An error occurred while processing your query."
            error_msg_localized = self.translate_from_english(error_msg, detected_lang if 'detected_lang' in locals() else "en-US")
            
            return {
                "original_text": user_input,
                "detected_language": detected_lang if 'detected_lang' in locals() else "en-US",
                "language_confidence": confidence if 'confidence' in locals() else 0.0,
                "execution_successful": False,
                "execution_error": str(e),
                "summary": error_msg_localized,
                "explanation_localized": error_msg_localized
            }

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def get_language_info(self) -> Dict[str, Any]:
        """Get last detected language info"""
        return {
            "language": self.last_language,
            "confidence": self.last_confidence,
            "allowed": self.last_language in self.ALLOWED_LANGS if self.last_language else False,
        }