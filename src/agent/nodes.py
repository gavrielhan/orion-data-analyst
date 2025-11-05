"""Node implementations for Orion agent."""

import pandas as pd
import re
import json
import time
from pathlib import Path
from textwrap import dedent
from typing import Dict, Any
from datetime import datetime
import google.generativeai as genai
from google.cloud import bigquery

from src.config import config
from src.agent.state import AgentState


# MetaQuestionHandler removed - LLM now decides if query is meta or SQL


class ContextNode:
    """
    Manages schema and conversation context.
    Loads previous queries/results for follow-up handling.
    """
    
    CACHE_DURATION_SEC = 3600  # 1 hour cache
    SCHEMA_FILE = Path(__file__).parent.parent.parent / "schemas.json"
    MAX_HISTORY = 5  # Keep last 5 interactions for context
    
    @classmethod
    def execute(cls, state: AgentState) -> Dict[str, Any]:
        """Load schema and conversation context."""
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        if state.get("_verbose"):
            print(OutputFormatter.info("  → ContextNode: Loading schema..."), end="\r")
        
        cache_timestamp = state.get("schema_cache_timestamp", 0)
        current_time = time.time()
        
        # Load schema if needed
        schema_context = state.get("schema_context")
        if not schema_context or (current_time - cache_timestamp) >= cls.CACHE_DURATION_SEC:
            schema_context = cls._load_schema()
        
        # Maintain conversation history (limit to last N)
        history = state.get("conversation_history", []) or []
        if len(history) > cls.MAX_HISTORY:
            history = history[-cls.MAX_HISTORY:]
        
        return {
            "schema_context": schema_context,
            "schema_cache_timestamp": current_time,
            "conversation_history": history
        }
    
    @classmethod
    def _load_schema(cls) -> str:
        """Load schema from file or fetch from BigQuery if file is missing/stale."""
        if cls.SCHEMA_FILE.exists():
            # Check file modification time
            file_mod_time = cls.SCHEMA_FILE.stat().st_mtime
            if (time.time() - file_mod_time) < cls.CACHE_DURATION_SEC:
                # File is fresh, use it
                with open(cls.SCHEMA_FILE, 'r') as f:
                    schemas = json.load(f)
                return cls._format_schema(schemas)
        
        # File doesn't exist or is stale - fetch from BigQuery
        try:
            schemas = cls._fetch_from_bigquery()
            # Save to file for future use
            with open(cls.SCHEMA_FILE, 'w') as f:
                json.dump(schemas, f, indent=2)
            return cls._format_schema(schemas)
        except Exception as e:
            # Fallback to file if fetch fails
            if cls.SCHEMA_FILE.exists():
                with open(cls.SCHEMA_FILE, 'r') as f:
                    schemas = json.load(f)
                return cls._format_schema(schemas)
            raise Exception(f"Failed to load schema: {e}")
    
    @classmethod
    def _fetch_from_bigquery(cls) -> dict:
        """Fetch schema directly from BigQuery."""
        client = bigquery.Client(project=config.google_cloud_project)
        dataset_ref = client.dataset("thelook_ecommerce", project="bigquery-public-data")
        
        schemas = {}
        for table in client.list_tables(dataset_ref):
            if table.table_id in ["orders", "order_items", "products", "users"]:
                table_ref = dataset_ref.table(table.table_id)
                table_obj = client.get_table(table_ref)
                
                schemas[table.table_id] = {
                    "description": table_obj.description or "",
                    "columns": [
                        {
                            "name": field.name,
                            "field_type": field.field_type,
                            "mode": field.mode,
                            "description": field.description or ""
                        }
                        for field in table_obj.schema
                    ]
                }
        
        return schemas
    
    @classmethod
    def _format_schema(cls, schemas: dict) -> str:
        """Format schema for LLM context."""
        parts = ["Available tables in bigquery-public-data.thelook_ecommerce:\n"]
        
        for table_name, schema in schemas.items():
            parts.append(f"\n{table_name}:")
            for col in schema["columns"]:
                parts.append(f"  - {col['name']} ({col['field_type']})")
        
        parts.append("\nJOINs: orders.user_id=users.id, orders.order_id=order_items.order_id, order_items.product_id=products.id")
        return "\n".join(parts)


class ApprovalNode:
    """
    Human-in-the-loop approval for high-cost or sensitive queries.
    Flags queries exceeding cost threshold for user approval.
    """
    
    APPROVAL_THRESHOLD_GB = 5.0  # Require approval for >5GB queries
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Check if query requires user approval."""
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        if state.get("_verbose"):
            print(OutputFormatter.info("  → ApprovalNode: Checking cost..."), end="\r")
        
        estimated_cost = state.get("estimated_cost_gb", 0)
        validation_passed = state.get("validation_passed", False)
        
        if not validation_passed:
            return {}
        
        # Check if approval needed
        if estimated_cost > ApprovalNode.APPROVAL_THRESHOLD_GB:
            return {
                "requires_approval": True,
                "approval_reason": f"Query will scan {estimated_cost:.2f} GB (threshold: {ApprovalNode.APPROVAL_THRESHOLD_GB} GB)"
            }
        
        return {
            "requires_approval": False,
            "approval_reason": None
        }


class ValidationNode:
    """Validates SQL queries for security, syntax, and cost."""
    
    BLOCKED_KEYWORDS = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"]
    MAX_COST_GB = 10.0  # Maximum 10GB scan
    
    @classmethod
    def execute(cls, state: AgentState) -> Dict[str, Any]:
        """Validate SQL query."""
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        if state.get("_verbose"):
            print(OutputFormatter.info("  → ValidationNode: Checking SQL..."), end="\r")
        
        sql_query = state.get("sql_query", "")
        
        if not sql_query:
            return {"validation_passed": False, "query_error": "No SQL query to validate"}
        
        # Safety check: ensure this isn't a META response that slipped through
        sql_upper = sql_query.upper().strip()
        if sql_upper.startswith("META:") or sql_upper.startswith("SQL:"):
            return {
                "validation_passed": False,
                "query_error": "Invalid SQL: Response prefix detected. Please retry.",
                "retry_count": state.get("retry_count", 0) + 1
            }
        
        # Security check: Block dangerous operations
        sql_upper = sql_query.upper()
        for keyword in cls.BLOCKED_KEYWORDS:
            if re.search(rf'\b{keyword}\b', sql_upper):
                return {
                    "validation_passed": False,
                    "query_error": f"Security violation: {keyword} operations are not allowed"
                }
        
        # Enforce row limit
        if "LIMIT" not in sql_upper:
            sql_query += "\nLIMIT 1000"
        
        # Cost estimation using BigQuery dry_run
        try:
            client = bigquery.Client(project=config.google_cloud_project)
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            query_job = client.query(sql_query, job_config=job_config)
            
            # Get estimated bytes processed
            bytes_processed = query_job.total_bytes_processed or 0
            gb_processed = bytes_processed / (1024 ** 3)
            
            if gb_processed > cls.MAX_COST_GB:
                return {
                    "validation_passed": False,
                    "estimated_cost_gb": gb_processed,
                    "query_error": f"Query too expensive: {gb_processed:.2f}GB (max: {cls.MAX_COST_GB}GB)"
                }
            
            return {
                "validation_passed": True,
                "estimated_cost_gb": gb_processed,
                "sql_query": sql_query  # Return potentially modified query (with LIMIT)
            }
            
        except Exception as e:
            return {
                "validation_passed": False,
                "query_error": f"Validation error: {str(e)}"
            }

class InputNode:
    """Receives and normalizes user query."""
    
    # Quick meta-question responses (no LLM needed)
    META_RESPONSES = {
        "help": "I can analyze e-commerce data from BigQuery. Ask me about sales, customers, products, orders, trends, and more. Try: 'show me top 10 products' or 'analyze sales by category'",
        "what can you do": "I can query the bigquery-public-data.thelook_ecommerce dataset with tables: orders, order_items, products, and users. I can analyze trends, create visualizations, segment customers, detect anomalies, and answer questions about your e-commerce data.",
        "hello": "Hello! I'm Orion, your AI data analyst. I can help you analyze e-commerce data. What would you like to know?",
        "hi": "Hi! I'm Orion. Ask me anything about orders, products, customers, or sales data.",
        "capabilities": "I can query BigQuery, generate SQL, create charts (bar, line, pie, scatter, box, candle), perform RFM analysis, detect outliers, compare time periods, and provide business insights.",
    }
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Process user input and classify intent."""
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        if state.get("_verbose"):
            print(OutputFormatter.info("  → InputNode: Processing query..."), end="\r")
        
        user_query = state.get("user_query", "")
        query_lower = user_query.lower().strip()
        
        # Fast path: Check for common meta-questions (instant response)
        # Use word boundaries to avoid substring matches (e.g., "hi" in "this")
        for pattern, response in InputNode.META_RESPONSES.items():
            # Match as whole query or at word boundaries
            if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.endswith(" " + pattern):
                return {
                    "query_intent": "meta_question",
                    "final_output": response
                }
        
        # Simple intent classification for data queries
        if any(keyword in query_lower for keyword in ["sales", "revenue", "total"]):
            intent = "aggregation"
        elif any(keyword in query_lower for keyword in ["top", "best", "highest"]):
            intent = "ranking"
        elif any(keyword in query_lower for keyword in ["trend", "over time", "monthly"]):
            intent = "trend_analysis"
        elif any(keyword in query_lower for keyword in ["count", "number of"]):
            intent = "counting"
        else:
            intent = "general_query"
        
        return {
            "query_intent": intent
        }


class QueryBuilderNode:
    """Generate SQL or meta answers with Gemini while minimising token-heavy prompts."""

    MAIN_PROMPT = dedent(
        """
        You are Orion, an analyst for the dataset `bigquery-public-data.thelook_ecommerce`.
        Use only these tables: orders, order_items, products, users.
        Always reference tables exactly as `bigquery-public-data.thelook_ecommerce.<table>` and include LIMIT 100 unless the user provides another limit.
        Respond with either:
          • `META: <concise answer>` for questions about the assistant or available data.
          • `SQL: <query>` for runnable Standard SQL.
        Never add prose before the prefix.
        {schema}

        User request: {user_query}
        {history}
        {discovery}
        {retry_hint}
        """
    )

    RETRY_PROMPT = dedent(
        """
        The previous SQL failed on BigQuery. Study the error log and return a corrected query.
        {schema}

        Original request: {user_query}
        Previous SQL:
        {previous_sql}

        Error history:
        {error_log}

        Return only `SQL: <query>` with fixes applied.
        """
    )

    RATE_LIMIT_BACKOFF = (15, 45)

    def __init__(self):
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel(config.gemini_model)
        from src.utils.rate_limiter import get_global_rate_limiter

        self.rate_limiter = get_global_rate_limiter()
        self._schema_context = self._load_schema_context()
        self._cache: Dict[str, str] = {}

    def _load_schema_context(self) -> str:
        schema_file = Path(__file__).parent.parent.parent / 'schema_context.txt'
        if schema_file.exists():
            try:
                return schema_file.read_text()
            except Exception:
                pass

        schema_json = Path(__file__).parent.parent.parent / 'schemas.json'
        if schema_json.exists():
            try:
                schemas = json.loads(schema_json.read_text())
                return self._format_schema(schemas)
            except Exception:
                pass

        return dedent(
            """
            Tables: orders, order_items, products, users from `bigquery-public-data.thelook_ecommerce`.
            Key joins: orders.user_id = users.id, orders.order_id = order_items.order_id, order_items.product_id = products.id.
            """
        ).strip()

    def _format_schema(self, schemas: dict) -> str:
        lines = ['Columns by table:']
        for table, meta in schemas.items():
            cols = ', '.join(col['name'] for col in meta.get('columns', []))
            if cols:
                lines.append(f'- {table}: {cols}')
        return '\n'.join(lines)

    def _cache_key(self, state: AgentState) -> str | None:
        if state.get('query_error') or state.get('discovery_result') or state.get('conversation_history'):
            return None
        user_query = (state.get('user_query') or '').strip().lower()
        return user_query or None

    def execute(self, state: AgentState) -> Dict[str, Any]:
        from src.utils.formatter import OutputFormatter

        if state.get('_verbose'):
            phase = 'retry' if state.get('retry_count') else 'analyzing'
            if state.get('discovery_result'):
                phase = 'discovery'
            print(OutputFormatter.info(f'  → QueryBuilder: {phase}...'), end='\r')

        cache_key = self._cache_key(state)
        if cache_key and cache_key in self._cache:
            return {'sql_query': self._cache[cache_key], 'discovery_result': None, 'query_error': None}

        prompt = self._build_prompt(state)

        try:
            self.rate_limiter.wait_if_needed()
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0.15, max_output_tokens=384),
            )
            raw_text = self._extract_text(response)

            upper = raw_text.upper()
            if upper.startswith('META:'):
                return {'final_output': raw_text[5:].strip(), 'retry_count': 0}

            if upper.startswith('SQL:'):
                raw_text = raw_text[4:].strip()

            sql_query = self._sanitize_sql(raw_text)

            if cache_key:
                self._cache[cache_key] = sql_query

            return {'sql_query': sql_query, 'discovery_result': None, 'query_error': None}

        except Exception as exc:
            handled = self._handle_generation_error(str(exc), state)
            if handled:
                return handled
            return {'query_error': f'Failed to generate SQL: {exc}', 'retry_count': state.get('retry_count', 0) + 1}

    def _build_prompt(self, state: AgentState) -> str:
        schema = self._schema_context
        user_query = state.get('user_query', '').strip()
        retry_count = state.get('retry_count', 0)
        discovery = state.get('discovery_result')
        history_items = state.get('conversation_history', []) or []

        history = ''
        if history_items:
            tail = history_items[-3:]
            formatted = [f"{i + 1}. {item.get('query', 'N/A')}: {item.get('result_summary', 'N/A')[:150]}" for i, item in enumerate(tail)]
            history = 'Recent context:\n' + '\n'.join(formatted)

        discovery_text = f'Known values:\n{discovery}' if discovery else ''

        if state.get('query_error') and state.get('sql_query'):
            error_history = state.get('error_history', []) or []
            error_log = '\n'.join(f'- {msg}' for msg in error_history) or '(no error message provided)'
            return self.RETRY_PROMPT.format(
                schema=schema,
                user_query=user_query,
                previous_sql=state.get('sql_query', ''),
                error_log=error_log,
            )

        retry_hint = 'Ensure the reply starts with `META:` or `SQL:`.' if retry_count else ''

        return self.MAIN_PROMPT.format(
            schema=schema,
            user_query=user_query,
            history=history,
            discovery=discovery_text,
            retry_hint=retry_hint,
        )

    def _extract_text(self, response) -> str:
        text = getattr(response, 'text', '') if response else ''
        text = text.strip()
        if not text:
            raise ValueError('Gemini API returned no content')
        if text.startswith('```'):
            text = re.sub(r'^```[a-zA-Z]*\n?|```$', '', text).strip()
        return text

    def _sanitize_sql(self, sql_query: str) -> str:
        sql_query = sql_query.strip().rstrip('`')
        if not sql_query:
            raise ValueError('Empty SQL generated')

        sql_query = re.sub(r'\bbigquery\s*\.\s*thelook_ecommerce', 'bigquery-public-data.thelook_ecommerce', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(
            r'bigquery\s*\.\s*(order_items|orders|products|users)\b',
            lambda m: f"bigquery-public-data.thelook_ecommerce.{m.group(1).lower()}",
            sql_query,
            flags=re.IGNORECASE,
        )
        sql_query = re.sub(
            r'(?<!`)bigquery-public-data\.thelook_ecommerce\.([a-z_]+)(?!`)',
            r'`bigquery-public-data.thelook_ecommerce.\1`',
            sql_query,
            flags=re.IGNORECASE,
        )
        sql_query = re.sub(
            r'`bigquery-public-data`\.`thelook_ecommerce`\.`([a-z_]+)`',
            r'`bigquery-public-data.thelook_ecommerce.\1`',
            sql_query,
            flags=re.IGNORECASE,
        )
        sql_query = re.sub(
            r'`bigquery-public-data`\.`thelook_ecommerce`\.`([a-z_]+)`\.`([a-z_]+)`',
            r'`bigquery-public-data.thelook_ecommerce.\1`.\2',
            sql_query,
            flags=re.IGNORECASE,
        )

        lowered = sql_query.lower()
        if re.search(r'\bbigquery(?!-public-data)[\s\.]', lowered):
            raise ValueError('Invalid SQL generated: use full dataset paths prefixed with `bigquery-public-data.thelook_ecommerce`.')

        if 'bigquery-public-data.thelook_ecommerce' not in lowered:
            raise ValueError('Invalid SQL generated: missing required dataset prefix.')

        return sql_query

    def _handle_generation_error(self, error_msg: str, state: AgentState) -> Dict[str, Any] | None:
        retry_count = state.get('retry_count', 0)

        if any(token in error_msg for token in ['API_KEY', 'API key', 'INVALID_ARGUMENT']):
            return {'query_error': '❌ Invalid Gemini API key. Check GEMINI_API_KEY in your environment.'}

        if any(token in error_msg.lower() for token in ['429', 'resource exhausted', 'rate limit']):
            if retry_count < len(self.RATE_LIMIT_BACKOFF):
                wait_seconds = self.RATE_LIMIT_BACKOFF[retry_count]
                if state.get('_verbose'):
                    from src.utils.formatter import OutputFormatter
                    print(OutputFormatter.warning(f'\n⏳ Gemini rate limit hit. Waiting {wait_seconds}s...'))
                time.sleep(wait_seconds)
                return {
                    'query_error': f'Rate limit exceeded. Retrying after {wait_seconds}s...',
                    'retry_count': retry_count + 1,
                }
            return {
                'query_error': '⚠️ Gemini rate limit reached. Please wait a minute before retrying.',
                'retry_count': retry_count,
            }

        return None

class BigQueryExecutorNode:
    """Executes SQL query on BigQuery with logging."""
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Execute SQL query or discovery query and return results."""
        from src.utils.formatter import OutputFormatter
        
        # Check if this is a discovery query
        discovery_query = state.get("discovery_query", "")
        sql_query = state.get("sql_query", "")
        is_discovery = bool(discovery_query and not sql_query)
        
        # Progress indicator
        if state.get("_verbose"):
            if is_discovery:
                print(OutputFormatter.info("  → Discovering data values..."), end="\r")
            else:
                print(OutputFormatter.info("  → Executing query on BigQuery..."), end="\r")
        
        query_to_execute = discovery_query if is_discovery else sql_query
        
        if not query_to_execute:
            return {
                "query_error": "No SQL query to execute",
                "query_result": None
            }
        
        try:
            client = bigquery.Client(project=config.google_cloud_project)
            
            # Track execution time
            start_time = time.time()
            query_job = client.query(query_to_execute)
            df = query_job.to_dataframe(max_results=config.max_query_rows)
            execution_time = time.time() - start_time
            
            # Log query execution
            BigQueryExecutorNode._log_query(
                query_to_execute, 
                execution_time, 
                query_job.total_bytes_processed,
                success=True
            )
            
            # Handle discovery results differently
            if is_discovery:
                # Format discovery results as a readable string
                discovery_result = "Discovered values:\n"
                for col in df.columns:
                    values = df[col].dropna().unique().tolist()[:20]  # Limit to 20 values
                    discovery_result += f"  {col}: {', '.join(map(str, values))}\n"

                return {
                    "discovery_result": discovery_result,
                    "discovery_query": None,  # Clear discovery query
                    "query_result": None,  # Ensure stale results don't persist between runs
                    "query_error": None,
                    "execution_time_sec": execution_time,
                }
            
            # Regular SQL query results
            return {
                "query_result": df,
                "query_error": None,
                "execution_time_sec": execution_time
            }
        except Exception as e:
            # Log failed query
            BigQueryExecutorNode._log_query(
                query_to_execute, 
                0, 
                0,
                success=False,
                error=str(e)
            )
            
            # Return error with helpful messages
            error_msg = str(e)
            
            # Provide helpful guidance for common errors
            if "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
                error_msg = "❌ BigQuery authentication failed.\n   Check GOOGLE_APPLICATION_CREDENTIALS in .env file.\n   Download service account key from: https://console.cloud.google.com/iam-admin/serviceaccounts"
            elif "project" in error_msg.lower() and "not found" in error_msg.lower():
                error_msg = "❌ Google Cloud project not found.\n   Check GOOGLE_CLOUD_PROJECT in .env file.\n   Find your project ID at: https://console.cloud.google.com/"
            elif "API has not been used" in error_msg or "disabled" in error_msg.lower():
                error_msg = "❌ BigQuery API is not enabled.\n   Enable it at: https://console.cloud.google.com/apis/library/bigquery.googleapis.com"
            elif "BigQuery execution error:" not in error_msg:
                error_msg = f"BigQuery execution error: {error_msg}"
            
            return {
                "query_error": error_msg,
                "query_result": None,
                "retry_count": state.get("retry_count", 0) + 1
            }
    
    @staticmethod
    def _log_query(sql: str, exec_time: float, bytes_processed: int, success: bool, error: str = None):
        """Log query execution details."""
        log_file = Path(__file__).parent.parent.parent / "query_log.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": sql[:500],  # Truncate long queries
            "execution_time_sec": round(exec_time, 3),
            "bytes_processed": bytes_processed,
            "cost_gb": round(bytes_processed / (1024 ** 3), 6) if bytes_processed else 0,
            "success": success,
            "error": error
        }
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Silent fail on logging errors


class ResultCheckNode:
    """
    Evaluates query execution results and determines next action.
    Routes to appropriate node based on: errors, empty results, or success.
    """
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Analyze execution results and set routing flags."""
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        if state.get("_verbose"):
            print(OutputFormatter.info("  → ResultCheckNode: Analyzing results..."), end="\r")
        
        query_error = state.get("query_error")
        query_result = state.get("query_result")
        retry_count = state.get("retry_count", 0)
        
        # Track error history for context propagation
        error_history = state.get("error_history", []) or []
        if query_error and query_error not in error_history:
            error_history.append(query_error)
        
        # Case 1: Query execution error - retry if under limit
        if query_error and retry_count < 3:
            return {
                "error_history": error_history,
                "has_empty_results": False
            }
        
        # Case 2: Check for empty results (successful query but no data)
        if query_result is not None and len(query_result) == 0:
            return {
                "has_empty_results": True,
                "error_history": error_history
            }
        
        # Case 3: Success with data
        return {
            "has_empty_results": False,
            "error_history": error_history
        }


class AnalysisNode:
    """
    Performs statistical analysis on query results.
    Supports: basic analysis + advanced (RFM, anomaly detection, comparative).
    """
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Analyze data based on query intent."""
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        if state.get("_verbose"):
            print(OutputFormatter.info("  → AnalysisNode: Analyzing data..."), end="\r")
        
        df = state.get("query_result")
        query_intent = state.get("query_intent", "general_query")
        user_query = state.get("user_query", "").lower()
        
        if df is None or len(df) == 0:
            return {}
        
        # Detect analysis type from query
        analysis_type = AnalysisNode._detect_analysis_type(user_query, query_intent)
        key_findings = []
        
        try:
            # Check for advanced analysis requests
            if "rfm" in user_query or "customer segment" in user_query:
                key_findings = AnalysisNode._rfm_analysis(df)
                analysis_type = "rfm_segmentation"
            elif "anomal" in user_query or "outlier" in user_query:
                key_findings = AnalysisNode._anomaly_detection(df)
                analysis_type = "anomaly_detection"
            elif "compar" in user_query or "versus" in user_query or "vs" in user_query:
                key_findings = AnalysisNode._comparative_analysis(df)
                analysis_type = "comparative"
            elif analysis_type == "ranking":
                key_findings = AnalysisNode._analyze_ranking(df)
            elif analysis_type == "trends":
                key_findings = AnalysisNode._analyze_trends(df)
            elif analysis_type == "segmentation":
                key_findings = AnalysisNode._analyze_segmentation(df)
            else:
                key_findings = AnalysisNode._analyze_aggregation(df)
            
            return {
                "analysis_type": analysis_type,
                "key_findings": key_findings
            }
        except Exception:
            return {
                "analysis_type": "aggregation",
                "key_findings": [f"Returned {len(df)} rows"]
            }
    
    @staticmethod
    def _detect_analysis_type(query: str, intent: str) -> str:
        """Detect type of analysis needed from query."""
        if any(kw in query for kw in ["top", "best", "highest", "lowest", "rank"]):
            return "ranking"
        elif any(kw in query for kw in ["trend", "over time", "monthly", "growth", "change"]):
            return "trends"
        elif any(kw in query for kw in ["by", "segment", "group", "category", "breakdown"]):
            return "segmentation"
        elif intent in ["ranking", "trend_analysis"]:
            return intent.replace("_analysis", "s")
        else:
            return "aggregation"
    
    @staticmethod
    def _analyze_ranking(df: pd.DataFrame) -> list:
        """Analyze ranked data and extract key insights."""
        findings = []
        
        if len(df) == 0:
            return findings
        
        # Find numeric column for ranking
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return [f"Top {min(5, len(df))} results"]
        
        value_col = numeric_cols[0]
        total = df[value_col].sum()
        
        # Top contributor
        if len(df) > 0:
            top_val = df.iloc[0][value_col]
            top_pct = (top_val / total * 100) if total > 0 else 0
            findings.append(f"Top result: {top_pct:.1f}% of total")
        
        # Top 3 concentration
        if len(df) >= 3:
            top3_val = df.head(3)[value_col].sum()
            top3_pct = (top3_val / total * 100) if total > 0 else 0
            findings.append(f"Top 3 represent {top3_pct:.1f}% of total")
        
        return findings
    
    @staticmethod
    def _analyze_trends(df: pd.DataFrame) -> list:
        """Analyze time-series trends and growth rates."""
        findings = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0 or len(df) < 2:
            return findings
        
        value_col = numeric_cols[0]
        values = df[value_col].values
        
        # Calculate growth rate
        if len(values) >= 2:
            first_val = values[0]
            last_val = values[-1]
            if first_val != 0:
                growth = ((last_val - first_val) / first_val) * 100
                findings.append(f"Overall change: {growth:+.1f}%")
        
        # Trend direction
        if len(values) >= 3:
            increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
            if increases > len(values) * 0.6:
                findings.append("Upward trend detected")
            elif increases < len(values) * 0.4:
                findings.append("Downward trend detected")
        
        return findings
    
    @staticmethod
    def _analyze_segmentation(df: pd.DataFrame) -> list:
        """Analyze segmented/grouped data."""
        findings = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return [f"{len(df)} segments identified"]
        
        value_col = numeric_cols[0]
        
        # Distribution stats
        findings.append(f"{len(df)} segments, avg: {df[value_col].mean():.1f}")
        
        # Identify largest segment
        if len(df) > 0:
            max_idx = df[value_col].idxmax()
            findings.append(f"Largest segment: {df.iloc[max_idx].iloc[0]}")
        
        return findings
    
    @staticmethod
    def _analyze_aggregation(df: pd.DataFrame) -> list:
        """Basic aggregation analysis."""
        findings = []
        
        findings.append(f"{len(df)} rows returned")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            total = df[col].sum()
            avg = df[col].mean()
            findings.append(f"Total: {total:.2f}, Average: {avg:.2f}")
        
        return findings
    
    @staticmethod
    def _rfm_analysis(df: pd.DataFrame) -> list:
        """RFM (Recency, Frequency, Monetary) customer segmentation."""
        findings = []
        
        # Look for relevant columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return ["RFM analysis requires numeric data"]
        
        # Calculate quartiles for segmentation
        value_col = numeric_cols[0]
        quartiles = df[value_col].quantile([0.25, 0.5, 0.75])
        
        # Segment customers
        high_value = df[df[value_col] >= quartiles[0.75]]
        medium_value = df[(df[value_col] >= quartiles[0.25]) & (df[value_col] < quartiles[0.75])]
        low_value = df[df[value_col] < quartiles[0.25]]
        
        findings.append(f"High-value segment: {len(high_value)} customers ({len(high_value)/len(df)*100:.1f}%)")
        findings.append(f"Medium-value segment: {len(medium_value)} customers ({len(medium_value)/len(df)*100:.1f}%)")
        findings.append(f"Low-value segment: {len(low_value)} customers ({len(low_value)/len(df)*100:.1f}%)")
        
        if len(high_value) > 0:
            findings.append(f"High-value avg: {high_value[value_col].mean():.2f}")
        
        return findings
    
    @staticmethod
    def _anomaly_detection(df: pd.DataFrame) -> list:
        """Detect outliers and unusual patterns using IQR method."""
        findings = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return ["No numeric data for anomaly detection"]
        
        value_col = numeric_cols[0]
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[value_col] < lower_bound) | (df[value_col] > upper_bound)]
        
        if len(outliers) > 0:
            findings.append(f"Detected {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
            findings.append(f"Outlier range: <{lower_bound:.2f} or >{upper_bound:.2f}")
            if len(outliers) <= 3:
                for idx in outliers.head(3).index:
                    findings.append(f"  Outlier: {df.loc[idx, value_col]:.2f}")
        else:
            findings.append("No significant outliers detected")
        
        return findings
    
    @staticmethod
    def _comparative_analysis(df: pd.DataFrame) -> list:
        """Period-over-period or segment comparison."""
        findings = []
        
        if len(df) < 2:
            return ["Insufficient data for comparison"]
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return ["No numeric data for comparison"]
        
        value_col = numeric_cols[0]
        
        # Compare first half vs second half
        mid = len(df) // 2
        first_half = df.iloc[:mid][value_col]
        second_half = df.iloc[mid:][value_col]
        
        first_avg = first_half.mean()
        second_avg = second_half.mean()
        
        if first_avg != 0:
            change_pct = ((second_avg - first_avg) / first_avg) * 100
            findings.append(f"Period 1 avg: {first_avg:.2f}")
            findings.append(f"Period 2 avg: {second_avg:.2f}")
            findings.append(f"Change: {change_pct:+.1f}%")
        else:
            findings.append("Cannot compute percentage change (division by zero)")
        
        return findings


class InsightGeneratorNode:
    """
    Generates natural language insights from analyzed data using LLM.
    Handles both empty results and data-rich analyses.
    """
    
    def __init__(self):
        genai.configure(api_key=config.gemini_api_key)
        # Use configured Gemini model (default: gemini-2.0-flash-exp)
        self.model = genai.GenerativeModel(config.gemini_model)
        
        # Use shared global rate limiter for all Gemini API calls
        from src.utils.rate_limiter import get_global_rate_limiter
        self.rate_limiter = get_global_rate_limiter()
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Generate insights from empty results or data analysis."""
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        if state.get("_verbose"):
            print(OutputFormatter.info("  → Generating insights..."), end="\r")
        
        user_query = state.get("user_query", "")
        has_empty_results = state.get("has_empty_results", False)
        
        # Handle empty results case
        if has_empty_results:
            return self._explain_empty_results(state)
        
        # Handle data analysis insights
        return self._generate_business_insights(state)
    
    def _explain_empty_results(self, state: AgentState) -> Dict[str, Any]:
        """Generate explanation for empty query results."""
        user_query = state.get("user_query", "")
        sql_query = state.get("sql_query", "")
        
        prompt = f"""A query returned no results. Explain why briefly (2 sentences max).

User question: {user_query}
SQL: {sql_query}

Possible reasons: filters too restrictive, no data for time period, typos, etc."""
        
        try:
            self.rate_limiter.wait_if_needed()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=150,
                )
            )
            
            insight = response.text.strip() if response and hasattr(response, 'text') else "No data found matching your criteria."
            
            return {
                "analysis_result": insight,
                "final_output": f"📭 No results found.\n\n💡 {insight}"
            }
        except Exception:
            return {
                "analysis_result": "No data found.",
                "final_output": "📭 No results found. Try adjusting your query criteria."
            }
    
    def _generate_business_insights(self, state: AgentState) -> Dict[str, Any]:
        """Generate actionable business insights from analysis."""
        user_query = state.get("user_query", "")
        analysis_type = state.get("analysis_type", "aggregation")
        key_findings = state.get("key_findings", [])
        df = state.get("query_result")
        
        # Build context from data
        data_summary = f"Analysis type: {analysis_type}\n"
        data_summary += f"Key findings:\n" + "\n".join([f"- {f}" for f in key_findings])
        
        if df is not None and len(df) <= 10:
            data_summary += f"\n\nData preview:\n{df.to_string(index=False)}"
        
        # NOTE: Visualization suggestion is now generated lazily in CLI when user requests a chart
        # This saves 1 LLM call per query (25% reduction in API usage)
        
        prompt = f"""You are a business analyst. Generate actionable insights from this data analysis.

User question: {user_query}

{data_summary}

Provide:
1. Brief interpretation (1 sentence)
2. Key insight or pattern (1 sentence)
3. Actionable recommendation if applicable (1 sentence)

Keep it concise and business-focused. Use bullet points."""
        
        try:
            self.rate_limiter.wait_if_needed()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=300,
                )
            )
            
            insights = response.text.strip() if response and hasattr(response, 'text') else "Analysis complete."
            
            return {"analysis_result": insights}
        except Exception:
            # Fallback to key findings
            return {
                "analysis_result": "\n".join(key_findings) if key_findings else "Analysis complete."
            }
    
    def _suggest_visualization(self, user_query: str, df, analysis_type: str) -> dict:
        """Use LLM to suggest the best visualization configuration based on query and data."""
        if df is None or len(df) == 0:
            return None
        
        # Get column info
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].head(3).tolist()
            unique_count = df[col].nunique()
            columns_info.append(f"- {col} ({dtype}, {unique_count} unique values): {sample_values}")
        
        columns_str = "\n".join(columns_info)
        
        prompt = f"""Analyze the user's query and data to suggest the optimal visualization.

User query: {user_query}
Analysis type: {analysis_type}

Available columns:
{columns_str}

CRITICAL: Parse the user's explicit chart specifications from their query (these override all heuristics):
- "year on x-axis" / "x as the years" / "with x as year" → x_col: order_year (or year column)
- "count on y-axis" / "y as the amounts" / "with y as count" → y_col: count
- "grouped by gender" / "one bar for females and one for males" / "each year must contain 2 bars" → hue_col: gender
- "plot them in a bar chart" → chart_type: bar
- "show as pie chart" → chart_type: pie

IMPORTANT: User queries often combine data request + chart specs in ONE sentence:
Example: "show female and male counts per year, plot them in bar chart with x as years and y as amounts, each year must contain 2 bars"
→ This means: x_col=order_year, y_col=count, hue_col=gender, chart_type=bar

Chart type guidelines:
- Bar: categorical comparison (sales by category, top products, counts by group)
- Line: trends over time (monthly revenue, daily orders, time series)
- Pie: distribution/composition (market share, category breakdown, single dimension)
- Scatter: correlation between two numeric values
- Box: distribution analysis of a numeric variable

Grouping (hue_col):
- Use hue_col when query mentions multiple categories/groups
- Keywords: "by gender", "by region", "by category", "male and female", "each X contains N bars"
- Example: "sales by region and product" → x_col: region, y_col: sales, hue_col: product
- Example: "female and male counts per year" → x_col: order_year, y_col: count, hue_col: gender
- Example: "each year contains 2 bars" → hue_col: (grouping column like gender/status)

Axis selection priority:
1. ALWAYS follow explicit user specifications (e.g., "year on x-axis")
2. For time-based queries: x_col = time/date column, y_col = metric
3. For grouped bars: x_col = category, y_col = value, hue_col = grouping

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{"chart_type": "bar|line|pie|scatter|box", "x_col": "column_name", "y_col": "column_name", "hue_col": "column_name_or_null", "title": "Chart Title"}}

Rules:
- For pie charts: x_col is labels, y_col is values, hue_col is null
- For box plots: x_col and y_col are the same numeric column, hue_col is null
- For grouped charts: set hue_col to the grouping column (e.g., gender, category, region)
- Always provide a descriptive title based on the query"""
        
        try:
            self.rate_limiter.wait_if_needed()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=200,  # Increased for processing explicit user specifications
                )
            )
            
            if not response or not hasattr(response, 'text'):
                return None
            
            # Parse JSON response
            import json
            import re
            
            text = response.text.strip()
            # Remove markdown code blocks if present
            text = re.sub(r'```json\s*|\s*```', '', text)
            
            suggestion = json.loads(text)
            
            # Validate suggestion
            required_keys = ['chart_type', 'x_col', 'y_col', 'title']
            if all(k in suggestion for k in required_keys):
                # Ensure columns exist in dataframe
                if suggestion['x_col'] in df.columns and suggestion['y_col'] in df.columns:
                    # Validate hue_col if provided (can be null)
                    if 'hue_col' in suggestion and suggestion['hue_col']:
                        if suggestion['hue_col'] not in df.columns:
                            suggestion['hue_col'] = None  # Invalid hue_col, ignore it
                    else:
                        suggestion['hue_col'] = None
                    return suggestion
            
            return None
            
        except Exception:
            return None


class OutputNode:
    """Formats and returns final output with metadata and visualizations."""
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Format output for display with analysis insights."""
        # Clear progress indicator line
        if state.get("_verbose"):
            print(" " * 80, end="\r")  # Clear the line
        
        # Check if final_output was already set (e.g., from META questions or empty results)
        existing_output = state.get("final_output", "")
        df = state.get("query_result")
        error = state.get("query_error")
        exec_time = state.get("execution_time_sec")
        cost_gb = state.get("estimated_cost_gb")
        retry_count = state.get("retry_count", 0)
        analysis_result = state.get("analysis_result")
        key_findings = state.get("key_findings", [])
        viz_path = state.get("visualization_path")
        
        if existing_output and existing_output.strip():
            return {"final_output": existing_output}
        
        # Build output with metadata
        output_parts = []
        
        if error:
            # Show retry attempts if any
            if retry_count > 0:
                output_parts.append(f"❌ Error (after {retry_count} retries): {error}")
            else:
                output_parts.append(f"❌ Error: {error}")
        elif df is not None:
            # Show cost estimate if available
            if cost_gb is not None and cost_gb > 0:
                output_parts.append(f"💰 Estimated cost: {cost_gb:.4f} GB scanned")
            
            if df.empty:
                output_parts.append("📭 No results found.")
            else:
                # Show key findings if available
                if key_findings:
                    output_parts.append("📈 Key Findings:")
                    for finding in key_findings:
                        output_parts.append(f"  • {finding}")
                    output_parts.append("")
                
                # Show data
                output_parts.append(f"📊 Results ({len(df)} rows):\n")
                output_parts.append(df.to_string(index=False))
                
                # Show insights if available
                if analysis_result:
                    output_parts.append(f"\n💡 Insights:\n{analysis_result}")
                
                # Show visualization if created
                if viz_path:
                    output_parts.append(f"\n📊 Chart saved to: {viz_path}")
            
            # Show execution time if available
            if exec_time is not None:
                output_parts.append(f"\n⏱️  Executed in {exec_time:.2f}s")
        else:
            output_parts.append("No results generated.")
        
        return {
            "final_output": "\n".join(output_parts)
        }


# Create singleton instances for the graph
input_node = InputNode()
query_builder_node = QueryBuilderNode()
approval_node = ApprovalNode()
bigquery_executor_node = BigQueryExecutorNode()
result_check_node = ResultCheckNode()
analysis_node = AnalysisNode()
insight_generator_node = InsightGeneratorNode()
output_node = OutputNode()

