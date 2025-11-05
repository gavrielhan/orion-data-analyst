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
    Provides canonical schema source cached once and shared across nodes.
    """
    
    SCHEMA_CACHE_FILE = Path(__file__).parent.parent.parent / "schema_context.txt"
    CACHE_DURATION_SEC = 3600
    MAX_HISTORY = 5  # Keep last 5 interactions for context
    
    @classmethod
    def get_schema_context(cls) -> str:
        """Get schema context from canonical source file."""
        try:
            if cls.SCHEMA_CACHE_FILE.exists():
                return cls.SCHEMA_CACHE_FILE.read_text()
            return "Schema unavailable. You can query orders, order_items, products, and users only."
        except Exception as e:
            # If schema file can't be read, return fallback message
            return f"Schema file error: {str(e)}. You can query orders, order_items, products, and users only."
    
    @classmethod
    def execute(cls, state: AgentState) -> Dict[str, Any]:
        """Load schema and conversation context."""
        import sys
        print("Visiting ContextNode")
        sys.stdout.flush()
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        import sys
        if state.get("_verbose"):
            print(OutputFormatter.info("  → Loading schema..."))
            sys.stdout.flush()
        
        cache_timestamp = state.get("schema_cache_timestamp", 0)
        current_time = time.time()
        
        # Load schema from canonical source
        schema_context = cls.get_schema_context()
        
        # Maintain conversation history (limit to last N)
        history = state.get("conversation_history", []) or []
        if len(history) > cls.MAX_HISTORY:
            history = history[-cls.MAX_HISTORY:]
        
        return {
            "schema_context": schema_context,
            "schema_cache_timestamp": current_time,
            "conversation_history": history
        }

class ApprovalNode:
    """
    Human-in-the-loop approval for high-cost or sensitive queries.
    Flags queries exceeding cost threshold for user approval.
    """
    
    APPROVAL_THRESHOLD_GB = 5.0  # Require approval for >5GB queries
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Check if query requires user approval."""
        import sys
        print("Visiting ApprovalNode")
        sys.stdout.flush()
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        import sys
        if state.get("_verbose"):
            print(OutputFormatter.info("  → Checking cost..."))
            sys.stdout.flush()
        
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
        import sys
        print("Visiting ValidationNode")
        sys.stdout.flush()
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        import sys
        if state.get("_verbose"):
            print(OutputFormatter.info("  → Validating SQL..."))
            sys.stdout.flush()
        
        sql_query = state.get("sql_query", "")
        
        if not sql_query:
            return {"validation_passed": False, "query_error": "No SQL query to validate"}
        
        # Safety check: ensure this isn't a META response that slipped through
        sql_upper = sql_query.upper().strip()
        if sql_upper.startswith("META:") or sql_upper.startswith("SQL:"):
            # CRITICAL: Don't increment retry_count here - QueryBuilderNode handles it
            # Just preserve the current retry_count
            current_retry_count = state.get("retry_count", 0)
            return {
                "validation_passed": False,
                "query_error": "Invalid SQL: Response prefix detected. Please retry.",
                "retry_count": current_retry_count,  # Preserve, don't increment - QueryBuilderNode handles it
                "sql_query": sql_query  # Preserve SQL query for retry
            }
        
        # Security check: Block dangerous operations
        sql_upper = sql_query.upper()
        for keyword in cls.BLOCKED_KEYWORDS:
            if re.search(rf'\b{keyword}\b', sql_upper):
                # CRITICAL: Preserve retry_count - QueryBuilderNode handles incrementing
                current_retry_count = state.get("retry_count", 0)
                return {
                    "validation_passed": False,
                    "query_error": f"Security violation: {keyword} operations are not allowed",
                    "sql_query": sql_query,  # Preserve SQL query for retry
                    "retry_count": current_retry_count  # Preserve retry_count
                }
        
        # Note: We no longer enforce LIMIT - queries can run without forced limits
        # Users can specify their own LIMIT if needed
        
        # Cost estimation using BigQuery dry_run
        try:
            client = bigquery.Client(project=config.google_cloud_project)
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            query_job = client.query(sql_query, job_config=job_config)
            
            # Get estimated bytes processed
            bytes_processed = query_job.total_bytes_processed or 0
            gb_processed = bytes_processed / (1024 ** 3)
            
            if gb_processed > cls.MAX_COST_GB:
                # CRITICAL: Preserve retry_count - QueryBuilderNode handles incrementing
                current_retry_count = state.get("retry_count", 0)
                return {
                    "validation_passed": False,
                    "estimated_cost_gb": gb_processed,
                    "query_error": f"Query too expensive: {gb_processed:.2f}GB (max: {cls.MAX_COST_GB}GB)",
                    "sql_query": sql_query,  # Preserve SQL query for retry
                    "retry_count": current_retry_count  # Preserve retry_count
                }
            
            return {
                "validation_passed": True,
                "estimated_cost_gb": gb_processed,
                "sql_query": sql_query  # Return potentially modified query (with LIMIT)
            }
            
        except Exception as e:
            # CRITICAL: Preserve retry_count so retries can be tracked properly
            current_retry_count = state.get("retry_count", 0)
            return {
                "validation_passed": False,
                "query_error": f"Validation error: {str(e)}",
                "sql_query": sql_query,  # Preserve SQL query for retry - QueryBuilderNode needs it
                "retry_count": current_retry_count  # Preserve retry_count - don't reset it
            }

class InputNode:
    """Receives and normalizes user query."""
    
    # Quick meta-question responses (no LLM needed)
    META_RESPONSES = {
        "help": "I can analyze e-commerce data from BigQuery. Ask me about sales, customers, products, orders, trends, and more. Try: 'show me top 10 products' or 'analyze sales by category'",
        "what can you do": "I can query the bigquery-public-data.thelook_ecommerce dataset with tables: orders, order_items, products, and users. I can analyze trends, create visualizations, segment customers, detect anomalies, and answer questions about your e-commerce data.",
        "hello": "Hello! I'm Orion, your AI data analyst. I can help you analyze e-commerce data. What would you like to know?",
        "hi": "Hi! I'm Orion. Ask me anything about orders, products, customers, or sales data.",
        "capabilities": "I can query BigQuery, generate SQL, create charts (bar, line, pie, scatter, box), perform RFM analysis, detect outliers, compare time periods, and provide business insights.",
    }
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Process user input and classify intent."""
        import sys
        print("Visiting InputNode")
        sys.stdout.flush()
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        import sys
        if state.get("_verbose"):
            print(OutputFormatter.info("  → Processing query..."))
            sys.stdout.flush()
        
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

    MAIN_PROMPT = dedent("""
You are **Orion**, an expert analyst for `bigquery-public-data.thelook_ecommerce`
(tables: orders, order_items, products, users).

Goal → Return **one line**:
- `META:` if the user asks about your abilities or available data.
- `SQL:` if they request actual data.
- `DISCOVER:` only if you truly need to inspect distinct column values first.

Rules:
- Use only the four tables above with full path and backticks:
  `bigquery-public-data.thelook_ecommerce.<table>`
- Prefix columns with aliases (o, oi, p, u) to avoid ambiguity.
- Don’t explain; output must begin with exactly `META:`, `SQL:` or `DISCOVER:`.

Joins:
  orders.user_id = users.id  
  orders.order_id = order_items.order_id  
  order_items.product_id = products.id

User query: {user_query}
{history}
{discovery}
{retry_hint}
""")


    RETRY_PROMPT = dedent("""
Previous SQL failed on BigQuery. Fix it.

{schema}
User query: {user_query}
Previous SQL: {previous_sql}
Error: {error_log}

Return only:
SQL: <corrected query>

Rules:
- Keep full table paths with backticks.
- Alias columns to resolve ambiguity.
""")


    RATE_LIMIT_BACKOFF = (15, 45)

    def __init__(self):
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel(config.gemini_model)
        from src.utils.rate_limiter import get_global_rate_limiter

        self.rate_limiter = get_global_rate_limiter()
        self._cache: Dict[str, str] = {}
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        from src.utils.formatter import OutputFormatter
        import sys
        
        print("Visiting QueryBuilderNode")
        sys.stdout.flush()
        
        # Extract state
        discovery_result = state.get("discovery_result")
        retry_count = state.get("retry_count", 0)
        user_query = state.get("user_query", "")
        
        # Check if this is a retry after a BigQuery error
        query_error = state.get("query_error")
        previous_sql = state.get("sql_query", "")
        
        try:
            context = ContextNode.get_schema_context()
        except Exception as e:
            raise
        
        # Progress indicator (verbose only)
        if state.get("_verbose"):
            status = "Generating SQL with discovered data" if discovery_result else f"Generating SQL (retry {retry_count})" if retry_count else "Analyzing query"
            print(OutputFormatter.info(f"  → {status}..."))
            sys.stdout.flush()
        
        # Check if this is a retry after a BigQuery error (already extracted above)
        error_history = state.get("error_history", []) or []
        
        # CRITICAL: Check retry limit BEFORE attempting retry
        if query_error and previous_sql:
            
            # Check limit BEFORE incrementing
            if retry_count >= 3:
                return {
                    "query_error": f"Query failed after {retry_count} retry attempts. Original error: {query_error[:200]}",
                    "retry_count": retry_count,
                    "sql_query": previous_sql  # Preserve the last SQL
                }
            
            # This is a retry - increment retry_count now that we're actually attempting a retry
            # Store this incremented value so we can use it in error returns
            retry_count = retry_count + 1
            
            # Add current error to history if not already there
            if query_error and query_error not in error_history:
                error_history.append(query_error)
            
            # Build error context from history for better self-healing
            error_context = "\n".join([f"- Attempt {i+1}: {err}" for i, err in enumerate(error_history)])
            
            # This is a retry - include full error context for self-healing
            prompt = f"""
You are a SQL expert. Previous SQL queries failed. Learn from these errors and fix the query.

{context}

Original user query: {user_query}

Previous SQL query that failed:
{previous_sql}

Error history (most recent last):
{error_context}

This is retry attempt {retry_count} of 3. Carefully analyze the error pattern and generate a corrected query.

CRITICAL RULES:
- Fix the SQL query based on the error message above
- Use standard SQL syntax for BigQuery
- ALWAYS prefix ALL table names with the FULL path: 'bigquery-public-data.thelook_ecommerce.'
- IMPORTANT: For BigQuery public datasets, use backticks around the ENTIRE path, not each part
- Examples of CORRECT table references:
  * `bigquery-public-data.thelook_ecommerce.order_items`
  * `bigquery-public-data.thelook_ecommerce.orders`
  * `bigquery-public-data.thelook_ecommerce.products`
  * `bigquery-public-data.thelook_ecommerce.users`
- When referencing columns, use the table alias or full path:
  * oi.sale_price (if table is aliased as oi)
  * `bigquery-public-data.thelook_ecommerce.order_items`.sale_price
- Examples of INCORRECT (DO NOT USE):
  * `bigquery-public-data`.`thelook_ecommerce`.`order_items` ❌ (separate backticks)
  * bigquery-public-data.thelook_ecommerce.order_items ❌ (missing backticks)
  * bigquery.order_items ❌
  * thelook_ecommerce.order_items ❌
- ALWAYS use table aliases and prefix column names with the alias to avoid ambiguity
  Example: SELECT u.gender, EXTRACT(YEAR FROM u.created_at) AS year, COUNT(*) AS count
           FROM `bigquery-public-data.thelook_ecommerce.users` AS u
           GROUP BY u.gender, year
- Common ambiguous columns: created_at, updated_at, id, status - ALWAYS prefix with table alias
- Use clear column aliases

🚨 CRITICAL: DATE/TIME TYPE HANDLING - READ THIS FIRST IF ERROR MENTIONS TIMESTAMP/DATE! 🚨

1. TIMESTAMP_SUB vs DATE_SUB:
   - ERROR: "TIMESTAMP_SUB does not support the MONTH date part"
   - CAUSE: You used TIMESTAMP_SUB with MONTH/YEAR interval - BigQuery doesn't support this!
   - WRONG: WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 MONTH)
   - CORRECT: WHERE created_at >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH))
   - RULE: For MONTH/YEAR intervals, ALWAYS use DATE_SUB, then wrap in TIMESTAMP() if comparing with TIMESTAMP column
   - For DAY intervals with timestamps, you CAN use TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)

2. Type Matching:
   - ERROR: "No matching signature for operator >= for argument types: TIMESTAMP, DATE"
   - CAUSE: You're comparing TIMESTAMP column (e.g., created_at) with DATE value
   - SOLUTION: Wrap DATE in TIMESTAMP() function: TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR))
   - Example: WHERE o.created_at >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH))
   - Example: WHERE o.created_at >= TIMESTAMP('2024-01-01')

3. Common Patterns:
   - NEVER: WHERE created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH)
   - ALWAYS: WHERE created_at >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH))
   - NEVER: TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 MONTH)
   - ALWAYS: TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH))

4. Quick Reference:
   - Use DATE_SUB for: MONTH, YEAR intervals
   - Use TIMESTAMP_SUB for: DAY, HOUR, MINUTE, SECOND intervals (when working with timestamps)
   - created_at columns are TIMESTAMP type - cast DATE values to TIMESTAMP using TIMESTAMP()

REVENUE/SALES QUERY BEST PRACTICES (if query involves revenue/sales):
- ALWAYS filter order_items.status = 'Complete' when calculating revenue - only completed orders count
- Join order_items → orders → users for geographical filters (country, state, city)
- Join order_items → products to get product names/info
- Use SUM(oi.sale_price) for revenue calculations
- If query returned empty results, check: 1) Did you filter status='Complete'? 2) Did you use correct geographical location name?

GEOGRAPHICAL QUERY BEST PRACTICES (if query involves location/country/region):
- Always join order_items → orders → users to access geographical fields (u.country, u.state, u.city)
- IMPORTANT: If uncertain about exact format (e.g., "United States" vs "US"), use DISCOVER first:
  * DISCOVER: SELECT DISTINCT country FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 20
  * This will show the exact format used in the database
- Use exact location names as they appear in the discovered results
- For countries: discover to see if database uses full names ('United States') or abbreviations ('US')
- For states/cities: discover to see exact values if uncertain
- If query mentions "[country/region/state/city]", filter by the appropriate geographical field (u.country, u.state, u.city) using discovered exact values
- Examples: "in the US" → DISCOVER country values first to see if it's 'US' or 'United States', then use discovered value
- If previous query returned empty results, DISCOVER geographical values to identify correct format

- Return ONLY the fixed SQL query, no explanations

Fixed SQL Query:
"""
        else:
            # Initial query - determine if it's a meta-question or requires SQL
            
            # Check if this is a retry due to missing prefix
            retry_instruction = ""
            if retry_count > 0:
                retry_instruction = f"\n\n⚠️ ATTENTION: Previous response was invalid. Your response MUST start with either 'META:' or 'SQL:' - nothing else. This is attempt {retry_count + 1} of 3.\n"
            
            # Build conversation context for follow-up questions
            conversation_history = state.get("conversation_history", []) or []
            conv_context = ""
            if conversation_history:
                conv_context = "\n\nCONVERSATION HISTORY (for follow-up context):\n"
                for i, entry in enumerate(conversation_history[-3:], 1):  # Last 3 only
                    conv_context += f"{i}. User: {entry.get('query', 'N/A')}\n"
                    result_summary = entry.get('result_summary', 'N/A')
                    if len(result_summary) > 150:
                        result_summary = result_summary[:150] + "..."
                    conv_context += f"   Result: {result_summary}\n"
            
            # Add discovery results if available
            discovery_result = state.get("discovery_result")
            discovery_count = state.get("discovery_count", 0)
            discovery_context = ""
            if discovery_result:
                discovery_context = f"\n\nDISCOVERY RESULTS:\nYou previously discovered these data values:\n{discovery_result}\n\nCRITICAL: You MUST now generate the main SQL query using this information. Do NOT generate another DISCOVER query - use the discovery results above to generate SQL directly.\n"
            elif discovery_count > 0:
                discovery_context = f"\n\n⚠️ WARNING: You have already generated {discovery_count} discovery queries. You MUST generate SQL now, not another DISCOVER query.\n"
            
            prompt = f"""
You are an intelligent data analysis assistant named Orion. Your role is to help users query and analyze e-commerce data.

{context}
{conv_context}
{discovery_context}
User query: {user_query}
{retry_instruction}
ANALYZE THE QUERY:
Handle follow-up questions (e.g., "show the same for last quarter", "break that down by region") by referencing conversation history above.
Pay attention to corrections/clarifications (e.g., "I think you're wrong, check again", "actually it's X not Y") - use these to fix previous errors.

Carefully determine what the user is asking:
- If they're asking about YOUR CAPABILITIES, WHAT datasets/tables/columns are AVAILABLE, or general HELP → This is a META question about you
- If they're asking about ACTUAL DATA (specific values, records, numbers, calculations from the database) → This needs a SQL query

DATA DISCOVERY APPROACH (USE SPARINGLY):
CRITICAL: If discovery results are provided above, DO NOT generate another DISCOVER query - use those results to generate SQL directly.

PREFER DIRECT SQL GENERATION:
- For simple queries like "males and females count", use common encodings directly:
  * Gender: 'M' for male, 'F' for female
  * Status: 'Complete', 'Shipped', 'Processing', etc.
- Only use DISCOVER if you're truly uncertain about the exact encoding AND the query explicitly requires it

WHEN TO USE DISCOVERY FOR GEOGRAPHICAL QUERIES:
- If you're uncertain about the exact format of country/state/city names (e.g., "United States" vs "US", "United Kingdom" vs "UK")
- If the query mentions a country/region but you're unsure of the exact spelling or format in the database
- If previous queries returned empty results, discovery can help identify the correct format
- Examples of when to discover:
  * Query mentions "US" or "United States" → DISCOVER: SELECT DISTINCT country FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 20
  * Query mentions "UK" or "United Kingdom" → DISCOVER: SELECT DISTINCT country FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 20
  * Query mentions a state/city but format is uncertain → DISCOVER: SELECT DISTINCT state/city FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 20

If you're UNCERTAIN about data values AND no discovery results exist above:
1. Generate a DISCOVERY query to explore the data
2. Use DISTINCT to find unique values in the relevant column
3. Limit to 20 rows for fast results
4. For geographical queries, discover country/state/city values to see exact format

DISCOVERY QUERY FORMAT (ONLY if no discovery results exist above AND truly uncertain):
When you need to discover data values, respond with "DISCOVER:" prefix:
Example: "DISCOVER: SELECT DISTINCT gender FROM \`bigquery-public-data.thelook_ecommerce.users\` LIMIT 20"
Geographical example: "DISCOVER: SELECT DISTINCT country FROM \`bigquery-public-data.thelook_ecommerce.users\` LIMIT 20"

After seeing discovery results, you'll be asked again to generate the main query using discovered information.

EXAMPLES OF META QUESTIONS (answer directly, no SQL needed):
- "which dataset can you query?"
- "what tables are available?"
- "what columns are in the orders table?"
- "what can you do?"
- "help"
- "tell me about your capabilities"

EXAMPLES OF SQL QUESTIONS:
- "what is the most expensive product?" → Direct SQL
- "how many orders were placed?" → Direct SQL
- "show me the top 10 customers" → Direct SQL
- "show me the males and females count per year" → Direct SQL (use gender IN ('M', 'F'))
- "how many females are in orders?" → Direct SQL (use gender = 'F')
- "what are the top products by revenue?" → Direct SQL (MUST filter oi.status = 'Complete' for revenue)
- "what are the top 3 products sold in the US by revenue?" → DISCOVER country values first (to see if it's 'US' or 'United States'), then SQL
- "what are the top 3 products sold in United States by revenue?" → DISCOVER country values first (to confirm exact format), then SQL
- "what are the top products in [country/region]?" → If uncertain about format, DISCOVER first, then SQL (join order_items → orders → users → products, filter by u.country/u.state/u.city)
- "show me sales by country" → Direct SQL (join order_items → orders → users, group by u.country, filter oi.status='Complete')
- "what is the revenue in California?" → DISCOVER state values first (to confirm exact format), then SQL
- "what are the order statuses?" → DISCOVER status values first (if truly uncertain), then SQL

RESPONSE FORMAT (CRITICAL - FOLLOW EXACTLY):
You MUST respond in one of THREE formats. Your response MUST start with one of these prefixes:

1. If META question (about capabilities/datasets/tables):
   Response: "META: <your answer>"
   Example: "META: I can query the bigquery-public-data.thelook_ecommerce dataset..."
   
2. If you need to DISCOVER data values first:
   Response: "DISCOVER: <exploration query>"
   Example: "DISCOVER: SELECT DISTINCT gender FROM \`bigquery-public-data.thelook_ecommerce.users\` LIMIT 20"
   
3. If you have enough information for SQL:
   Response: "SQL: <query>"
   Example: "SQL: SELECT * FROM \`bigquery-public-data.thelook_ecommerce.orders\` LIMIT 10"

CRITICAL: Your response MUST start with exactly "META:", "DISCOVER:", or "SQL:" - no other text before it!

IMPORTANT: 
- For META questions, provide a helpful answer directly - do NOT generate SQL
- For SQL questions, provide ONLY the SQL query - no explanations or text before the SQL:

CRITICAL SQL RULES (for SQL and DISCOVER queries):
- Use standard SQL syntax for BigQuery
- ALWAYS prefix ALL table names with the FULL path: 'bigquery-public-data.thelook_ecommerce.'
- IMPORTANT: For BigQuery public datasets, use backticks around the ENTIRE path, not each part
- Examples: `bigquery-public-data.thelook_ecommerce.order_items`
- ALWAYS use table aliases and prefix column names with the alias to avoid ambiguity
  Example: SELECT u.gender, EXTRACT(YEAR FROM u.created_at) AS year, COUNT(*) AS count
           FROM `bigquery-public-data.thelook_ecommerce.users` AS u
           GROUP BY u.gender, year
- Common ambiguous columns: created_at, updated_at, id, status - ALWAYS prefix with table alias
- Use clear column aliases

REVENUE/SALES QUERY BEST PRACTICES (CRITICAL):
When calculating revenue or sales from order_items:
1. ALWAYS filter order_items.status = 'Complete' - only completed orders count as revenue
2. Join order_items → orders → users for geographical filters (country, state, city)
3. Join order_items → products to get product names/info
4. Use SUM(oi.sale_price) for revenue calculations

GEOGRAPHICAL QUERY BEST PRACTICES:
When filtering by geographical location (country, state, city):
1. Always join order_items → orders → users to access geographical fields (u.country, u.state, u.city)
2. IMPORTANT: Use exact geographical names as they appear in the database
3. If uncertain about the exact format (e.g., "United States" vs "US", "United Kingdom" vs "UK"), use DISCOVER first:
   - DISCOVER: SELECT DISTINCT country FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 20
   - This will show you the exact format used in the database
4. Country names may be full names (e.g., 'United States') or abbreviations (e.g., 'US') - discover to see which format is used
5. For state/city filters, if uncertain, discover the exact values:
   - DISCOVER: SELECT DISTINCT state FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 20
   - DISCOVER: SELECT DISTINCT city FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 20
6. After discovery, use the exact values you found in your SQL query
7. Examples of geographical filters (use discovered values):
   - Country: WHERE u.country = 'United States' (if discovered) or WHERE u.country = 'US' (if discovered)
   - State: WHERE u.state = 'California' (use exact value from discovery)
   - City: WHERE u.city = 'New York' (use exact value from discovery)
   - Multiple countries: WHERE u.country IN ('United States', 'Canada') (use discovered values)
8. When querying "top products by revenue in [country/region]":
   - If uncertain about country format, DISCOVER first
   - Join: order_items → orders → users → products
   - Filter by: u.country (or u.state, u.city) = '[exact discovered location value]'
   - Filter by: oi.status = 'Complete' (for revenue queries)
   - Group by: product fields (p.id, p.name)
   - Order by: revenue DESC
9. For geographical analysis queries (e.g., "sales by country", "revenue by state"):
   - Group by: u.country (or u.state, u.city)
   - Join: order_items → orders → users
   - Filter by: oi.status = 'Complete' (for revenue queries)

Your response (MUST start with META: or SQL:):
"""
        
        try:
            
            # Rate limiting to prevent API quota exhaustion
            status = self.rate_limiter.get_status()
            if state.get("_verbose"):
                print(OutputFormatter.info(f"  → Rate limit: {status['current_calls']}/{status['max_calls']} calls"))
            
            wait_time = self.rate_limiter.wait_if_needed(verbose=state.get("_verbose", False))
            if wait_time:
                print(f"⏱️  [QueryBuilderNode] Waited {wait_time:.1f}s for rate limit")
            sys.stdout.flush()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0.15, max_output_tokens=384),
            )
            
            # Handle potential None or empty response
            if not response or not hasattr(response, 'text'):
                # CRITICAL: Preserve existing sql_query if it exists
                existing_sql = state.get("sql_query", "")
                return {
                    "query_error": "No response from Gemini. Please check your API key.",
                    "retry_count": state.get("retry_count", 0) + 1,
                    "sql_query": existing_sql  # ALWAYS preserve existing SQL query
                }
            
            response_text = response.text.strip()
            
            # Check for empty response
            if not response_text:
                # CRITICAL: Preserve existing sql_query if it exists
                existing_sql = state.get("sql_query", "")
                return {
                    "query_error": "Gemini returned an empty response. Please rephrase your question.",
                    "retry_count": state.get("retry_count", 0) + 1,
                    "sql_query": existing_sql  # ALWAYS preserve existing SQL query
                }
            
            # Normalize response for checking
            response_upper = response_text.upper()
            # NOTE: Don't re-read retry_count from state here - we may have incremented it locally in the retry branch!
            
            # Check if this is a META question response (only for initial queries, not retries)
            if not query_error:
                # Primary check: LLM explicitly marked it as META
                if response_upper.startswith("META:"):
                    # Extract the answer (remove "META:" prefix)
                    meta_answer = response_text[5:].strip()
                    if meta_answer:
                        return {
                            "final_output": meta_answer,
                            "retry_count": 0  # Reset retry count on success
                        }
                
                # Fallback check: If no prefix but response clearly looks like meta answer
                # This handles cases where LLM forgot the prefix but gave a meta answer
                looks_like_sql = any(keyword in response_upper for keyword in 
                                   ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", 
                                    "LIMIT", "UNION", "`", "BIGQUERY-PUBLIC-DATA"])
                
                # More flexible meta keyword matching (handles plurals, variations)
                meta_patterns = [
                    "dataset", "datasets", "data set", "data sets",
                    "table", "tables", "column", "columns",
                    "available", "can query", "you can", "i can",
                    "capabilities", "help", "assistant", "orion"
                ]
                looks_like_meta = any(pattern in response_upper for pattern in meta_patterns)
                
                # If it clearly looks like a meta answer (not SQL), treat it as meta
                # Lower threshold for length (20 chars instead of 30) to catch shorter answers
                if looks_like_meta and not looks_like_sql and len(response_text) > 20:
                    return {
                        "final_output": response_text,
                        "retry_count": 0  # Reset retry count on success
                    }
            
            # Check if this is a DISCOVERY query (needs to explore data first)
            if response_upper.startswith("DISCOVER:"):
                discovery_result = state.get("discovery_result")
                if discovery_result:
                    # Discovery already done - LLM should use it, not generate another
                    # CRITICAL: Preserve existing sql_query if it exists
                    existing_sql = state.get("sql_query", "")
                    return {
                        "query_error": "Discovery already completed. Please generate SQL using the discovery results provided above.",
                        "retry_count": state.get("retry_count", 0) + 1,
                        "sql_query": existing_sql  # ALWAYS preserve existing SQL query
                    }
                
                discovery_query = response_text[9:].strip()
                if discovery_query:
                    # Prevent infinite discovery loops - limit to 2 discovery queries per query
                    discovery_count = state.get("discovery_count", 0)
                    if discovery_count >= 2:
                        # Too many discovery queries - force SQL generation
                        # CRITICAL: Preserve existing sql_query if it exists
                        existing_sql = state.get("sql_query", "")
                        return {
                            "query_error": "Too many discovery queries. Please generate SQL directly using available schema information.",
                            "retry_count": state.get("retry_count", 0) + 1,
                            "sql_query": existing_sql  # ALWAYS preserve existing SQL query
                        }
                    return {
                        "discovery_query": discovery_query,
                        "discovery_result": None,  # Clear old discovery result when starting new discovery
                        "discovery_count": discovery_count + 1  # Track discovery count
                    }
            
            # Check if this is a SQL question response
            if response_upper.startswith("SQL:"):
                # Extract the SQL query (remove "SQL:" prefix)
                sql_query = response_text[4:].strip()
                
                
            else:
                # Check if response lacks both META: and SQL: prefixes
                # Only check this for initial queries (not retries after BigQuery errors)
                if not query_error and not response_upper.startswith("META:") and not response_upper.startswith("SQL:"):
                    # If no prefix and we haven't exceeded max retries, retry with clearer instruction
                    if retry_count < 3:
                        # CRITICAL: Preserve existing sql_query if it exists
                        existing_sql = state.get("sql_query", "")
                        return {
                            "query_error": f"Invalid response format: Response must start with 'META:' or 'SQL:'. Please try again with proper format.",
                            "retry_count": retry_count + 1,
                            "sql_query": existing_sql  # ALWAYS preserve existing SQL query
                        }
                    else:
                        # Max retries exceeded, assume it's SQL and try to process it
                        sql_query = response_text
                else:
                    # For retries after BigQuery errors or when already processed, assume it's SQL
                    sql_query = response_text
            
            # Clean SQL - remove markdown code blocks if present
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.startswith("```"):
                sql_query = sql_query[3:]
            sql_query = sql_query.strip().rstrip("`")
            
            # Final safety check: ensure we didn't leave any prefixes in the SQL
            sql_query_upper = sql_query.upper().strip()
            if sql_query_upper.startswith("SQL:"):
                sql_query = sql_query[4:].strip()
            elif sql_query_upper.startswith("META:"):
                # This should have been caught earlier, but as a final safety measure
                return {
                    "final_output": sql_query[5:].strip(),
                    "retry_count": 0
                }
            
            # Post-process: Fix common mistakes automatically
            # Replace patterns like "bigquery.table" or "FROM bigquery.table" with correct path
            sql_lower_temp = sql_query.lower()
            if re.search(r'\bbigquery\s*\.\s*thelook_ecommerce', sql_lower_temp):
                # Pattern: bigquery.thelook_ecommerce -> bigquery-public-data.thelook_ecommerce
                sql_query = re.sub(
                    r'\bbigquery\s*\.\s*thelook_ecommerce',
                    'bigquery-public-data.thelook_ecommerce',
                    sql_query,
                    flags=re.IGNORECASE
                )
            
            if re.search(r'bigquery\s*\.\s*(order_items|orders|products|users)\b', sql_lower_temp):
                # Pattern: bigquery.table -> bigquery-public-data.thelook_ecommerce.table
                sql_query = re.sub(
                    r'bigquery\s*\.\s*(order_items|orders|products|users)\b',
                    lambda m: f'bigquery-public-data.thelook_ecommerce.{m.group(1)}',
                    sql_query,
                    flags=re.IGNORECASE
                )
            
            # Post-process: Add backticks for BigQuery identifiers with hyphens
            # BigQuery requires backticks around the ENTIRE path for public datasets with hyphens
            # Pattern: bigquery-public-data.thelook_ecommerce.table -> `bigquery-public-data.thelook_ecommerce.table`
            # We need to be careful not to double-quote already quoted identifiers
            if 'bigquery-public-data' in sql_query:
                # Check if already has backticks - if so, might need to fix format
                # First, handle table references: bigquery-public-data.thelook_ecommerce.table
                # Replace with backticks around entire path
                sql_query = re.sub(
                    r'(?<!`)bigquery-public-data\.thelook_ecommerce\.([a-z_]+)(?!`)',
                    r'`bigquery-public-data.thelook_ecommerce.\1`',
                    sql_query,
                    flags=re.IGNORECASE
                )
                # Fix any incorrectly quoted patterns like `bigquery-public-data`.`thelook_ecommerce`.`table`
                # Convert to `bigquery-public-data.thelook_ecommerce.table`
                sql_query = re.sub(
                    r'`bigquery-public-data`\.`thelook_ecommerce`\.`([a-z_]+)`',
                    r'`bigquery-public-data.thelook_ecommerce.\1`',
                    sql_query,
                    flags=re.IGNORECASE
                )
                # Fix column references that might have incorrect quoting
                sql_query = re.sub(
                    r'`bigquery-public-data`\.`thelook_ecommerce`\.`([a-z_]+)`\.`([a-z_]+)`',
                    r'`bigquery-public-data.thelook_ecommerce.\1`.\2',
                    sql_query,
                    flags=re.IGNORECASE
                )
            
            # Check if the LLM detected an invalid dataset
            if sql_query.startswith("ERROR:"):
                error_message = sql_query.replace("ERROR:", "").strip()
                return {
                    "final_output": error_message
                }
            
            # Validate SQL query - check for common issues
            sql_lower = sql_query.lower()
            
            # Check if query incorrectly references "bigquery" as a standalone identifier
            # This catches patterns like: FROM bigquery.table, JOIN bigquery.table, bigquery.table_name
            # Also catches: bigquery.table, FROM bigquery, JOIN bigquery, etc.
            # But exclude "bigquery-public-data" which is valid
            invalid_patterns = [
                r'\bbigquery\s*\.',  # bigquery.table or bigquery. table (but not bigquery-public-data)
                r'from\s+bigquery\s',  # FROM bigquery (space after)
                r'join\s+bigquery\s',  # JOIN bigquery (space after)
                r'\bbigquery\s+[a-z_]',  # bigquery followed by word (like "bigquery table")
            ]
            
            for pattern in invalid_patterns:
                matches = re.finditer(pattern, sql_lower)
                for match in matches:
                    # Check if this match is NOT part of "bigquery-public-data"
                    start, end = match.span()
                    context = sql_lower[max(0, start-20):min(len(sql_lower), end+20)]
                    # If the match is followed by "-public-data", it's valid
                    if not sql_lower[end:end+12].startswith('-public-data'):
                        return {
                            "query_error": "Invalid SQL generated: Query incorrectly references 'bigquery' as a table name. Please ensure all tables use the full path: 'bigquery-public-data.thelook_ecommerce.table_name'",
                            "retry_count": state.get("retry_count", 0) + 1
                        }
            
            # Ensure the query uses the correct dataset prefix
            # Check for correct prefix - accept both formats: with or without backticks
            has_correct_prefix = (
                "bigquery-public-data.thelook_ecommerce" in sql_lower or
                re.search(r'`bigquery-public-data\.thelook_ecommerce', sql_query, re.IGNORECASE)
            )
            
            if not has_correct_prefix:
                # Check if it references table names without the full path
                if any(re.search(rf'\b{table}\b', sql_lower) for table in ["order_items", "orders", "products", "users"]):
                    # If we see table names but not the full path, this is likely an error
                    # CRITICAL: Preserve the existing sql_query from state (for retries)
                    existing_sql = state.get("sql_query", "")
                    return {
                        "query_error": f"Invalid SQL generated: Query must use full table paths starting with 'bigquery-public-data.thelook_ecommerce.'. Generated query: {sql_query[:200]}",
                        "retry_count": state.get("retry_count", 0) + 1,
                        "sql_query": existing_sql if existing_sql else sql_query  # Preserve existing, or use new if none exists
                    }
            # Note: We removed the "identical SQL" check - let validation handle SQL errors
            # If the model generates identical SQL, validation will fail again and retry_count will increment
            # After 3 retries, the routing logic will stop retrying
            
            result = {
                "sql_query": sql_query,
                "discovery_result": None,  # Clear discovery result after using it
                "discovery_query": None,  # Clear any leftover discovery query
                "discovery_count": 0,  # Reset discovery count after successful SQL generation
                "query_error": None,  # Clear any previous errors
                "retry_count": retry_count,  # Use the incremented retry_count (was incremented in retry branch if this was a retry)
                "error_history": error_history  # Preserve error history
            }
            
            return result
        except Exception as e:
            error_str = str(e)
            retry_count = state.get("retry_count", 0)
            
            # API key errors
            if "API_KEY" in error_str or "API key" in error_str or "INVALID_ARGUMENT" in error_str:
                # CRITICAL: Preserve existing sql_query if it exists
                existing_sql = state.get("sql_query", "")
                return {
                    'query_error': 'Rate limit exceeded. Retrying...',
                    'retry_count': retry_count + 1,
                    'sql_query': existing_sql  # ALWAYS preserve existing SQL query
                }
            
            # Check for rate limit errors (429)
            # Retrying just makes another API call which hits rate limit again
            if "429" in error_str or "Resource exhausted" in error_str or "rate limit" in error_str.lower():
                
                # Check if our rate limiter shows low usage but API still says rate limited
                # This means quota was exhausted outside this session (previous sessions, other apps, etc.)
                status = self.rate_limiter.get_status()
                
                if status["current_calls"] < 5:
                    # Our rate limiter shows low usage, but API is rate limited
                    # This means the global Gemini quota is exhausted - we need to wait the full window
                    print(f"⏱️  Global Gemini API quota exhausted (not from this session).")
                    print(f"⏱️  Waiting 60 seconds for quota to reset...")
                    
                    # Wait the full 60 seconds to let Gemini's quota reset
                    for i in range(60, 0, -10):
                        print(f"⏱️  Waiting {i} seconds... (press Ctrl+C to cancel)", end="\r")
                        time.sleep(min(10, i))
                    print("\n⏱️  Wait complete. Quota should be reset now.")
                    
                    # Reset our rate limiter to reflect that we've waited
                    self.rate_limiter.reset()
                else:
                    # Our rate limiter also shows high usage - just reset it
                    self.rate_limiter.reset()
                    print(f"⏱️  Rate limiter reset - quota exhausted. Please wait 60 seconds.")
                
                # CRITICAL: Preserve existing sql_query if it exists
                existing_sql = state.get("sql_query", "")
                return {
                    "query_error": "⚠️ Rate limit exceeded. Gemini API quota exhausted.\n   Waited 60 seconds for quota reset. Please try again now.\n   If still failing, wait another 60 seconds or check if other apps are using your API key.",
                    "retry_count": retry_count,  # Don't increment - we're not retrying
                    "sql_query": existing_sql  # ALWAYS preserve existing SQL query
                }
            
            # Other errors
            # CRITICAL: Preserve existing sql_query if it exists
            existing_sql = state.get("sql_query", "")
            return {
                'query_error': '⚠️ Gemini rate limit reached. Please wait a minute before retrying.',
                'retry_count': retry_count,
                'sql_query': existing_sql  # ALWAYS preserve existing SQL query
            }

class BigQueryExecutorNode:
    """Executes SQL query on BigQuery with logging."""
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Execute SQL query or discovery query and return results."""
        import sys
        print("Visiting BigQueryExecutorNode")
        sys.stdout.flush()
        from src.utils.formatter import OutputFormatter
        
        # Check if this is a discovery query
        discovery_query = state.get("discovery_query", "")
        sql_query = state.get("sql_query", "")
        is_discovery = bool(discovery_query and not sql_query)

        
        # Progress indicator
        if state.get("_verbose"):
            if is_discovery:
                print(OutputFormatter.info("  → Discovering data values..."))
            else:
                print(OutputFormatter.info("  → Executing query on BigQuery..."))
            sys.stdout.flush()
        
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
                
                import sys
                if state.get("_verbose"):
                    print(OutputFormatter.success(f"  → Discovery completed: {len(df.columns)} columns found"))
                    sys.stdout.flush()
                return {
                    "discovery_result": discovery_result,
                    "discovery_query": None,  # CRITICAL: Clear discovery query after execution
                    "query_error": None
                }
            
            # Regular SQL query results
            return {
                "query_result": df,
                "query_error": None,
                "execution_time_sec": execution_time,
                "discovery_query": None  # Clear any leftover discovery query
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
                # NOTE: Don't increment retry_count here - it gets incremented in QueryBuilderNode when actually retrying
                # retry_count represents "how many retries have been attempted", not "how many errors occurred"
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
        import sys
        print("Visiting ResultCheckNode")
        sys.stdout.flush()
        from src.utils.formatter import OutputFormatter
        
        query_error = state.get("query_error")
        query_result = state.get("query_result")
        retry_count = state.get("retry_count", 0)

        
        # Progress indicator
        if state.get("_verbose"):
            print(OutputFormatter.info("  → Checking results..."))
            sys.stdout.flush()
        
        # Track error history for context propagation
        error_history = state.get("error_history", []) or []
        if query_error and query_error not in error_history:
            error_history.append(query_error)
        
        # Case 1: Query execution error - retry if under limit
        if query_error and retry_count < 3:
            if state.get("_verbose"):
                import sys
                # retry_count represents attempts so far, so next attempt is retry_count + 1
                print(OutputFormatter.warning(f"  → Query error detected (attempting retry {retry_count + 1}/3): {query_error[:100]}"))
                sys.stdout.flush()
            return {
                "error_history": error_history,
                "has_empty_results": False,
                # NOTE: Don't increment retry_count here - it gets incremented in QueryBuilderNode when actually attempting a retry
                # retry_count represents "how many retries have been attempted", not "how many errors occurred"
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
        import sys
        print("Visiting AnalysisNode")
        sys.stdout.flush()
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        import sys
        if state.get("_verbose"):
            print(OutputFormatter.info("  → Analyzing data..."))
            sys.stdout.flush()
        
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
        import sys
        print("Visiting InsightGeneratorNode")
        sys.stdout.flush()
        from src.utils.formatter import OutputFormatter
        
        # Progress indicator
        import sys
        if state.get("_verbose"):
            print(OutputFormatter.info("  → Generating insights..."))
            sys.stdout.flush()
        
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

Parse explicit specifications from query (override heuristics):
- Axis specs: "X on x-axis" / "Y on y-axis" → use those columns
- Chart type: "bar chart", "line chart", "pie chart" → chart_type
- Grouping: "grouped by X", "by X", "each Y contains N bars", "multiple categories" → hue_col: X
- Data structure: If x_col values repeat and there's a categorical column (2-10 values), use it as hue_col

Chart type selection:
- Bar: categorical comparisons, counts by group
- Line: trends over time, time series
- Pie: distribution/composition (single dimension)
- Scatter: correlation between two numeric values
- Box: distribution analysis (x_col=categorical, y_col=numeric, optional hue_col for grouping)

Grouping (hue_col):
- Set when: query mentions multiple categories OR data shows repeated x values with categorical grouping column
- Keywords: "by [category]", "grouped", "each X contains N", "male and female"
- Data pattern: x_col duplicates (2019, 2019, 2020...) + categorical column → use categorical as hue_col

Axis selection:
1. Follow explicit user specs (e.g., "year on x-axis")
2. Time-based: x_col = time/date, y_col = metric
3. Grouped: x_col = category, y_col = value, hue_col = grouping

Respond ONLY in JSON (no markdown):
{{"chart_type": "bar|line|pie|scatter|box", "x_col": "column_name", "y_col": "column_name", "hue_col": "column_name_or_null", "title": "Chart Title"}}

Rules:
- Pie: x_col=labels, y_col=values, hue_col=null
- Box: x_col=categorical (optional), y_col=numeric variable, hue_col=optional grouping
- Grouped charts: set hue_col to grouping column
- Always provide descriptive title"""
        
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
        import sys
        print("Visiting OutputNode")
        sys.stdout.flush()
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
                
                # Show data - limit display to first 50 rows if there are more
                total_rows = len(df)
                if total_rows > 50:
                    output_parts.append(f"📊 Results ({total_rows} rows, showing first 50):\n")
                    output_parts.append(df.head(50).to_string(index=False))
                    output_parts.append(f"\n... ({total_rows - 50} more rows)")
                else:
                    output_parts.append(f"📊 Results ({total_rows} rows):\n")
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

