"""Node implementations for Orion agent."""

import pandas as pd
import re
import json
from pathlib import Path
from typing import Dict, Any
import google.generativeai as genai

from src.config import config
from src.agent.state import AgentState


# MetaQuestionHandler removed - LLM now decides if query is meta or SQL

class InputNode:
    """Receives and normalizes user query."""
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Process user input and classify intent."""
        user_query = state.get("user_query", "")
        
        # Simple intent classification for MVP
        query_lower = user_query.lower()
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
    """Generates SQL query using Gemini API."""
    
    def __init__(self):
        # Initialize Gemini API
        genai.configure(api_key=config.gemini_api_key)
        # Use Gemini 2.0 Flash for faster responses
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
    def _load_schema_context(self) -> str:
        """Load schema context from saved schema file or fallback to hardcoded."""
        # Try to load from schema_context.txt file
        schema_file = Path(__file__).parent.parent.parent / "schema_context.txt"
        
        if schema_file.exists():
            try:
                with open(schema_file, 'r') as f:
                    return f.read()
            except Exception as e:
                # Fallback to hardcoded schema if file can't be read
                print(f"Warning: Could not load schema file: {e}")
        
        # Fallback: Try to load from JSON and format it
        schema_json = Path(__file__).parent.parent.parent / "schemas.json"
        if schema_json.exists():
            try:
                with open(schema_json, 'r') as f:
                    schemas = json.load(f)
                return self._format_schema_from_json(schemas)
            except Exception as e:
                print(f"Warning: Could not load schema JSON: {e}")
        
        # Final fallback to hardcoded schema
        return self._get_hardcoded_schema()
    
    def _format_schema_from_json(self, schemas: dict) -> str:
        """Format schema information from JSON for LLM context."""
        context_parts = [
            "CRITICAL: You can ONLY query these 4 tables in bigquery-public-data.thelook_ecommerce:\n"
        ]
        
        table_descriptions = {
            "orders": "customer orders with timestamps, status, and number of items",
            "order_items": "products within each order, including price and cost",
            "products": "catalog metadata like category, brand, and pricing",
            "users": "customer demographics like age, gender, and location"
        }
        
        for i, (table_name, schema) in enumerate(schemas.items(), 1):
            desc = table_descriptions.get(table_name, schema.get("description", ""))
            context_parts.append(f"{i}. {table_name} - {desc}")
            
            # Add column information
            columns = schema.get("columns", [])
            if columns:
                col_list = ", ".join([col["name"] for col in columns])
                context_parts.append(f"\n   Columns: {col_list}")
                
                # Add detailed column info with types
                context_parts.append(f"\n   Column details:")
                for col in columns:
                    col_type = col.get("field_type", "")
                    col_mode = col.get("mode", "NULLABLE")
                    col_desc = col.get("description", "")
                    detail = f"   - {col['name']} ({col_type}"
                    if col_mode != "NULLABLE":
                        detail += f", {col_mode}"
                    detail += ")"
                    if col_desc:
                        detail += f" - {col_desc}"
                    context_parts.append(detail)
            context_parts.append("")
        
        # Add join information
        context_parts.append("Important joins:")
        context_parts.append("- orders.user_id = users.id")
        context_parts.append("- orders.order_id = order_items.order_id")
        context_parts.append("- order_items.product_id = products.id")
        context_parts.append("")
        
        context_parts.append(
            "SECURITY: If the query references any other dataset or table that is NOT one of these 4 tables above, "
            "respond with: \"I can only answer questions about orders, order_items, products, and users data. "
            "Please clarify which dataset you're interested in.\""
        )
        context_parts.append("")
        context_parts.append("Generate clean, valid SQL queries only.")
        
        return "\n".join(context_parts)
    
    def _get_hardcoded_schema(self) -> str:
        """Fallback hardcoded schema context."""
        return """
CRITICAL: You can ONLY query these 4 tables in bigquery-public-data.thelook_ecommerce:

1. orders - customer orders with timestamps, status, and number of items
2. order_items - products within each order, including price and cost
3. products - catalog metadata like category, brand, and pricing
4. users - customer demographics like age, gender, and location

Available tables in bigquery-public-data.thelook_ecommerce:

1. orders (order_id, user_id, status, created_at, returned_at, shipped_at, delivered_at, num_of_item)
2. order_items (order_id, id, product_id, inventory_item_id, status, created_at, shipped_at, delivered_at, returned_at, sale_price, cost, sku)
3. products (id, cost, category, name, brand, retail_price, department, sku, distribution_center_id)
4. users (id, first_name, last_name, email, age, gender, state, street_address, postal_code, city, country, latitude, longitude, traffic_source, created_at)

Important joins:
- orders.user_id = users.id
- orders.order_id = order_items.order_id
- order_items.product_id = products.id

SECURITY: If the query references any other dataset or table that is NOT one of these 4 tables above, 
respond with: "I can only answer questions about orders, order_items, products, and users data. 
Please clarify which dataset you're interested in."

Generate clean, valid SQL queries only.
"""
    
    def _get_schema_context(self) -> str:
        """Get schema context (loads from file or uses fallback)."""
        return self._load_schema_context()
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Generate SQL query from user query or answer meta-question."""
        user_query = state.get("user_query", "")
        context = self._get_schema_context()
        
        # Check if this is a retry after a BigQuery error
        query_error = state.get("query_error")
        previous_sql = state.get("sql_query", "")
        retry_count = state.get("retry_count", 0)
        
        if query_error and previous_sql:
            # This is a retry - include the error and previous query in the prompt
            prompt = f"""
You are a SQL expert. A previous SQL query you generated failed with a BigQuery error. Please fix it.

{context}

Original user query: {user_query}

Previous SQL query that failed:
{previous_sql}

BigQuery error message:
{query_error}

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
- Always use LIMIT to restrict results (default 100 rows max)
- Use clear column aliases
- Return ONLY the fixed SQL query, no explanations

Fixed SQL Query:
"""
        else:
            # Initial query - determine if it's a meta-question or requires SQL
            # Check if this is a retry due to missing prefix
            retry_instruction = ""
            if retry_count > 0:
                retry_instruction = f"\n\n⚠️ ATTENTION: Previous response was invalid. Your response MUST start with either 'META:' or 'SQL:' - nothing else. This is attempt {retry_count + 1} of 3.\n"
            
            prompt = f"""
You are an intelligent data analysis assistant named Orion. Your role is to help users query and analyze e-commerce data.

{context}

User query: {user_query}
{retry_instruction}
ANALYZE THE QUERY:
Carefully determine what the user is asking:
- If they're asking about YOUR CAPABILITIES, WHAT datasets/tables/columns are AVAILABLE, or general HELP → This is a META question about you
- If they're asking about ACTUAL DATA (specific values, records, numbers, calculations from the database) → This needs a SQL query

EXAMPLES OF META QUESTIONS (answer directly, no SQL needed):
- "which dataset can you query?"
- "what tables are available?"
- "what columns are in the orders table?"
- "what can you do?"
- "help"
- "tell me about your capabilities"

EXAMPLES OF SQL QUESTIONS (generate SQL to query data):
- "what is the most expensive product?"
- "how many orders were placed?"
- "show me the top 10 customers"
- "what is the total revenue?"

RESPONSE FORMAT (CRITICAL - FOLLOW EXACTLY):
You MUST respond in one of two formats. Your response MUST start with one of these prefixes:

1. If META question (about capabilities/datasets/tables):
   Your response MUST start with: "META:"
   Example: "META: I can query the bigquery-public-data.thelook_ecommerce dataset..."
   
2. If SQL question (needs data from database):
   Your response MUST start with: "SQL:"
   Example: "SQL: SELECT * FROM \`bigquery-public-data.thelook_ecommerce.orders\` LIMIT 10"

CRITICAL: Your response MUST start with exactly "META:" or "SQL:" - no other text before it!

IMPORTANT: 
- For META questions, provide a helpful answer directly - do NOT generate SQL
- For SQL questions, provide ONLY the SQL query - no explanations or text before the SQL:

CRITICAL SQL RULES (only if responding with SQL):
- Use standard SQL syntax for BigQuery
- ALWAYS prefix ALL table names with the FULL path: 'bigquery-public-data.thelook_ecommerce.'
- IMPORTANT: For BigQuery public datasets, use backticks around the ENTIRE path, not each part
- Examples: `bigquery-public-data.thelook_ecommerce.order_items`
- Always use LIMIT to restrict results (default 100 rows max)
- Use clear column aliases

Your response (MUST start with META: or SQL:):
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )
            )
            
            # Handle potential None or empty response
            if not response or not hasattr(response, 'text'):
                return {
                    "query_error": "No response from Gemini. Please check your API key.",
                    "retry_count": state.get("retry_count", 0) + 1
                }
            
            response_text = response.text.strip()
            
            # Check for empty response
            if not response_text:
                return {
                    "query_error": "Gemini returned an empty response. Please rephrase your question.",
                    "retry_count": state.get("retry_count", 0) + 1
                }
            
            # Normalize response for checking
            response_upper = response_text.upper()
            retry_count = state.get("retry_count", 0)
            
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
                        return {
                            "query_error": f"Invalid response format: Response must start with 'META:' or 'SQL:'. Please try again with proper format.",
                            "retry_count": retry_count + 1
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
                    return {
                        "query_error": f"Invalid SQL generated: Query must use full table paths starting with 'bigquery-public-data.thelook_ecommerce.'. Generated query: {sql_query[:200]}",
                        "retry_count": state.get("retry_count", 0) + 1
                    }
            
            return {
                "sql_query": sql_query,
                "query_error": None  # Clear any previous errors
            }
        except Exception as e:
            error_str = str(e)
            retry_count = state.get("retry_count", 0)
            
            # Check for rate limit errors (429)
            if "429" in error_str or "Resource exhausted" in error_str or "rate limit" in error_str.lower():
                if retry_count < 3:
                    # Rate limit error - can retry after a delay
                    return {
                        "query_error": f"Rate limit exceeded. Retrying... (attempt {retry_count + 1}/3)",
                        "retry_count": retry_count + 1
                    }
                else:
                    # Max retries reached for rate limit
                    return {
                        "query_error": "Rate limit exceeded. Please wait a moment and try again later.",
                        "retry_count": retry_count
                    }
            
            # Other errors
            return {
                "query_error": f"Failed to generate SQL: {error_str}",
                "retry_count": retry_count + 1
            }


class BigQueryExecutorNode:
    """Executes SQL query on BigQuery."""
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        from google.cloud import bigquery
        
        sql_query = state.get("sql_query", "")
        
        if not sql_query:
            return {
                "query_error": "No SQL query to execute",
                "query_result": None
            }
        
        try:
            client = bigquery.Client(project=config.google_cloud_project)
            
            query_job = client.query(sql_query)
            df = query_job.to_dataframe(max_results=config.max_query_rows)
            
            return {
                "query_result": df,
                "query_error": None
            }
        except Exception as e:
            # Return error - graph will handle retry logic
            error_msg = str(e)
            # Extract the core error message (remove location/job ID if present)
            if "BigQuery execution error:" not in error_msg:
                error_msg = f"BigQuery execution error: {error_msg}"
            
            return {
                "query_error": error_msg,
                "query_result": None,
                "retry_count": state.get("retry_count", 0) + 1
            }


class OutputNode:
    """Formats and returns final output."""
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Format output for display."""
        # Check if final_output was already set (e.g., from META questions)
        existing_output = state.get("final_output", "")
        df = state.get("query_result")
        error = state.get("query_error")
        
        if existing_output and existing_output.strip():
            return {
                "final_output": existing_output
            }
        
        # Otherwise, format based on query results or errors
        
        if error:
            output = f"❌ Error: {error}"
        elif df is not None:
            if df.empty:
                output = "No results found."
            else:
                output = f"\n📊 Results:\n\n{df.to_string(index=False)}\n"
        else:
            output = "No results generated."
        
        return {
            "final_output": output
        }


# Create singleton instances for the graph
input_node = InputNode()
query_builder_node = QueryBuilderNode()
bigquery_executor_node = BigQueryExecutorNode()
output_node = OutputNode()

