"""Node implementations for Orion agent."""

import pandas as pd
from typing import Dict, Any
from vertexai.generative_models import GenerativeModel
import vertexai

from src.config import config
from src.agent.state import AgentState


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
    """Generates SQL query using Gemini via Vertex AI."""
    
    def __init__(self):
        # Initialize Vertex AI
        vertexai.init(
            project=config.google_cloud_project, 
            location=config.vertex_ai_location
        )
        # Use Gemini 1.5 Flash for faster responses
        self.model = GenerativeModel("gemini-1.5-flash")
        
    def _get_schema_context(self) -> str:
        """Provide hardcoded schema context for MVP."""
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
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Generate SQL query from user query."""
        user_query = state.get("user_query", "")
        context = self._get_schema_context()
        
        prompt = f"""
You are a SQL expert. Generate a SQL query for BigQuery based on the user's request.

{context}

User query: {user_query}

Rules:
- Use standard SQL syntax for BigQuery
- Prefix all table names with 'bigquery-public-data.thelook_ecommerce.'
- Always use LIMIT to restrict results (default 100 rows max)
- Use clear column aliases
- Return ONLY the SQL query, no explanations
- If query is about datasets not in the allowed list, return ERROR: followed by the error message

SQL Query:
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 1000,
                }
            )
            sql_query = response.text.strip()
            
            # Clean SQL - remove markdown code blocks if present
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.startswith("```"):
                sql_query = sql_query[3:]
            sql_query = sql_query.strip().rstrip("`")
            
            # Check if the LLM detected an invalid dataset
            if sql_query.startswith("ERROR:"):
                error_message = sql_query.replace("ERROR:", "").strip()
                return {
                    "final_output": error_message
                }
            
            return {
                "sql_query": sql_query
            }
        except Exception as e:
            return {
                "query_error": f"Failed to generate SQL: {str(e)}",
                "retry_count": state.get("retry_count", 0) + 1
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
            return {
                "query_error": f"BigQuery execution error: {str(e)}",
                "query_result": None,
                "retry_count": state.get("retry_count", 0) + 1
            }


class OutputNode:
    """Formats and returns final output."""
    
    @staticmethod
    def execute(state: AgentState) -> Dict[str, Any]:
        """Format output for display."""
        df = state.get("query_result")
        error = state.get("query_error")
        
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

