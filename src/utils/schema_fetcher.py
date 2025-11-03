"""Utility script to fetch and save BigQuery table schemas."""

import json
from pathlib import Path
from google.cloud import bigquery
from src.config import config


def fetch_table_schema(table_name: str) -> dict:
    """Fetch schema information for a BigQuery table."""
    client = bigquery.Client(project=config.google_cloud_project)
    
    # Get table reference
    table_ref = client.get_table(f"bigquery-public-data.thelook_ecommerce.{table_name}")
    
    # Extract schema information
    schema_info = {
        "table_name": table_name,
        "description": table_ref.description or f"{table_name} table",
        "columns": []
    }
    
    for field in table_ref.schema:
        column_info = {
            "name": field.name,
            "field_type": field.field_type,
            "mode": field.mode or "NULLABLE",
            "description": field.description or ""
        }
        schema_info["columns"].append(column_info)
    
    return schema_info


def fetch_all_schemas() -> dict:
    """Fetch schema information for all tables in the dataset."""
    tables = ["orders", "order_items", "products", "users"]
    schemas = {}
    
    print("Fetching schema information from BigQuery...")
    for table in tables:
        print(f"  Fetching {table}...")
        try:
            schemas[table] = fetch_table_schema(table)
            print(f"    ✓ Found {len(schemas[table]['columns'])} columns")
        except Exception as e:
            print(f"    ✗ Error fetching {table}: {e}")
    
    return schemas


def save_schemas(schemas: dict, output_path: Path) -> None:
    """Save schema information to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(schemas, f, indent=2)
    print(f"\n✓ Schema information saved to {output_path}")


def format_schema_for_context(schemas: dict) -> str:
    """Format schema information for use in LLM context."""
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
            col_details = []
            for col in columns:
                col_type = col.get("field_type", "")
                col_mode = col.get("mode", "NULLABLE")
                col_desc = col.get("description", "")
                detail = f"{col['name']} ({col_type}"
                if col_mode != "NULLABLE":
                    detail += f", {col_mode}"
                detail += ")"
                if col_desc:
                    detail += f" - {col_desc}"
                col_details.append(detail)
            
            context_parts.append(f"\n   Column details:")
            for detail in col_details:
                context_parts.append(f"   - {detail}")
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


def main():
    """Main function to fetch and save schemas."""
    # Fetch schemas from BigQuery
    schemas = fetch_all_schemas()
    
    # Save to JSON file
    schema_dir = Path(__file__).parent.parent.parent
    schema_file = schema_dir / "schemas.json"
    save_schemas(schemas, schema_file)
    
    # Also save formatted context
    context_file = schema_dir / "schema_context.txt"
    formatted_context = format_schema_for_context(schemas)
    with open(context_file, 'w') as f:
        f.write(formatted_context)
    print(f"✓ Formatted context saved to {context_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Schema Summary:")
    print("="*60)
    for table_name, schema in schemas.items():
        print(f"\n{table_name.upper()}:")
        print(f"  {len(schema['columns'])} columns")
        for col in schema['columns']:
            print(f"    - {col['name']}: {col['field_type']} ({col['mode']})")
    
    return schemas


if __name__ == "__main__":
    main()

