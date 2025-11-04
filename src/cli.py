"""Command-line interface for Orion agent."""

import sys
from src.agent.graph import OrionGraph
from src.config import config
from src.utils.visualizer import Visualizer


def print_banner():
    """Print Orion banner."""
    banner = """
    
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║      ██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗                      ║
║     ██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║                      ║
║     ██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║                      ║
║     ██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║                      ║
║     ╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║                      ║
║      ╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝                      ║
║                                                                   ║
║                 Data Analysis Agent 🚀                            ║
║         AI-Powered BigQuery Intelligence Platform                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

    """
    print(banner)


def validate_config():
    """Validate configuration and exit if missing."""
    missing = config.validate()
    if missing:
        print("❌ Configuration Error:")
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("\nPlease set these in your .env file (see .env.example)")
        sys.exit(1)


def handle_export_options(df, visualizer, user_query_lower):
    """
    Handle export options sequentially.
    Returns True if exports were requested in the original query.
    """
    if df is None or len(df) == 0:
        return False
    
    # Check if user already specified exports in their query
    wants_csv = any(kw in user_query_lower for kw in ["save csv", "export csv", "as csv", "to csv"])
    wants_chart = "chart" in user_query_lower
    chart_type = None
    
    # Extract chart type if specified
    if wants_chart:
        for ctype in ["bar", "line", "pie", "scatter", "box", "candle"]:
            if ctype in user_query_lower:
                chart_type = ctype
                break
        if not chart_type:
            chart_type = "bar"  # Default
    
    # If already specified, handle immediately
    if wants_csv or wants_chart:
        if wants_csv:
            print("\n💾 Exporting to CSV...")
            filepath = visualizer.save_csv(df)
            print(f"✅ CSV saved to: {filepath}")
        
        if wants_chart:
            print(f"\n📊 Creating {chart_type} chart...")
            filepath = visualizer.create_chart(df, chart_type)
            if filepath:
                print(f"✅ Chart saved to: {filepath}")
            else:
                print("❌ Failed to create chart.")
        
        return True
    
    # Otherwise, ask sequentially
    # Ask about CSV first
    csv_response = input("\n💾 Would you like to save the results as CSV? (type 'save csv' or 'no'): ").strip().lower()
    
    if csv_response in ["save csv", "yes", "y", "csv"]:
        print("\n💾 Exporting to CSV...")
        filepath = visualizer.save_csv(df)
        print(f"✅ CSV saved to: {filepath}")
    
    # Ask about chart second
    chart_response = input("\n📊 Would you like to create a chart? (type 'chart [type]' or 'no')\n    Types: bar, line, pie, scatter, box, candle\n    → ").strip().lower()
    
    if chart_response.startswith("chart "):
        chart_type = chart_response.replace("chart ", "").strip()
        print(f"\n📊 Creating {chart_type} chart...")
        filepath = visualizer.create_chart(df, chart_type)
        if filepath:
            print(f"✅ Chart saved to: {filepath}")
        else:
            print("❌ Failed to create chart.")
    elif chart_response in ["yes", "y"] or any(ct in chart_response for ct in ["bar", "line", "pie", "scatter", "box", "candle"]):
        # Try to extract chart type from response
        chart_type = "bar"  # Default
        for ctype in ["bar", "line", "pie", "scatter", "box", "candle"]:
            if ctype in chart_response:
                chart_type = ctype
                break
        
        print(f"\n📊 Creating {chart_type} chart...")
        filepath = visualizer.create_chart(df, chart_type)
        if filepath:
            print(f"✅ Chart saved to: {filepath}")
        else:
            print("❌ Failed to create chart.")
    
    return False


def main():
    """Main CLI entry point with visualization and export support."""
    print_banner()
    validate_config()
    
    print(f"🔗 Connected to: {config.bigquery_dataset}")
    print("💡 Ask me anything about the e-commerce data!")
    print("   (Type 'exit' or 'quit' to leave)\n")
    
    agent = OrionGraph()
    visualizer = Visualizer()
    
    while True:
        try:
            # Get user query
            user_query = input("\n You: ").strip()
            
            if not user_query:
                continue
            
            if user_query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            # Execute agent for queries
            print("\n🤖 Orion thinking...")
            result = agent.invoke(user_query)
            
            # Display output
            print(result.get("final_output", "No output generated"))
            
            # Handle export options if there's data
            df = result.get("query_result")
            if df is not None and len(df) > 0:
                handle_export_options(df, visualizer, user_query.lower())
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    main()

