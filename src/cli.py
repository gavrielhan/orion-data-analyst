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


def main():
    """Main CLI entry point with visualization and export support."""
    print_banner()
    validate_config()
    
    print(f"🔗 Connected to: {config.bigquery_dataset}")
    print("💡 Ask me anything about the e-commerce data!")
    print("   (Type 'exit' or 'quit' to leave)\n")
    
    agent = OrionGraph()
    visualizer = Visualizer()
    last_result = None  # Store last result for viz/export commands
    
    while True:
        try:
            # Get user query
            user_query = input("\n You: ").strip()
            
            if not user_query:
                continue
            
            if user_query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            # Check if it's a visualization command
            query_lower = user_query.lower()
            if query_lower.startswith("chart ") and last_result:
                chart_type = query_lower.replace("chart ", "").strip()
                df = last_result.get("query_result")
                
                if df is not None and len(df) > 0:
                    print(f"\n📊 Creating {chart_type} chart...")
                    filepath = visualizer.create_chart(df, chart_type)
                    
                    if filepath:
                        print(f"✅ Chart saved to: {filepath}")
                    else:
                        print("❌ Failed to create chart. Check data format.")
                else:
                    print("❌ No data available for visualization.")
                continue
            
            # Check if it's a CSV export command
            if query_lower in ["save csv", "export csv", "csv"] and last_result:
                df = last_result.get("query_result")
                
                if df is not None and len(df) > 0:
                    print("\n💾 Exporting to CSV...")
                    filepath = visualizer.save_csv(df)
                    print(f"✅ CSV saved to: {filepath}")
                else:
                    print("❌ No data available to export.")
                continue
            
            # Execute agent for regular queries
            print("\n🤖 Orion thinking...")
            result = agent.invoke(user_query)
            last_result = result  # Save for viz/export commands
            
            # Display output
            print(result.get("final_output", "No output generated"))
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    main()

