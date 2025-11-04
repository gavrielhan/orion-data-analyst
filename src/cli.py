"""Command-line interface for Orion agent."""

import sys
import json
from pathlib import Path
from datetime import datetime
from src.agent.graph import OrionGraph
from src.config import config
from src.utils.visualizer import Visualizer
from src.utils.cache import QueryCache
from src.utils.formatter import OutputFormatter


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


def save_session(conversation_history: list, session_name: str = None):
    """Save conversation history to file."""
    sessions_dir = Path.home() / "Desktop" / "results" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    if not session_name:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    
    filepath = sessions_dir / f"{session_name}.json"
    
    with open(filepath, 'w') as f:
        json.dump(conversation_history, f, indent=2, default=str)
    
    return str(filepath)


def load_session(session_path: str) -> list:
    """Load conversation history from file."""
    try:
        with open(session_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load session: {e}")
        return []


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
    """Main CLI entry point with conversation memory and session management."""
    print_banner()
    validate_config()
    
    print(OutputFormatter.format(f"🔗 **Connected to:** {config.bigquery_dataset}"))
    print(OutputFormatter.format("💡 **Ask me anything about the e-commerce data!**"))
    print("   Commands: 'exit', 'save session', 'load session [path]', 'clear cache'\n")
    
    agent = OrionGraph()
    visualizer = Visualizer()
    cache = QueryCache()
    conversation_history = []
    
    while True:
        try:
            # Get user query
            user_query = input("\n You: ").strip()
            
            if not user_query:
                continue
            
            query_lower = user_query.lower()
            
            # Handle commands
            if query_lower in ["exit", "quit", "q"]:
                # Offer to save session
                if conversation_history:
                    save_prompt = input("💾 Save conversation? (yes/no): ").strip().lower()
                    if save_prompt in ["yes", "y"]:
                        filepath = save_session(conversation_history)
                        print(f"✅ Session saved to: {filepath}")
                print("\n👋 Goodbye!")
                break
            
            if query_lower == "save session":
                filepath = save_session(conversation_history)
                print(f"✅ Session saved to: {filepath}")
                continue
            
            if query_lower.startswith("load session "):
                session_path = user_query[13:].strip()
                conversation_history = load_session(session_path)
                print(OutputFormatter.success(f"Loaded {len(conversation_history)} previous interactions"))
                continue
            
            if query_lower == "clear cache":
                cache.clear()
                print(OutputFormatter.success("Cache cleared"))
                continue
            
            # Check cache first
            cached_result = cache.get(user_query)
            if cached_result:
                print(OutputFormatter.info("Using cached result (instant) ⚡"))
                result = cached_result
            else:
                # Execute agent with conversation context
                print(OutputFormatter.format("\n🤖 **Orion thinking...**"))
                result = agent.invoke(user_query, conversation_history)
                
                # Handle approval if needed
                requires_approval = result.get("requires_approval", False)
                approval_reason = result.get("approval_reason")
                
                if requires_approval and approval_reason:
                    print(OutputFormatter.warning(f"Approval Required: {approval_reason}"))
                    approval = input("Proceed? (yes/no): ").strip().lower()
                    
                    if approval not in ["yes", "y"]:
                        print(OutputFormatter.error("Query cancelled"))
                        continue
                    
                    print(OutputFormatter.format("\n🤖 **Executing approved query...**"))
                
                # Cache successful results
                if not result.get("query_error"):
                    cache.set(user_query, result)
            
            # Display output with beautiful formatting
            output = result.get("final_output", "No output generated")
            print(OutputFormatter.format(output))
            
            # Update conversation history (limit to last 5)
            df = result.get("query_result")
            result_summary = "No results" if df is None or len(df) == 0 else f"{len(df)} rows"
            conversation_history.append({
                "query": user_query,
                "result_summary": result_summary,
                "timestamp": datetime.now().isoformat()
            })
            if len(conversation_history) > 5:
                conversation_history = conversation_history[-5:]
            
            # Handle export options if there's data
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

