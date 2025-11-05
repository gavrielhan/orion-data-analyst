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
║      ██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗                       ║
║     ██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║                       ║
║     ██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║                       ║
║     ██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║                       ║
║     ╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║                       ║
║      ╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝                       ║
║                                                                   ║
║                 Data Analysis Agent 🚀                            ║
║         AI-Powered BigQuery Intelligence Platform                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

    """
    print(banner)


def validate_config():
    """Validate configuration and provide helpful setup instructions."""
    from pathlib import Path
    
    # Check if .env file exists
    env_file = Path.cwd() / ".env"
    env_in_parent = Path(__file__).parent.parent / ".env"
    
    if not env_file.exists() and not env_in_parent.exists():
        print(OutputFormatter.error("Configuration file not found!"))
        print("\n📝 You need to create a .env file with your API keys.")
        print("\n" + "─" * 60)
        print("Create a file named '.env' in the project directory with:\n")
        print("# Google Cloud Configuration")
        print("GOOGLE_CLOUD_PROJECT=your-project-id")
        print("GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json")
        print("")
        print("# Gemini AI API Key")
        print("GEMINI_API_KEY=your-gemini-api-key-here")
        print("")
        print("# Optional: Gemini Model (default: gemini-2.0-flash-exp)")
        print("GEMINI_MODEL=gemini-2.0-flash-exp")
        print("")
        print("# Optional: Output Directory (default: ~/orion_results)")
        print("# ORION_OUTPUT_DIR=/path/to/your/output")
        print("")
        print("# Optional BigQuery Settings")
        print("BIGQUERY_DATASET=bigquery-public-data.thelook_ecommerce")
        print("─" * 60)
        print("\n💡 Get your Gemini API key: https://makersuite.google.com/app/apikey")
        print("💡 Get Google Cloud credentials: https://console.cloud.google.com/")
        sys.exit(1)
    
    # Validate required variables
    missing = config.validate()
    if missing:
        print(OutputFormatter.error("Missing required configuration!"))
        print(f"\n❌ Missing: {', '.join(missing)}")
        print("\n📝 Update your .env file with:")
        for var in missing:
            if "GEMINI" in var:
                print(f"   {var}=your-gemini-api-key")
                print(f"   → Get it at: https://makersuite.google.com/app/apikey")
            elif "PROJECT" in var:
                print(f"   {var}=your-gcp-project-id")
                print(f"   → Find it at: https://console.cloud.google.com/")
            elif "CREDENTIALS" in var:
                print(f"   {var}=/path/to/your-service-account.json")
                print(f"   → Create at: https://console.cloud.google.com/iam-admin/serviceaccounts")
        sys.exit(1)


def save_session(conversation_history: list, session_name: str = None):
    """Save conversation history to file."""
    sessions_dir = Path(config.output_directory) / "sessions"
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


def _generate_viz_suggestion_lazy(user_query, df, analysis_type="aggregation"):
    """Generate visualization suggestion on-demand (lazy, only when needed)."""
    try:
        from src.agent.nodes import InsightGeneratorNode
        insight_gen = InsightGeneratorNode()
        return insight_gen._suggest_visualization(user_query, df, analysis_type)
    except Exception as e:
        # Log error but don't fail the entire flow - visualization suggestion is optional
        # Only log if verbose mode is enabled (check if we can determine this)
        import logging
        logging.getLogger(__name__).debug(f"Failed to generate visualization suggestion: {str(e)}")
        return None


def handle_export_options(df, visualizer, user_query_lower, result=None):
    """
    Handle export options sequentially, generating visualization suggestions lazily.
    Returns True if exports were requested in the original query.
    """
    # Validate df is a DataFrame
    import pandas as pd
    if df is None:
        return False
    if not isinstance(df, pd.DataFrame):
        print(OutputFormatter.error(f"❌ Invalid data type: expected DataFrame, got {type(df).__name__}"))
        return False
    if len(df) == 0:
        return False
    
    # NOTE: visualization_suggestion is NO LONGER auto-generated
    # We generate it lazily only when user wants a chart (saves 25% of LLM calls)
    
    # Check if user already specified exports in their query
    wants_csv = any(kw in user_query_lower for kw in ["save csv", "export csv", "as csv", "to csv"])
    wants_chart = "chart" in user_query_lower
    chart_type = None
    x_col = None
    y_col = None
    title = None
    hue_col = None
    
    # Extract chart type if specified
    if wants_chart:
        for ctype in ["bar", "line", "pie", "scatter", "box"]:
            if ctype in user_query_lower:
                chart_type = ctype
                break
        
        # Generate visualization suggestion ONLY when user wants a chart
        # This is lazy evaluation - saves 1 LLM call per query if no chart needed
        user_query = result.get("user_query", "") if result else ""
        analysis_type = result.get("analysis_type", "aggregation") if result else "aggregation"
        viz_suggestion = _generate_viz_suggestion_lazy(user_query, df, analysis_type)
        
        if not chart_type and viz_suggestion:
            # Use LLM suggestion
            chart_type = viz_suggestion.get("chart_type", "bar")
            x_col = viz_suggestion.get("x_col")
            y_col = viz_suggestion.get("y_col")
            title = viz_suggestion.get("title")
            hue_col = viz_suggestion.get("hue_col")
        elif not chart_type:
            chart_type = "bar"  # Fallback default
    
    # If already specified, handle immediately
    if wants_csv or wants_chart:
        if wants_csv:
            print("\n💾 Exporting to CSV...")
            filepath = visualizer.save_csv(df)
            print(f"✅ CSV saved to: {filepath}")
        
        if wants_chart:
            if viz_suggestion:
                suggestion_text = f"{chart_type} chart with {x_col or 'auto'} (x) vs {y_col or 'auto'} (y)"
                if hue_col:
                    suggestion_text += f", grouped by {hue_col}"
                print(OutputFormatter.info(f"💡 Gemini suggests: {suggestion_text}"))
            print(f"\n📊 Creating {chart_type} chart...")
            filepath, error_msg = visualizer.create_chart(df, chart_type, x_col, y_col, title, hue_col)
            if filepath:
                print(f"✅ Chart saved to: {filepath}")
            else:
                print(OutputFormatter.error(f"❌ Chart creation failed: {error_msg or 'Unknown error'}"))
        
        return True
    
    # Otherwise, ask sequentially
    # Ask about CSV first
    csv_response = input("\n💾 Would you like to save the results as CSV? (type 'save csv' or 'no'): ").strip().lower()
    
    if csv_response in ["save csv", "yes", "y", "csv"]:
        print("\n💾 Exporting to CSV...")
        filepath = visualizer.save_csv(df)
        print(f"✅ CSV saved to: {filepath}")
    
    # Ask about chart second - ONLY generate suggestion AFTER user says they want a chart
    chart_prompt = "\n📊 Would you like to create a chart? (type 'chart [type]' or 'no')\n    Types: bar, line, pie, scatter, box\n    → "
    chart_response = input(chart_prompt).strip().lower()
    
    # Only generate suggestion AFTER user says they want a chart (saves API calls)
    if chart_response not in ["no", "n", ""]:
        user_query = result.get("user_query", "") if result else ""
        analysis_type = result.get("analysis_type", "aggregation") if result else "aggregation"
        viz_suggestion = _generate_viz_suggestion_lazy(user_query, df, analysis_type)
        
        if viz_suggestion:
            suggested_type = viz_suggestion.get("chart_type", "bar")
            suggested_x = viz_suggestion.get("x_col", "")
            suggested_y = viz_suggestion.get("y_col", "")
            suggested_hue = viz_suggestion.get("hue_col")
            suggestion_text = f"{suggested_type} chart ({suggested_x} vs {suggested_y}"
            if suggested_hue:
                suggestion_text += f", grouped by {suggested_hue}"
            suggestion_text += ")"
            print(OutputFormatter.info(f"💡 Gemini suggests: {suggestion_text}"))
    
    if chart_response.startswith("chart "):
        # User specified a type manually
        chart_type = chart_response.replace("chart ", "").strip()
        print(f"\n📊 Creating {chart_type} chart...")
        filepath, error_msg = visualizer.create_chart(df, chart_type)
        if filepath:
            print(f"✅ Chart saved to: {filepath}")
        else:
            print(OutputFormatter.error(f"❌ Chart creation failed: {error_msg or 'Unknown error'}"))
    elif chart_response in ["yes", "y"]:
        # Use LLM suggestion if available, otherwise default
        if viz_suggestion:
            chart_type = viz_suggestion.get("chart_type", "bar")
            x_col = viz_suggestion.get("x_col")
            y_col = viz_suggestion.get("y_col")
            title = viz_suggestion.get("title")
            hue_col = viz_suggestion.get("hue_col")
            print(f"\n📊 Creating {chart_type} chart with Gemini's suggestions...")
            filepath, error_msg = visualizer.create_chart(df, chart_type, x_col, y_col, title, hue_col)
        else:
            chart_type = "bar"
            print(f"\n📊 Creating {chart_type} chart...")
            filepath, error_msg = visualizer.create_chart(df, chart_type)
        
        if filepath:
            print(f"✅ Chart saved to: {filepath}")
        else:
            print(OutputFormatter.error(f"❌ Chart creation failed: {error_msg or 'Unknown error'}"))
    elif any(ct in chart_response for ct in ["bar", "line", "pie", "scatter", "box", "candle"]):
        # Try to extract chart type from response
        chart_type = "bar"  # Default
        for ctype in ["bar", "line", "pie", "scatter", "box"]:
            if ctype in chart_response:
                chart_type = ctype
                break
        
        print(f"\n📊 Creating {chart_type} chart...")
        filepath, error_msg = visualizer.create_chart(df, chart_type)
        if filepath:
            print(f"✅ Chart saved to: {filepath}")
        else:
            print(OutputFormatter.error(f"❌ Chart creation failed: {error_msg or 'Unknown error'}"))
    
    return False


def _is_chart_customization_query(current_query: str, previous_query: str) -> bool:
    """
    Use Gemini to determine if current query is asking to re-visualize previous data.
    Returns True if it's a chart customization, False if it's a new data query.
    """
    try:
        import google.generativeai as genai
        from src.config import config
        from src.utils.rate_limiter import get_global_rate_limiter
        
        # Use shared global rate limiter
        rate_limiter = get_global_rate_limiter()
        rate_limiter.wait_if_needed()
        
        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel(config.gemini_model)
        
        prompt = f"""Analyze if the current query is asking to visualize the SAME data from the previous query in a different way.

Previous query: {previous_query}
Current query: {current_query}

Is the current query:
A) Requesting a NEW data query (different data, filters, time periods, metrics)
B) Requesting to RE-VISUALIZE the same data with different chart layout/specifications

Examples of B (chart customization):
- Previous: "show female and male counts per year"
  Current: "make bar chart with year on x-axis" → B (same data, different layout)
- Previous: "sales by product"
  Current: "show it as a pie chart" → B (same data, different chart type)
- Previous: "top customers"
  Current: "group by region" → B (same data, add grouping)

Examples of A (new data query):
- Previous: "show sales for 2023"
  Current: "show sales for 2024" → A (different time period)
- Previous: "female and male counts"
  Current: "show orders by status" → A (completely different data)
- Previous: "top products"
  Current: "bottom products" → A (different metric/filter)

Respond with ONLY one letter: A or B"""

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=10,
            )
        )
        
        if response and hasattr(response, 'text'):
            answer = response.text.strip().upper()
            return 'B' in answer
        
        return False
    except Exception as e:
        # If rate limit or LLM fails, assume it's a new query (safer default)
        error_msg = str(e).lower()
        if 'rate limit' in error_msg or '429' in error_msg or 'quota' in error_msg:
            print(OutputFormatter.warning("⚠️  Rate limit reached, treating as new query"))
        return False


def main():
    """Main CLI entry point with conversation memory and session management."""
    print_banner()
    
    # Validate configuration with helpful error messages
    validate_config()
    
    # Show successful connection
    print(OutputFormatter.success(f"Connected to: {config.bigquery_dataset}"))
    print(OutputFormatter.info(f"Using Gemini model: {config.gemini_model}"))
    print(OutputFormatter.info(f"Results directory: {config.output_directory}"))
    print(OutputFormatter.format("💡 **Ask me anything about the e-commerce data!**"))
    print("   Commands: 'exit', 'save session', 'load session [path]', 'clear cache', 'rate limit status', 'reset rate limiter'")
    print(OutputFormatter.format("   💾 Tip: Results are cached for faster repeated queries\n"))
    
    agent = OrionGraph()
    visualizer = Visualizer()
    cache = QueryCache()
    conversation_history = []
    last_result = None  # Store last result for chart re-generation
    
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
            
            if query_lower == "rate limit status":
                from src.utils.rate_limiter import get_global_rate_limiter
                rate_limiter = get_global_rate_limiter()
                status = rate_limiter.get_status()
                print(f"📊 Rate Limiter Status:")
                print(f"   Calls used: {status['current_calls']}/{status['max_calls']}")
                print(f"   Window: {status['window_seconds']} seconds")
                print(f"   Can proceed: {'Yes' if status['can_proceed'] else 'No (waiting)'}")
                continue
            
            if query_lower == "reset rate limiter":
                from src.utils.rate_limiter import get_global_rate_limiter
                rate_limiter = get_global_rate_limiter()
                rate_limiter.reset()
                print(OutputFormatter.success("Rate limiter reset - ready for new calls"))
                continue
            
            # Use LLM to detect if this is a chart customization request for previous data
            # Only check if user query contains chart-related keywords (saves API calls)
            is_chart_query = False
            chart_keywords = ["chart", "graph", "plot", "visualize", "bar", "line", "pie", "scatter", "box"]
            if last_result is not None and last_result.get("query_result") is not None:
                # Only call LLM if query contains chart keywords (saves unnecessary API calls)
                if any(kw in user_query.lower() for kw in chart_keywords):
                    is_chart_query = _is_chart_customization_query(user_query, last_result.get("user_query", ""))
            
            if is_chart_query:
                # Regenerate visualization suggestion with new specifications
                print(OutputFormatter.info("🎨 Creating custom chart from previous data..."))
                df = last_result.get("query_result")
                analysis_type = last_result.get("analysis_type", "aggregation")
                
                # Import InsightGeneratorNode to regenerate visualization
                from src.agent.nodes import InsightGeneratorNode
                insight_gen = InsightGeneratorNode()
                viz_suggestion = insight_gen._suggest_visualization(user_query, df, analysis_type)
                
                # Use the same result but with new visualization suggestion
                result = last_result.copy()
                result["visualization_suggestion"] = viz_suggestion
                print(OutputFormatter.success("✓ Chart specification updated"))
            else:
                # Check cache first
                cached_result = cache.get(user_query)
                if cached_result:
                    print(OutputFormatter.info("Using cached result (instant) ⚡"))
                    result = cached_result
                else:
                    # Execute agent with conversation context and show progress
                    print(OutputFormatter.format("\n🤖 **Orion working...**"))
                    result = agent.invoke(user_query, conversation_history, verbose=True)

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
            
            # Display output with beautiful formatting (skip for chart-only queries)
            if not is_chart_query:
                output = result.get("final_output", "No output generated")
                print(OutputFormatter.format(output))
            
            # Store last result for potential chart re-generation (with original query)
            if result.get("query_result") is not None:
                last_result = result.copy()
                last_result["user_query"] = user_query
            
            # Update conversation history (limit to last 5)
            df = result.get("query_result")
            # Validate df is a DataFrame (not corrupted from cache)
            import pandas as pd
            if df is not None and not isinstance(df, pd.DataFrame):
                print(OutputFormatter.error(f"❌ Data corruption detected: query_result is {type(df).__name__}, expected DataFrame. Clearing cache..."))
                cache.clear()
                # Skip this result
                continue
            result_summary = "No results" if df is None or len(df) == 0 else f"{len(df)} rows"
            conversation_history.append({
                "query": user_query,
                "result_summary": result_summary,
                "timestamp": datetime.now().isoformat()
            })
            if len(conversation_history) > 5:
                conversation_history = conversation_history[-5:]
            
            # Handle export options if there's data (for chart-only queries, go directly to chart)
            if df is not None and len(df) > 0:
                if is_chart_query:
                    # For chart-only queries, directly create the chart
                    viz_suggestion = result.get("visualization_suggestion")
                    if viz_suggestion:
                        chart_type = viz_suggestion.get("chart_type", "bar")
                        x_col = viz_suggestion.get("x_col")
                        y_col = viz_suggestion.get("y_col")
                        title = viz_suggestion.get("title")
                        hue_col = viz_suggestion.get("hue_col")
                        
                        print(f"\n📊 Creating {chart_type} chart with your specifications...")
                        filepath, error_msg = visualizer.create_chart(df, chart_type, x_col, y_col, title, hue_col)
                        if filepath:
                            print(f"✅ Chart saved to: {filepath}")
                        else:
                            print(OutputFormatter.error(f"❌ Chart creation failed: {error_msg or 'Unknown error'}"))
                    else:
                        print(OutputFormatter.error("❌ Could not generate chart specification from your request"))
                else:
                    handle_export_options(df, visualizer, user_query.lower(), result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    main()

