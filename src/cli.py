"""Command-line interface for Orion agent."""

import sys
from src.agent.graph import OrionGraph
from src.config import config


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
    """Main CLI entry point."""
    print_banner()
    validate_config()
    
    print(f"🔗 Connected to: {config.bigquery_dataset}")
    print("💡 Ask me anything about the e-commerce data!")
    print("   (Type 'exit' or 'quit' to leave)\n")
    
    agent = OrionGraph()
    
    while True:
        try:
            # Get user query
            user_query = input("\n❓ You: ").strip()
            
            if not user_query:
                continue
            
            if user_query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            # Execute agent
            print("\n🤖 Orion thinking...")
            result = agent.invoke(user_query)
            
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

