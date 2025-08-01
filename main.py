"""
Multi-Agent Research and Summarization System
Main execution script with interactive interface and examples
"""

import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any
import json
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflow.multi_agent_workflow import MultiAgentWorkflow


def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    return api_key


def print_banner():
    """Print system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                Multi-Agent Research & Summarization System           ║
║                                                                      ║
║  🤖 Router Agent      - Intelligent query routing                   ║
║  🌐 Web Research      - Real-time information gathering             ║
║  📚 RAG Agent         - Knowledge base retrieval                    ║
║  📝 Summarization     - Multi-source synthesis                      ║
║  🧠 Memory System     - Conversational context                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_response(response: Dict[str, Any]):
    """Pretty print the system response"""
    print("\n" + "="*80)
    print("📋 QUERY ANALYSIS")
    print("="*80)
    print(f"Query Type: {response.get('query_type', 'Unknown')}")
    print(f"Routing Confidence: {response.get('routing_confidence', 0):.2f}")
    print(f"Overall Confidence: {response.get('confidence', 0)}%")
    
    if response.get('executive_summary'):
        print("\n" + "="*80)
        print("📊 EXECUTIVE SUMMARY")
        print("="*80)
        print(response['executive_summary'])
    
    print("\n" + "="*80)
    print("📝 DETAILED RESPONSE")
    print("="*80)
    print(response.get('response', 'No response'))
    
    if response.get('sources_used'):
        print("\n" + "="*80)
        print("📚 SOURCES USED")
        print("="*80)
        for i, source in enumerate(response['sources_used'], 1):
            print(f"{i}. {source.get('title', 'Unknown')} ({source.get('type', 'unknown')})")
            if source.get('url'):
                print(f"   URL: {source['url']}")
            if source.get('relevance'):
                print(f"   Relevance: {source['relevance']:.3f}")
    
    if response.get('errors'):
        print("\n" + "="*80)
        print("⚠️  ERRORS")
        print("="*80)
        for error in response['errors']:
            print(f"- {error}")
    
    print("\n" + "="*80)
    print("🔧 PROCESSING METADATA")
    print("="*80)
    print(f"Processing Steps: {', '.join(response.get('processing_steps', []))}")
    print(f"Session ID: {response.get('session_id', 'None')}")
    print(f"Timestamp: {response.get('timestamp', 'None')}")
    print(f"Success: {response.get('success', False)}")


def run_examples(workflow: MultiAgentWorkflow):
    """Run predefined examples to demonstrate system capabilities"""
    examples = [
        {
            "name": "Web Search Example",
            "query": "What are the latest developments in artificial intelligence in 2024?",
            "description": "Should trigger web research for current information"
        },
        {
            "name": "Knowledge Base Example", 
            "query": "What is machine learning and how does it work?",
            "description": "Should trigger RAG retrieval from knowledge base"
        },
        {
            "name": "Direct LLM Example",
            "query": "Solve this equation: 2x + 5 = 15. Show your work.",
            "description": "Should trigger direct LLM processing for math"
        },
        {
            "name": "Complex Analysis Example",
            "query": "Compare quantum computing with classical computing and explain the latest quantum breakthroughs",
            "description": "Should combine knowledge base and web search"
        }
    ]
    
    print("\n🎯 Running Example Queries")
    print("="*80)
    
    for i, example in enumerate(examples, 1):
        print(f"\n📋 Example {i}: {example['name']}")
        print(f"Description: {example['description']}")
        print(f"Query: {example['query']}")
        
        input("\nPress Enter to execute this example...")
        
        response = workflow.process_query(example['query'])
        print_response(response)
        
        print(f"\n✅ Example {i} completed")
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")


def interactive_mode(workflow: MultiAgentWorkflow):
    """Run interactive mode for user queries"""
    print("\n🎮 Interactive Mode")
    print("="*80)
    print("Enter your queries below. Type 'quit' to exit, 'help' for commands.")
    print("Special commands:")
    print("  - 'stats' : Show workflow statistics")
    print("  - 'reset' : Reset conversation session")
    print("  - 'sessions' : List available sessions")
    print("  - 'help' : Show this help message")
    print("  - 'quit' : Exit the system")
    
    session_id = None
    
    while True:
        try:
            print("\n" + "-"*80)
            query = input("🤔 Your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - 'stats' : Show workflow statistics")
                print("  - 'reset' : Reset conversation session")
                print("  - 'sessions' : List available sessions")
                print("  - 'help' : Show this help message")
                print("  - 'quit' : Exit the system")
                continue
            
            elif query.lower() == 'stats':
                stats = workflow.get_workflow_stats()
                print("\n📊 Workflow Statistics:")
                print(json.dumps(stats, indent=2, default=str))
                continue
            
            elif query.lower() == 'reset':
                workflow.reset_session()
                session_id = None
                print("✅ Session reset successfully")
                continue
            
            elif query.lower() == 'sessions':
                sessions = workflow.list_sessions()
                print(f"\n📁 Available Sessions ({len(sessions)}):")
                for session in sessions[:10]:  # Show last 10 sessions
                    print(f"  - {session['session_id']} ({session['turn_count']} turns)")
                continue
            
            # Process the query
            print("\n🔄 Processing your query...")
            response = workflow.process_query(query, session_id)
            
            # Update session ID if this is the first query
            if not session_id:
                session_id = response.get('session_id')
            
            print_response(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def save_example_results(workflow: MultiAgentWorkflow):
    """Save example results for documentation"""
    examples = [
        "What is quantum computing?",
        "Latest news about artificial intelligence",
        "Calculate the derivative of x^2 + 3x + 1"
    ]
    
    results = []
    
    for query in examples:
        print(f"Processing example: {query}")
        response = workflow.process_query(query)
        results.append({
            'query': query,
            'response': response
        })
    
    # Save to file
    with open('example_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("✅ Example results saved to example_results.json")


def main():
    """Main execution function"""
    print_banner()
    
    try:
        # Load environment
        print("🔧 Loading environment...")
        api_key = load_environment()
        
        # Initialize workflow
        print("🚀 Initializing multi-agent workflow...")
        workflow = MultiAgentWorkflow(api_key)
        
        print("✅ System initialized successfully!")
        
        # Show menu
        while True:
            print("\n" + "="*80)
            print("🎯 MAIN MENU")
            print("="*80)
            print("1. Run Example Queries")
            print("2. Interactive Mode")
            print("3. Save Example Results (for documentation)")
            print("4. Show System Statistics")
            print("5. Exit")
            
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == '1':
                run_examples(workflow)
            
            elif choice == '2':
                interactive_mode(workflow)
            
            elif choice == '3':
                save_example_results(workflow)
            
            elif choice == '4':
                stats = workflow.get_workflow_stats()
                print("\n📊 System Statistics:")
                print(json.dumps(stats, indent=2, default=str))
            
            elif choice == '5':
                print("👋 Thank you for using the Multi-Agent Research System!")
                break
            
            else:
                print("❌ Invalid option. Please select 1-5.")
    
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted. Goodbye!")
    
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()