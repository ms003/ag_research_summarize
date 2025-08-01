#!/usr/bin/env python3
"""Quick test of the simplified workflow"""

import os
import sys

# Test imports
try:
    from simple_workflow import SimpleMultiAgentWorkflow, print_response
    print("✅ Successfully imported simplified workflow")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_workflow():
    """Test the basic workflow functionality"""
    print("🧪 Testing Simple Multi-Agent Workflow...")
    
    # Initialize workflow
    workflow = SimpleMultiAgentWorkflow()
    
    # Test queries
    test_queries = [
        "What is machine learning?",  # Should route to RAG
        "Latest news about AI",       # Should route to web search  
        "Calculate 15 + 25",         # Should route to direct LLM
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        try:
            result = workflow.process_query(query)
            print(f"✅ Route: {result.get('query_type')}")
            print(f"✅ Confidence: {result.get('confidence')}%")
            print(f"✅ Success: {result.get('success')}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎉 Basic workflow test completed!")

if __name__ == "__main__":
    test_workflow()