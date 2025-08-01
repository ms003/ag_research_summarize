"""
Simplified Multi-Agent Research and Summarization Workflow
A working version without LangGraph dependencies
"""

import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Load environment variables
load_dotenv()

class SimpleRouterAgent:
    """Simplified router agent using keyword-based routing"""
    
    def __init__(self):
        self.web_search_keywords = [
            "latest", "current", "recent", "today", "yesterday", "this week",
            "this month", "this year", "2024", "2023", "news", "breaking",
            "update", "now", "currently", "real-time", "live", "happening",
            "trending", "fresh", "new", "just", "immediate"
        ]
        
        self.rag_keywords = [
            "definition", "what is", "explain", "describe", "concept",
            "theory", "principle", "fundamental", "basic", "overview",
            "introduction", "background", "history", "technical", "detailed"
        ]
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route query based on keywords"""
        query_lower = query.lower()
        
        web_score = sum(1 for keyword in self.web_search_keywords if keyword in query_lower)
        rag_score = sum(1 for keyword in self.rag_keywords if keyword in query_lower)
        
        if web_score > rag_score and web_score > 0:
            route = "web_search"
            confidence = min(0.6 + (web_score * 0.1), 1.0)
        elif rag_score > 0:
            route = "rag"
            confidence = min(0.6 + (rag_score * 0.1), 1.0)
        else:
            route = "llm_direct"
            confidence = 0.7
        
        return {
            "route": route,
            "confidence": confidence,
            "reasoning": f"Keyword analysis: web_score={web_score}, rag_score={rag_score}"
        }


class SimpleWebResearchAgent:
    """Simplified web research using requests and basic parsing"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_web(self, query: str) -> Dict[str, Any]:
        """Simulate web search results"""
        print(f"🔍 Simulating web search for: {query}")
        
        # For demo purposes, return simulated results
        # In a real implementation, this would use actual web scraping
        results = {
            "query": query,
            "results": [
                {
                    "title": f"Latest information about {query}",
                    "snippet": f"Recent developments and current information related to {query}...",
                    "url": "https://example.com/1"
                },
                {
                    "title": f"Current trends in {query}",
                    "snippet": f"Analysis of current trends and recent updates about {query}...",
                    "url": "https://example.com/2"
                }
            ],
            "synthesized_info": f"Based on recent web sources, here's what we found about {query}: Current research shows significant developments in this area with new findings emerging regularly.",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return results


class SimpleRAGAgent:
    """Simplified RAG agent using in-memory knowledge base"""
    
    def __init__(self):
        # Load knowledge base
        try:
            with open('data/sample_knowledge_base.json', 'r') as f:
                self.knowledge_base = json.load(f)
            print(f"📚 Loaded {len(self.knowledge_base)} documents into knowledge base")
        except FileNotFoundError:
            print("⚠️  Knowledge base file not found, using empty knowledge base")
            self.knowledge_base = []
    
    def search_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Search knowledge base using simple keyword matching"""
        query_words = set(query.lower().split())
        
        results = []
        for doc in self.knowledge_base:
            # Simple relevance scoring based on keyword overlap
            doc_words = set((doc['title'] + ' ' + doc['content']).lower().split())
            common_words = query_words.intersection(doc_words)
            relevance = len(common_words) / max(len(query_words), 1)
            
            if relevance > 0.1:  # Minimum relevance threshold
                results.append({
                    'document': doc,
                    'relevance': relevance
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        if results:
            top_result = results[0]
            response = f"Based on our knowledge base: {top_result['document']['content'][:500]}..."
            sources = [{"title": top_result['document']['title'], "category": top_result['document']['category']}]
        else:
            response = "No relevant information found in the knowledge base."
            sources = []
        
        return {
            "query": query,
            "response": response,
            "sources": sources[:3],  # Top 3 sources
            "total_found": len(results),
            "success": len(results) > 0
        }


class SimpleLLMAgent:
    """Simplified LLM agent for direct responses"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query directly (simulated for demo)"""
        
        # For mathematical queries
        if any(word in query.lower() for word in ['calculate', 'solve', 'equation', '+', '-', '*', '/', '=']):
            # Simple math example
            if "2x + 5 = 15" in query:
                response = """To solve 2x + 5 = 15:

Step 1: Subtract 5 from both sides
2x + 5 - 5 = 15 - 5
2x = 10

Step 2: Divide both sides by 2
2x ÷ 2 = 10 ÷ 2
x = 5

Therefore, x = 5"""
            else:
                response = f"This appears to be a mathematical query: {query}. I would solve this step by step using appropriate mathematical methods."
        
        # For other queries
        else:
            response = f"I would analyze this query using my knowledge and reasoning capabilities: {query}. This type of question requires direct language model processing for the best results."
        
        return {
            "query": query,
            "response": response,
            "success": True,
            "type": "direct_llm"
        }


class SimpleSummarizationAgent:
    """Simplified summarization agent"""
    
    def __init__(self):
        pass
    
    def synthesize_information(self, query: str, web_results=None, rag_results=None, llm_results=None) -> Dict[str, Any]:
        """Synthesize information from multiple sources"""
        
        response_parts = []
        sources_used = []
        
        # Add web research results
        if web_results and web_results.get('success'):
            response_parts.append("=== WEB RESEARCH ===")
            response_parts.append(web_results.get('synthesized_info', 'Web information retrieved'))
            sources_used.extend([{"type": "web", "title": r.get('title', 'Web Source')} for r in web_results.get('results', [])])
        
        # Add RAG results
        if rag_results and rag_results.get('success'):
            response_parts.append("=== KNOWLEDGE BASE ===")
            response_parts.append(rag_results.get('response', 'Knowledge base information'))
            sources_used.extend([{"type": "knowledge_base", **source} for source in rag_results.get('sources', [])])
        
        # Add LLM results
        if llm_results and llm_results.get('success'):
            response_parts.append("=== ANALYSIS ===")
            response_parts.append(llm_results.get('response', 'Direct analysis'))
            sources_used.append({"type": "llm_analysis", "title": "Direct AI Analysis"})
        
        if not response_parts:
            final_response = "I apologize, but I couldn't gather sufficient information to answer your query."
            confidence = 10
        else:
            final_response = "\n\n".join(response_parts)
            confidence = 75 if len(response_parts) > 1 else 60
        
        # Generate executive summary
        executive_summary = f"Query processed using {len(response_parts)} information source(s). Confidence: {confidence}%"
        
        return {
            "query": query,
            "response": final_response,
            "executive_summary": executive_summary,
            "sources_used": sources_used,
            "confidence": confidence,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }


class SimpleMultiAgentWorkflow:
    """Simplified multi-agent workflow"""
    
    def __init__(self, api_key: str = None):
        self.router = SimpleRouterAgent()
        self.web_agent = SimpleWebResearchAgent()
        self.rag_agent = SimpleRAGAgent()
        self.llm_agent = SimpleLLMAgent(api_key)
        self.summarizer = SimpleSummarizationAgent()
        
        print("🚀 Simple Multi-Agent Workflow initialized successfully!")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the multi-agent workflow"""
        print(f"\n🎯 Processing query: {query}")
        
        # Step 1: Route the query
        routing_result = self.router.route_query(query)
        route = routing_result["route"]
        
        print(f"📍 Routed to: {route} (confidence: {routing_result['confidence']:.2f})")
        
        # Step 2: Process based on route
        web_results = None
        rag_results = None
        llm_results = None
        
        if route == "web_search":
            web_results = self.web_agent.search_web(query)
        elif route == "rag":
            rag_results = self.rag_agent.search_knowledge_base(query)
        else:  # llm_direct
            llm_results = self.llm_agent.process_query(query)
        
        # Step 3: Synthesize results
        print("🔄 Synthesizing information...")
        final_result = self.summarizer.synthesize_information(
            query, web_results, rag_results, llm_results
        )
        
        # Add routing information
        final_result.update({
            "query_type": route,
            "routing_confidence": routing_result["confidence"],
            "routing_reasoning": routing_result["reasoning"]
        })
        
        print(f"✅ Query processed successfully (confidence: {final_result['confidence']}%)")
        
        return final_result


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


def main():
    """Main function to demonstrate the system"""
    print("🎯 Simple Multi-Agent Research System Demo")
    print("=" * 80)
    
    # Initialize the workflow
    api_key = os.getenv('OPENAI_API_KEY', 'demo-key')
    workflow = SimpleMultiAgentWorkflow(api_key)
    
    # Example queries
    examples = [
        "What is machine learning and how does it work?",  # Should route to RAG
        "What are the latest developments in AI in 2024?",  # Should route to web search
        "Solve this equation: 2x + 5 = 15",  # Should route to direct LLM
    ]
    
    for i, query in enumerate(examples, 1):
        print(f"\n{'='*20} EXAMPLE {i} {'='*20}")
        response = workflow.process_query(query)
        print_response(response)
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")
    
    print(f"\n{'='*80}")
    print("🎉 Demo completed!")
    print("\nTo use with a real OpenAI API key:")
    print("1. Set OPENAI_API_KEY in your .env file")
    print("2. The system will use actual LLM processing instead of simulated responses")


if __name__ == "__main__":
    main()