"""Multi-Agent Research and Summarization Workflow using LangGraph"""

import os
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Import our custom agents
from agents.router_agent import RouterAgent, QueryType
from agents.web_research_agent import WebResearchAgent
from agents.rag_agent import RAGAgent
from agents.summarization_agent import SummarizationAgent
from memory.conversation_memory import ConversationMemory


class AgentState(TypedDict):
    """State object that flows through the LangGraph workflow"""
    # Input
    query: str
    session_id: Optional[str]
    
    # Routing information
    query_type: Optional[str]
    routing_confidence: Optional[float]
    routing_reasoning: Optional[str]
    
    # Conversation context
    conversation_context: Optional[str]
    
    # Agent results
    web_results: Optional[Dict[str, Any]]
    rag_results: Optional[Dict[str, Any]]
    llm_results: Optional[Dict[str, Any]]
    
    # Final output
    final_response: Optional[str]
    executive_summary: Optional[str]
    sources_used: Optional[List[Dict[str, Any]]]
    confidence: Optional[int]
    
    # Metadata
    processing_steps: List[str]
    errors: List[str]
    timestamp: Optional[str]


class MultiAgentWorkflow:
    """
    Multi-Agent Research and Summarization Workflow using LangGraph.
    
    Orchestrates Router, Web Research, RAG, and Summarization agents
    with conditional routing and memory management.
    """
    
    def __init__(self, openai_api_key: str):
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        
        # Initialize agents
        self.router_agent = RouterAgent(self.llm)
        self.web_research_agent = WebResearchAgent(self.llm)
        self.rag_agent = RAGAgent(self.llm)
        self.summarization_agent = SummarizationAgent(self.llm)
        
        # Initialize memory system
        self.memory = ConversationMemory()
        
        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        
        # LLM for direct processing
        self.direct_llm_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable AI assistant. Provide comprehensive, accurate responses to user queries.
            
Use your training data and reasoning capabilities to answer questions that don't require:
- Current/recent information (use web search for that)
- Specific knowledge base lookup (use RAG for that)

Focus on:
- General knowledge and reasoning
- Mathematical calculations
- Analysis and explanation
- Creative tasks
- Problem-solving

Provide clear, well-structured responses with appropriate detail."""),
            ("human", "{query}")
        ])
        
        self.direct_llm_chain = self.direct_llm_prompt | self.llm | StrOutputParser()
        
        print("🚀 Multi-Agent Workflow initialized successfully")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with conditional routing"""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("get_context", self._get_conversation_context)
        workflow.add_node("web_research", self._web_research)
        workflow.add_node("rag_processing", self._rag_processing)
        workflow.add_node("llm_direct", self._llm_direct_processing)
        workflow.add_node("synthesize", self._synthesize_results)
        workflow.add_node("save_memory", self._save_to_memory)
        
        # Set entry point
        workflow.set_entry_point("get_context")
        
        # Add edges
        workflow.add_edge("get_context", "route_query")
        
        # Conditional routing based on query type
        workflow.add_conditional_edges(
            "route_query",
            self._routing_condition,
            {
                "web_search": "web_research",
                "rag": "rag_processing", 
                "llm_direct": "llm_direct"
            }
        )
        
        # All paths lead to synthesis
        workflow.add_edge("web_research", "synthesize")
        workflow.add_edge("rag_processing", "synthesize")
        workflow.add_edge("llm_direct", "synthesize")
        
        # Final steps
        workflow.add_edge("synthesize", "save_memory")
        workflow.add_edge("save_memory", END)
        
        return workflow.compile()
    
    def _get_conversation_context(self, state: AgentState) -> AgentState:
        """Get conversation context from memory"""
        print("🧠 Retrieving conversation context...")
        
        try:
            # Start session if needed
            if not state.get("session_id"):
                session_id = self.memory.start_new_session()
                state["session_id"] = session_id
            
            # Get relevant context
            context = self.memory.get_conversation_context_string(state["query"])
            state["conversation_context"] = context
            state["processing_steps"] = state.get("processing_steps", []) + ["context_retrieved"]
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            state["errors"] = state.get("errors", []) + [f"Context retrieval error: {str(e)}"]
            state["conversation_context"] = "No conversation context available."
        
        return state
    
    def _route_query(self, state: AgentState) -> AgentState:
        """Route the query to appropriate agent(s)"""
        print("🔀 Routing query...")
        
        try:
            # Get routing decision
            routing_result = self.router_agent.route_query(state["query"])
            
            state["query_type"] = routing_result["query_type"].value
            state["routing_confidence"] = routing_result["confidence"]
            state["routing_reasoning"] = routing_result["reasoning"]
            state["processing_steps"] = state.get("processing_steps", []) + ["query_routed"]
            
            print(f"📍 Routed to: {state['query_type']} (confidence: {state['routing_confidence']:.2f})")
            
        except Exception as e:
            print(f"❌ Error in routing: {e}")
            state["errors"] = state.get("errors", []) + [f"Routing error: {str(e)}"]
            state["query_type"] = "llm_direct"  # Fallback
            state["routing_confidence"] = 0.5
        
        return state
    
    def _routing_condition(self, state: AgentState) -> str:
        """Determine which path to take based on routing decision"""
        return state.get("query_type", "llm_direct")
    
    def _web_research(self, state: AgentState) -> AgentState:
        """Perform web research"""
        print("🌐 Performing web research...")
        
        try:
            web_results = self.web_research_agent.search_web(state["query"], max_results=3)
            state["web_results"] = web_results
            state["processing_steps"] = state.get("processing_steps", []) + ["web_research_completed"]
            
        except Exception as e:
            print(f"❌ Error in web research: {e}")
            state["errors"] = state.get("errors", []) + [f"Web research error: {str(e)}"]
            state["web_results"] = {"success": False, "error": str(e)}
        
        return state
    
    def _rag_processing(self, state: AgentState) -> AgentState:
        """Perform RAG processing"""
        print("📚 Performing RAG processing...")
        
        try:
            rag_results = self.rag_agent.generate_rag_response(state["query"], max_results=3)
            state["rag_results"] = rag_results
            state["processing_steps"] = state.get("processing_steps", []) + ["rag_completed"]
            
        except Exception as e:
            print(f"❌ Error in RAG processing: {e}")
            state["errors"] = state.get("errors", []) + [f"RAG error: {str(e)}"]
            state["rag_results"] = {"retrieval_successful": False, "error": str(e)}
        
        return state
    
    def _llm_direct_processing(self, state: AgentState) -> AgentState:
        """Direct LLM processing"""
        print("🤖 Performing direct LLM processing...")
        
        try:
            # Include conversation context in the prompt if available
            query_with_context = state["query"]
            if state.get("conversation_context") and state["conversation_context"] != "No relevant conversation history.":
                query_with_context = f"""Previous conversation context:
{state['conversation_context']}

Current query: {state['query']}

Please consider the conversation history when responding."""
            
            response = self.direct_llm_chain.invoke({"query": query_with_context})
            
            state["llm_results"] = {
                "query": state["query"],
                "response": response,
                "success": True
            }
            state["processing_steps"] = state.get("processing_steps", []) + ["llm_direct_completed"]
            
        except Exception as e:
            print(f"❌ Error in direct LLM processing: {e}")
            state["errors"] = state.get("errors", []) + [f"LLM processing error: {str(e)}"]
            state["llm_results"] = {"success": False, "error": str(e)}
        
        return state
    
    def _synthesize_results(self, state: AgentState) -> AgentState:
        """Synthesize results from all sources"""
        print("🔄 Synthesizing results...")
        
        try:
            synthesis_result = self.summarization_agent.synthesize_information(
                query=state["query"],
                web_results=state.get("web_results"),
                rag_results=state.get("rag_results"),
                llm_results=state.get("llm_results")
            )
            
            state["final_response"] = synthesis_result["response"]
            state["executive_summary"] = synthesis_result["executive_summary"]
            state["sources_used"] = synthesis_result["sources_used"]
            state["confidence"] = synthesis_result["confidence"]
            state["processing_steps"] = state.get("processing_steps", []) + ["synthesis_completed"]
            
        except Exception as e:
            print(f"❌ Error in synthesis: {e}")
            state["errors"] = state.get("errors", []) + [f"Synthesis error: {str(e)}"]
            
            # Fallback: use the best available result
            if state.get("llm_results", {}).get("response"):
                state["final_response"] = state["llm_results"]["response"]
            elif state.get("rag_results", {}).get("response"):
                state["final_response"] = state["rag_results"]["response"]
            elif state.get("web_results", {}).get("synthesized_info"):
                state["final_response"] = state["web_results"]["synthesized_info"]
            else:
                state["final_response"] = "I apologize, but I encountered errors processing your query."
            
            state["confidence"] = 30  # Low confidence due to errors
        
        return state
    
    def _save_to_memory(self, state: AgentState) -> AgentState:
        """Save the conversation turn to memory"""
        print("💾 Saving to memory...")
        
        try:
            self.memory.add_conversation_turn(
                user_query=state["query"],
                system_response=state["final_response"],
                query_type=state.get("query_type", "unknown"),
                sources_used=state.get("sources_used", []),
                confidence=state.get("confidence", 50)
            )
            
            state["processing_steps"] = state.get("processing_steps", []) + ["memory_saved"]
            
        except Exception as e:
            print(f"❌ Error saving to memory: {e}")
            state["errors"] = state.get("errors", []) + [f"Memory save error: {str(e)}"]
        
        state["timestamp"] = self.memory.current_session_memory[-1].timestamp if self.memory.current_session_memory else None
        
        return state
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent workflow.
        
        Args:
            query (str): User query
            session_id (str, optional): Session ID for conversation continuity
            
        Returns:
            Dict containing the final response and metadata
        """
        print(f"\n🎯 Processing query: {query}")
        
        # Initialize state
        initial_state = {
            "query": query,
            "session_id": session_id,
            "processing_steps": [],
            "errors": []
        }
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Format response
            response = {
                "query": query,
                "response": final_state.get("final_response", "No response generated"),
                "executive_summary": final_state.get("executive_summary"),
                "query_type": final_state.get("query_type"),
                "routing_confidence": final_state.get("routing_confidence"),
                "sources_used": final_state.get("sources_used", []),
                "confidence": final_state.get("confidence", 0),
                "session_id": final_state.get("session_id"),
                "processing_steps": final_state.get("processing_steps", []),
                "errors": final_state.get("errors", []),
                "timestamp": final_state.get("timestamp"),
                "success": len(final_state.get("errors", [])) == 0
            }
            
            print(f"✅ Query processed successfully (confidence: {response['confidence']}%)")
            return response
            
        except Exception as e:
            print(f"❌ Workflow execution failed: {e}")
            return {
                "query": query,
                "response": f"I apologize, but I encountered a system error: {str(e)}",
                "executive_summary": "System error occurred",
                "query_type": "error",
                "confidence": 0,
                "sources_used": [],
                "errors": [str(e)],
                "success": False
            }
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about the workflow"""
        memory_stats = self.memory.get_session_stats()
        
        return {
            "current_session": memory_stats,
            "available_sessions": len(self.memory.list_sessions()),
            "agents_initialized": {
                "router": bool(self.router_agent),
                "web_research": bool(self.web_research_agent),
                "rag": bool(self.rag_agent),
                "summarization": bool(self.summarization_agent)
            },
            "memory_system_active": bool(self.memory.current_session_id)
        }
    
    def reset_session(self):
        """Reset the current conversation session"""
        self.memory.clear_session()
        print("🔄 Session reset successfully")
    
    def load_session(self, session_id: str) -> bool:
        """Load a specific conversation session"""
        return self.memory.load_session(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available conversation sessions"""
        return self.memory.list_sessions()