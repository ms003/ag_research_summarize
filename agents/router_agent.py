"""Router Agent for determining query processing strategy"""

import re
from typing import Dict, Any, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from enum import Enum


class QueryType(Enum):
    """Enumeration of query types for routing decisions"""
    WEB_SEARCH = "web_search"
    RAG = "rag"
    LLM_DIRECT = "llm_direct"


class RouterAgent:
    """
    Router Agent that determines the appropriate processing strategy for user queries.
    
    Routes queries to:
    - Web Research Agent: for current events, latest information
    - RAG Agent: for knowledge base queries
    - Direct LLM: for general reasoning tasks
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
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
        
        from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a query router that determines how to process user queries.
            
Analyze the query and determine the best processing approach:

1. WEB_SEARCH: For queries requiring current, real-time, or recent information
   - Keywords like: latest, current, recent, today, news, 2024, trending, now
   - Examples: "latest AI developments", "current stock prices", "recent news about Tesla"

2. RAG: For queries about established knowledge, definitions, or historical information
   - Keywords like: what is, explain, define, concept, theory, history
   - Examples: "what is machine learning", "explain quantum computing", "history of computers"

3. LLM_DIRECT: For general reasoning, math, creative tasks, or analysis
   - Examples: "solve this equation", "write a poem", "analyze this data"

Respond with only one word: WEB_SEARCH, RAG, or LLM_DIRECT"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template("Query: {query}")
        
        self.router_prompt = ChatPromptTemplate.from_messages([
            system_message,
            human_message
        ])
        
        self.router_chain = self.router_prompt | self.llm | StrOutputParser()
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Determine the routing strategy for a given query.
        
        Args:
            query (str): The user query to analyze
            
        Returns:
            Dict containing routing decision and reasoning
        """
        query_lower = query.lower()
        
        # First, use keyword-based routing for quick decisions
        keyword_decision = self._keyword_based_routing(query_lower)
        
        # Then use LLM for more nuanced analysis
        try:
            llm_decision = self.router_chain.invoke({"query": query})
            llm_decision = llm_decision.strip().upper()
            
            # Validate LLM response
            if llm_decision in ["WEB_SEARCH", "RAG", "LLM_DIRECT"]:
                final_decision = QueryType(llm_decision.lower())
            else:
                # Fallback to keyword-based decision
                final_decision = keyword_decision
                
        except Exception as e:
            print(f"Error in LLM routing: {e}")
            final_decision = keyword_decision
        
        return {
            "query_type": final_decision,
            "query": query,
            "reasoning": self._get_routing_reasoning(query, final_decision),
            "confidence": self._calculate_confidence(query, final_decision)
        }
    
    def _keyword_based_routing(self, query_lower: str) -> QueryType:
        """Perform keyword-based routing as a baseline"""
        web_score = sum(1 for keyword in self.web_search_keywords if keyword in query_lower)
        rag_score = sum(1 for keyword in self.rag_keywords if keyword in query_lower)
        
        if web_score > rag_score and web_score > 0:
            return QueryType.WEB_SEARCH
        elif rag_score > 0:
            return QueryType.RAG
        else:
            return QueryType.LLM_DIRECT
    
    def _get_routing_reasoning(self, query: str, decision: QueryType) -> str:
        """Generate reasoning for the routing decision"""
        reasoning_map = {
            QueryType.WEB_SEARCH: f"Query requires current/recent information - routing to web search",
            QueryType.RAG: f"Query asks for established knowledge/definitions - routing to knowledge base",
            QueryType.LLM_DIRECT: f"Query requires reasoning/analysis - routing to direct LLM processing"
        }
        return reasoning_map.get(decision, "Default routing decision")
    
    def _calculate_confidence(self, query: str, decision: QueryType) -> float:
        """Calculate confidence score for routing decision"""
        query_lower = query.lower()
        
        if decision == QueryType.WEB_SEARCH:
            matches = sum(1 for keyword in self.web_search_keywords if keyword in query_lower)
            return min(0.6 + (matches * 0.1), 1.0)
        elif decision == QueryType.RAG:
            matches = sum(1 for keyword in self.rag_keywords if keyword in query_lower)
            return min(0.6 + (matches * 0.1), 1.0)
        else:
            return 0.7  # Default confidence for LLM_DIRECT
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing patterns"""
        return {
            "web_search_keywords": len(self.web_search_keywords),
            "rag_keywords": len(self.rag_keywords),
            "supported_routes": [route.value for route in QueryType]
        }