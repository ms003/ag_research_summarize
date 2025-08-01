"""Summarization Agent for synthesizing information from multiple sources"""

from typing import Dict, Any, List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import json
from datetime import datetime


class SummarizationAgent:
    """
    Summarization Agent that synthesizes information from multiple sources
    into coherent, well-structured responses.
    
    Features:
    - Multi-source information synthesis
    - Structured response formatting
    - Confidence assessment
    - Source attribution
    - Executive summary generation
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # Standard summarization prompt
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert information synthesizer and summarizer. Your task is to combine information from multiple sources into a comprehensive, well-structured response.

Guidelines:
1. SYNTHESIZE: Combine information from all provided sources into a coherent narrative
2. STRUCTURE: Organize information logically with clear sections and headings
3. ACCURACY: Only include information that is supported by the sources
4. ATTRIBUTION: Indicate which sources support each key point
5. CLARITY: Use clear, accessible language while maintaining technical accuracy
6. COMPLETENESS: Address all aspects of the user's query that are covered by the sources
7. CONFLICTS: If sources contradict each other, note the discrepancies and explain them

Format your response with:
- Executive Summary (2-3 sentences)
- Main Content (organized by topic/theme)
- Key Points (bullet points of main takeaways)
- Sources Used (brief list)
- Confidence Assessment (how well the sources address the query)"""),
            ("human", """User Query: {query}

Source Information:
{sources_info}

Please synthesize this information into a comprehensive response addressing the user's query.""")
        ])
        
        # Executive summary prompt
        self.executive_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Create a concise executive summary (2-3 sentences) that captures the essential answer to the user's query based on the provided information."""),
            ("human", """Query: {query}
Information: {information}

Executive Summary:""")
        ])
        
        # Confidence assessment prompt
        self.confidence_prompt = ChatPromptTemplate.from_messages([
            ("system", """Assess the confidence level (0-100%) for answering the given query based on the available sources. Consider:
- Completeness of information
- Quality and credibility of sources
- Consistency between sources
- Recency of information (if time-sensitive)

Respond with just a number between 0-100."""),
            ("human", """Query: {query}
Sources: {sources_summary}

Confidence Level:""")
        ])
        
        # Initialize chains
        self.summarization_chain = self.summarization_prompt | self.llm | StrOutputParser()
        self.executive_summary_chain = self.executive_summary_prompt | self.llm | StrOutputParser()
        self.confidence_chain = self.confidence_prompt | self.llm | StrOutputParser()
    
    def synthesize_information(self, 
                             query: str, 
                             web_results: Optional[Dict[str, Any]] = None,
                             rag_results: Optional[Dict[str, Any]] = None,
                             llm_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synthesize information from multiple agent sources.
        
        Args:
            query (str): Original user query
            web_results (Dict): Results from Web Research Agent
            rag_results (Dict): Results from RAG Agent
            llm_results (Dict): Results from direct LLM processing
            
        Returns:
            Dict containing synthesized response and metadata
        """
        print(f"🔄 Synthesizing information for: {query}")
        
        # Collect and format source information
        sources_info = self._format_sources(web_results, rag_results, llm_results)
        
        if not sources_info.strip():
            return {
                'query': query,
                'response': "No information was available to synthesize a response.",
                'executive_summary': "Insufficient information available.",
                'confidence': 0,
                'sources_used': [],
                'synthesis_successful': False,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Generate comprehensive synthesized response
            synthesized_response = self.summarization_chain.invoke({
                'query': query,
                'sources_info': sources_info
            })
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(query, synthesized_response)
            
            # Assess confidence
            confidence = self._assess_confidence(query, sources_info)
            
            # Extract sources used
            sources_used = self._extract_sources_used(web_results, rag_results, llm_results)
            
            return {
                'query': query,
                'response': synthesized_response,
                'executive_summary': executive_summary,
                'confidence': confidence,
                'sources_used': sources_used,
                'synthesis_successful': True,
                'timestamp': datetime.now().isoformat(),
                'word_count': len(synthesized_response.split()),
                'sources_count': len(sources_used)
            }
            
        except Exception as e:
            print(f"❌ Error in information synthesis: {e}")
            return {
                'query': query,
                'response': f"Error synthesizing information: {str(e)}",
                'executive_summary': "Synthesis failed due to technical error.",
                'confidence': 0,
                'sources_used': [],
                'synthesis_successful': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_sources(self, web_results: Optional[Dict], rag_results: Optional[Dict], llm_results: Optional[Dict]) -> str:
        """Format information from different sources for synthesis"""
        sources_parts = []
        
        # Format web research results
        if web_results and web_results.get('success') and web_results.get('synthesized_info'):
            sources_parts.append("=== WEB RESEARCH RESULTS ===")
            sources_parts.append(f"Query: {web_results.get('query', 'Unknown')}")
            sources_parts.append(f"Sources Found: {web_results.get('total_sources', 0)}")
            sources_parts.append("Information:")
            sources_parts.append(web_results['synthesized_info'])
            sources_parts.append("")
            
            # Add individual source details
            for i, info in enumerate(web_results.get('extracted_info', [])[:3]):  # Limit to top 3
                sources_parts.append(f"Web Source {i+1}: {info['source'].get('title', 'Unknown')}")
                sources_parts.append(f"URL: {info['source'].get('url', 'Unknown')}")
                sources_parts.append(f"Content: {info['extracted_content'][:500]}...")  # Limit content
                sources_parts.append("")
        
        # Format RAG results
        if rag_results and rag_results.get('retrieval_successful') and rag_results.get('response'):
            sources_parts.append("=== KNOWLEDGE BASE RESULTS ===")
            sources_parts.append(f"Query: {rag_results.get('query', 'Unknown')}")
            sources_parts.append(f"Sources Used: {rag_results.get('total_sources_used', 0)}")
            sources_parts.append("Response:")
            sources_parts.append(rag_results['response'])
            sources_parts.append("")
            
            # Add source details
            for source in rag_results.get('sources', []):
                sources_parts.append(f"KB Source: {source.get('title', 'Unknown')}")
                sources_parts.append(f"Category: {source.get('category', 'Unknown')}")
                sources_parts.append(f"Relevance: {source.get('similarity', 0):.3f}")
                sources_parts.append("")
        
        # Format direct LLM results
        if llm_results and llm_results.get('response'):
            sources_parts.append("=== DIRECT LLM ANALYSIS ===")
            sources_parts.append(f"Query: {llm_results.get('query', 'Unknown')}")
            sources_parts.append("Response:")
            sources_parts.append(llm_results['response'])
            sources_parts.append("")
        
        return "\n".join(sources_parts)
    
    def _generate_executive_summary(self, query: str, full_response: str) -> str:
        """Generate an executive summary of the response"""
        try:
            summary = self.executive_summary_chain.invoke({
                'query': query,
                'information': full_response[:2000]  # Limit input length
            })
            return summary.strip()
        except Exception as e:
            print(f"Error generating executive summary: {e}")
            return "Unable to generate executive summary."
    
    def _assess_confidence(self, query: str, sources_info: str) -> int:
        """Assess confidence level of the synthesized response"""
        try:
            # Create a summary of sources for confidence assessment
            sources_summary = f"Available sources: {len(sources_info.split('===')) - 1} different types"
            if "WEB RESEARCH" in sources_info:
                sources_summary += " (includes web research)"
            if "KNOWLEDGE BASE" in sources_info:
                sources_summary += " (includes knowledge base)"
            if "DIRECT LLM" in sources_info:
                sources_summary += " (includes LLM analysis)"
            
            confidence_str = self.confidence_chain.invoke({
                'query': query,
                'sources_summary': sources_summary
            })
            
            # Extract number from response
            confidence = int(''.join(filter(str.isdigit, confidence_str)))
            return max(0, min(100, confidence))  # Ensure 0-100 range
            
        except Exception as e:
            print(f"Error assessing confidence: {e}")
            return 50  # Default moderate confidence
    
    def _extract_sources_used(self, web_results: Optional[Dict], rag_results: Optional[Dict], llm_results: Optional[Dict]) -> List[Dict[str, Any]]:
        """Extract list of sources used in synthesis"""
        sources = []
        
        # Web sources
        if web_results and web_results.get('success'):
            for info in web_results.get('extracted_info', []):
                sources.append({
                    'type': 'web',
                    'title': info['source'].get('title', 'Unknown'),
                    'url': info['source'].get('url'),
                    'timestamp': info.get('extraction_timestamp')
                })
        
        # RAG sources
        if rag_results and rag_results.get('retrieval_successful'):
            for source in rag_results.get('sources', []):
                sources.append({
                    'type': 'knowledge_base',
                    'title': source.get('title', 'Unknown'),
                    'category': source.get('category'),
                    'relevance': source.get('similarity')
                })
        
        # LLM source
        if llm_results and llm_results.get('response'):
            sources.append({
                'type': 'llm_analysis',
                'title': 'Direct LLM Processing',
                'description': 'AI reasoning and analysis'
            })
        
        return sources
    
    def generate_comparative_summary(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comparative summary when multiple sources provide different perspectives"""
        comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", """Compare and contrast information from multiple sources. Identify:
1. Areas of agreement between sources
2. Conflicting information and possible reasons
3. Unique insights from each source
4. Overall reliability assessment

Structure your response clearly with these sections."""),
            ("human", """Query: {query}

Multiple Sources:
{sources}

Please provide a comparative analysis.""")
        ])
        
        try:
            sources_text = ""
            for i, result in enumerate(results):
                sources_text += f"\n--- Source {i+1} ---\n"
                sources_text += f"Type: {result.get('type', 'Unknown')}\n"
                sources_text += f"Content: {result.get('content', result.get('response', 'No content'))}\n"
            
            comparison_chain = comparison_prompt | self.llm | StrOutputParser()
            comparison = comparison_chain.invoke({
                'query': query,
                'sources': sources_text
            })
            
            return {
                'query': query,
                'comparative_analysis': comparison,
                'sources_compared': len(results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'query': query,
                'comparative_analysis': f"Error generating comparison: {str(e)}",
                'sources_compared': len(results),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_summary_stats(self, synthesis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the synthesis process"""
        return {
            'synthesis_successful': synthesis_result.get('synthesis_successful', False),
            'confidence_level': synthesis_result.get('confidence', 0),
            'sources_count': synthesis_result.get('sources_count', 0),
            'word_count': synthesis_result.get('word_count', 0),
            'source_types': list(set([
                source.get('type', 'unknown') 
                for source in synthesis_result.get('sources_used', [])
            ])),
            'has_executive_summary': bool(synthesis_result.get('executive_summary')),
            'timestamp': synthesis_result.get('timestamp')
        }