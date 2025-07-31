"""Web Research Agent for fetching current information from the web"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import time
from urllib.parse import quote_plus, urljoin
import json


class WebResearchAgent:
    """
    Web Research Agent that performs web searches and extracts relevant information.
    
    Features:
    - Performs web searches using search engines
    - Extracts and cleans content from web pages
    - Summarizes findings using LLM
    - Handles multiple sources and conflicting information
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.search_engines = {
            "duckduckgo": "https://duckduckgo.com/html/?q={query}",
            "bing": "https://www.bing.com/search?q={query}"
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a web content analyzer. Extract the most relevant information from the provided web content that answers the user's query.

Instructions:
1. Focus on factual, recent, and credible information
2. Ignore advertisements, navigation menus, and irrelevant content
3. Summarize key points clearly and concisely
4. If multiple sources provide conflicting information, note the discrepancies
5. Include relevant dates, numbers, and specific details
6. Maintain objectivity and cite when information seems uncertain

Format your response as structured information with clear sections."""),
            ("human", """Query: {query}

Web Content:
{content}

Please extract and organize the most relevant information that answers the query.""")
        ])
        
        self.extraction_chain = self.extraction_prompt | self.llm | StrOutputParser()
    
    def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform web search and extract information from multiple sources.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to process
            
        Returns:
            Dict containing search results and extracted information
        """
        print(f"🔍 Searching web for: {query}")
        
        search_results = []
        extracted_info = []
        
        try:
            # Perform search using DuckDuckGo (more reliable for automated searches)
            results = self._search_duckduckgo(query, max_results)
            search_results.extend(results)
            
            # Extract content from top results
            for i, result in enumerate(results[:max_results]):
                print(f"📄 Processing result {i+1}: {result.get('title', 'Unknown')}")
                
                content = self._extract_content(result['url'])
                if content:
                    # Use LLM to extract relevant information
                    extracted = self._extract_relevant_info(query, content, result)
                    if extracted:
                        extracted_info.append({
                            'source': result,
                            'extracted_content': extracted,
                            'extraction_timestamp': time.time()
                        })
                
                # Add delay to be respectful to websites
                time.sleep(1)
        
        except Exception as e:
            print(f"Error in web search: {e}")
            return {
                'query': query,
                'results': [],
                'extracted_info': [],
                'error': str(e),
                'success': False
            }
        
        # Synthesize information from multiple sources
        synthesized_info = self._synthesize_information(query, extracted_info)
        
        return {
            'query': query,
            'results': search_results,
            'extracted_info': extracted_info,
            'synthesized_info': synthesized_info,
            'total_sources': len(extracted_info),
            'success': True,
            'timestamp': time.time()
        }
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        try:
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Extract search results
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:max_results]:
                try:
                    title_elem = div.find('a', class_='result__a')
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href')
                        
                        # Extract snippet
                        snippet_elem = div.find('a', class_='result__snippet')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        if url and title:
                            results.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet,
                                'source': 'DuckDuckGo'
                            })
                            
                except Exception as e:
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
            return []
    
    def _extract_content(self, url: str, max_length: int = 5000) -> Optional[str]:
        """Extract text content from a webpage"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "sidebar"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit length
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            return text
            
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return None
    
    def _extract_relevant_info(self, query: str, content: str, source: Dict) -> Optional[str]:
        """Use LLM to extract relevant information from web content"""
        try:
            # Limit content length for LLM processing
            if len(content) > 4000:
                content = content[:4000] + "..."
            
            extracted = self.extraction_chain.invoke({
                "query": query,
                "content": content
            })
            
            return extracted
            
        except Exception as e:
            print(f"Error extracting relevant info: {e}")
            return None
    
    def _synthesize_information(self, query: str, extracted_info: List[Dict]) -> str:
        """Synthesize information from multiple sources"""
        if not extracted_info:
            return "No relevant information found from web sources."
        
        try:
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert information synthesizer. Combine information from multiple web sources to provide a comprehensive answer to the user's query.

Instructions:
1. Identify common themes and consistent information across sources
2. Note any conflicting information and explain discrepancies
3. Prioritize more recent and credible information
4. Provide a well-structured, comprehensive response
5. Include key facts, figures, and dates when available
6. Maintain objectivity and note uncertainty when appropriate"""),
                ("human", """Query: {query}

Information from {num_sources} sources:

{combined_content}

Please synthesize this information into a comprehensive, well-structured response.""")
            ])
            
            # Combine all extracted content
            combined_content = ""
            for i, info in enumerate(extracted_info):
                source_title = info['source'].get('title', 'Unknown Source')
                content = info['extracted_content']
                combined_content += f"\n--- Source {i+1}: {source_title} ---\n{content}\n"
            
            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            
            synthesized = synthesis_chain.invoke({
                "query": query,
                "num_sources": len(extracted_info),
                "combined_content": combined_content
            })
            
            return synthesized
            
        except Exception as e:
            print(f"Error synthesizing information: {e}")
            # Fallback to simple concatenation
            return "\n\n".join([info['extracted_content'] for info in extracted_info])
    
    def get_search_summary(self, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of search results"""
        return {
            "query": search_result.get('query'),
            "total_sources": search_result.get('total_sources', 0),
            "success": search_result.get('success', False),
            "has_synthesized_info": bool(search_result.get('synthesized_info')),
            "source_titles": [
                info['source'].get('title', 'Unknown') 
                for info in search_result.get('extracted_info', [])
            ]
        }