"""RAG Agent for retrieving information from vector database"""

import json
import os
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np


class RAGAgent:
    """
    Retrieval-Augmented Generation Agent that searches a vector database 
    for relevant information and generates responses using retrieved context.
    
    Features:
    - Vector database integration with ChromaDB
    - Semantic search using embeddings
    - Context-aware response generation
    - Relevance scoring and filtering
    """
    
    def __init__(self, llm: ChatOpenAI, knowledge_base_path: str = "data/sample_knowledge_base.json"):
        self.llm = llm
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Create or get collection
        self.collection_name = "knowledge_base"
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            print(f"📊 Loaded existing collection: {self.collection_name}")
        except ValueError:
            self.collection = self.chroma_client.create_collection(self.collection_name)
            print(f"📊 Created new collection: {self.collection_name}")
            self._initialize_knowledge_base()
        
        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant that provides accurate, detailed responses based on the given context.

Instructions:
1. Use the provided context to answer the user's question comprehensively
2. If the context doesn't contain enough information, clearly state what's missing
3. Cite relevant information from the context when appropriate
4. Provide structured, easy-to-understand responses
5. If multiple sources are provided, synthesize the information coherently
6. Don't make up information not present in the context"""),
            ("human", """Question: {question}

Context from knowledge base:
{context}

Please provide a comprehensive answer based on the context above.""")
        ])
        
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
    
    def _initialize_knowledge_base(self):
        """Load and index the knowledge base into vector database"""
        print("🔄 Initializing knowledge base...")
        
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for item in knowledge_data:
                # Combine title and content for better retrieval
                document_text = f"Title: {item['title']}\n\nContent: {item['content']}"
                documents.append(document_text)
                
                metadata = {
                    "title": item['title'],
                    "category": item['category'],
                    "tags": json.dumps(item['tags']),
                    "timestamp": item['timestamp']
                }
                metadatas.append(metadata)
                ids.append(item['id'])
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Indexed {len(documents)} documents into knowledge base")
            
        except Exception as e:
            print(f"❌ Error initializing knowledge base: {e}")
    
    def retrieve_relevant_info(self, query: str, max_results: int = 5, min_similarity: float = 0.3) -> Dict[str, Any]:
        """
        Retrieve relevant information from the knowledge base.
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of results to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            Dict containing retrieved documents and metadata
        """
        print(f"🔍 Searching knowledge base for: {query}")
        
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results
            )
            
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    similarity = 1 - distance
                    
                    if similarity >= min_similarity:
                        retrieved_docs.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity': similarity,
                            'rank': i + 1
                        })
            
            return {
                'query': query,
                'retrieved_documents': retrieved_docs,
                'total_found': len(retrieved_docs),
                'search_successful': True
            }
            
        except Exception as e:
            print(f"❌ Error retrieving information: {e}")
            return {
                'query': query,
                'retrieved_documents': [],
                'total_found': 0,
                'search_successful': False,
                'error': str(e)
            }
    
    def generate_rag_response(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Generate a response using retrieved context from the knowledge base.
        
        Args:
            query (str): User question
            max_results (int): Maximum number of documents to retrieve
            
        Returns:
            Dict containing the generated response and metadata
        """
        print(f"🤖 Generating RAG response for: {query}")
        
        # Retrieve relevant documents
        retrieval_result = self.retrieve_relevant_info(query, max_results)
        
        if not retrieval_result['search_successful'] or not retrieval_result['retrieved_documents']:
            return {
                'query': query,
                'response': "I couldn't find relevant information in the knowledge base to answer your question.",
                'sources': [],
                'retrieval_successful': False
            }
        
        # Prepare context from retrieved documents
        context_parts = []
        sources = []
        
        for doc in retrieval_result['retrieved_documents']:
            context_parts.append(f"Source: {doc['metadata']['title']}")
            context_parts.append(f"Category: {doc['metadata']['category']}")
            context_parts.append(f"Content: {doc['content']}")
            context_parts.append(f"Relevance Score: {doc['similarity']:.3f}")
            context_parts.append("---")
            
            sources.append({
                'title': doc['metadata']['title'],
                'category': doc['metadata']['category'],
                'similarity': doc['similarity'],
                'rank': doc['rank']
            })
        
        context = "\n".join(context_parts)
        
        try:
            # Generate response using LLM
            response = self.rag_chain.invoke({
                'question': query,
                'context': context
            })
            
            return {
                'query': query,
                'response': response,
                'sources': sources,
                'context_length': len(context),
                'retrieval_successful': True,
                'total_sources_used': len(sources)
            }
            
        except Exception as e:
            print(f"❌ Error generating RAG response: {e}")
            return {
                'query': query,
                'response': f"Error generating response: {str(e)}",
                'sources': sources,
                'retrieval_successful': True,
                'generation_error': str(e)
            }
    
    def search_by_category(self, category: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search documents by category"""
        try:
            results = self.collection.get(
                where={"category": category},
                limit=max_results
            )
            
            documents = []
            if results['documents']:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    documents.append({
                        'content': doc,
                        'metadata': metadata
                    })
            
            return documents
            
        except Exception as e:
            print(f"Error searching by category: {e}")
            return []
    
    def search_by_tags(self, tags: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
        """Search documents that contain any of the specified tags"""
        try:
            all_docs = self.collection.get()
            matching_docs = []
            
            for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                doc_tags = json.loads(metadata.get('tags', '[]'))
                if any(tag.lower() in [t.lower() for t in doc_tags] for tag in tags):
                    matching_docs.append({
                        'content': doc,
                        'metadata': metadata
                    })
                    
                    if len(matching_docs) >= max_results:
                        break
            
            return matching_docs
            
        except Exception as e:
            print(f"Error searching by tags: {e}")
            return []
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            collection_count = self.collection.count()
            
            # Get all documents to analyze categories and tags
            all_docs = self.collection.get()
            categories = set()
            all_tags = set()
            
            for metadata in all_docs['metadatas']:
                categories.add(metadata['category'])
                tags = json.loads(metadata.get('tags', '[]'))
                all_tags.update(tags)
            
            return {
                'total_documents': collection_count,
                'categories': list(categories),
                'total_categories': len(categories),
                'all_tags': list(all_tags),
                'total_unique_tags': len(all_tags),
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            print(f"Error getting knowledge base stats: {e}")
            return {}
    
    def add_document(self, doc_id: str, title: str, content: str, category: str, tags: List[str]) -> bool:
        """Add a new document to the knowledge base"""
        try:
            document_text = f"Title: {title}\n\nContent: {content}"
            metadata = {
                "title": title,
                "category": category,
                "tags": json.dumps(tags),
                "timestamp": "2024-01-01"  # You might want to use actual timestamp
            }
            
            self.collection.add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            print(f"✅ Added document: {title}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            return False