"""Conversational Memory System for maintaining context across interactions"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pickle


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: str
    user_query: str
    system_response: str
    query_type: str
    sources_used: List[Dict[str, Any]]
    confidence: int
    session_id: str
    turn_id: str


class ConversationMemory:
    """
    Conversational Memory System that maintains context across interactions.
    
    Features:
    - Session management
    - Context retrieval
    - Memory persistence
    - Relevance-based memory filtering
    - Memory summarization
    """
    
    def __init__(self, memory_dir: str = "memory", max_turns_per_session: int = 50):
        self.memory_dir = memory_dir
        self.max_turns_per_session = max_turns_per_session
        self.current_session_id = None
        self.current_session_memory = []
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
        
        # Memory configuration
        self.memory_config = {
            'max_context_turns': 5,  # Maximum previous turns to include in context
            'relevance_threshold': 0.3,  # Minimum relevance score for including memory
            'max_memory_age_days': 30,  # Maximum age of memories to consider
            'auto_summarize_threshold': 20  # Summarize session after this many turns
        }
    
    def start_new_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save current session if exists
        if self.current_session_id and self.current_session_memory:
            self._save_session()
        
        self.current_session_id = session_id
        self.current_session_memory = []
        
        print(f"🧠 Started new conversation session: {session_id}")
        return session_id
    
    def add_conversation_turn(self, 
                            user_query: str,
                            system_response: str,
                            query_type: str,
                            sources_used: List[Dict[str, Any]] = None,
                            confidence: int = 50) -> ConversationTurn:
        """Add a new conversation turn to memory"""
        if not self.current_session_id:
            self.start_new_session()
        
        turn_id = f"turn_{len(self.current_session_memory) + 1}"
        
        conversation_turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            system_response=system_response,
            query_type=query_type,
            sources_used=sources_used or [],
            confidence=confidence,
            session_id=self.current_session_id,
            turn_id=turn_id
        )
        
        self.current_session_memory.append(conversation_turn)
        
        # Auto-save periodically
        if len(self.current_session_memory) % 10 == 0:
            self._save_session()
        
        # Auto-summarize if session gets too long
        if len(self.current_session_memory) >= self.memory_config['auto_summarize_threshold']:
            self._summarize_session()
        
        print(f"💭 Added conversation turn: {turn_id}")
        return conversation_turn
    
    def get_relevant_context(self, current_query: str, max_turns: Optional[int] = None) -> List[ConversationTurn]:
        """Get relevant conversation context for the current query"""
        max_turns = max_turns or self.memory_config['max_context_turns']
        
        if not self.current_session_memory:
            return []
        
        # Get recent turns
        recent_turns = self.current_session_memory[-max_turns:]
        
        # For now, return recent turns (could implement semantic similarity in future)
        relevant_turns = []
        for turn in recent_turns:
            if self._is_turn_relevant(turn, current_query):
                relevant_turns.append(turn)
        
        return relevant_turns[-max_turns:]  # Limit to max_turns
    
    def _is_turn_relevant(self, turn: ConversationTurn, current_query: str) -> bool:
        """Determine if a conversation turn is relevant to the current query"""
        # Simple keyword-based relevance (could be enhanced with embeddings)
        query_words = set(current_query.lower().split())
        turn_words = set((turn.user_query + " " + turn.system_response).lower().split())
        
        common_words = query_words.intersection(turn_words)
        relevance_score = len(common_words) / max(len(query_words), 1)
        
        return relevance_score >= self.memory_config['relevance_threshold']
    
    def get_conversation_context_string(self, current_query: str) -> str:
        """Get conversation context as a formatted string"""
        relevant_turns = self.get_relevant_context(current_query)
        
        if not relevant_turns:
            return "No relevant conversation history."
        
        context_parts = ["=== CONVERSATION CONTEXT ==="]
        
        for turn in relevant_turns:
            context_parts.append(f"\nPrevious Query: {turn.user_query}")
            context_parts.append(f"Previous Response: {turn.system_response[:300]}...")  # Truncate long responses
            context_parts.append(f"Query Type: {turn.query_type}")
            context_parts.append(f"Confidence: {turn.confidence}%")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def load_session(self, session_id: str) -> bool:
        """Load a specific session from persistent storage"""
        session_file = os.path.join(self.memory_dir, f"{session_id}.pkl")
        
        try:
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
            
            self.current_session_id = session_id
            self.current_session_memory = session_data['turns']
            
            print(f"📁 Loaded session: {session_id} ({len(self.current_session_memory)} turns)")
            return True
            
        except FileNotFoundError:
            print(f"❌ Session not found: {session_id}")
            return False
        except Exception as e:
            print(f"❌ Error loading session {session_id}: {e}")
            return False
    
    def _save_session(self):
        """Save current session to persistent storage"""
        if not self.current_session_id or not self.current_session_memory:
            return
        
        session_file = os.path.join(self.memory_dir, f"{self.current_session_id}.pkl")
        
        try:
            session_data = {
                'session_id': self.current_session_id,
                'created_at': self.current_session_memory[0].timestamp if self.current_session_memory else datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'turn_count': len(self.current_session_memory),
                'turns': self.current_session_memory
            }
            
            with open(session_file, 'wb') as f:
                pickle.dump(session_data, f)
            
            print(f"💾 Saved session: {self.current_session_id}")
            
        except Exception as e:
            print(f"❌ Error saving session: {e}")
    
    def _summarize_session(self):
        """Summarize the current session when it gets too long"""
        if len(self.current_session_memory) < self.memory_config['auto_summarize_threshold']:
            return
        
        print("📝 Summarizing long conversation session...")
        
        # Keep recent turns and summarize older ones
        recent_turns = self.current_session_memory[-10:]
        older_turns = self.current_session_memory[:-10]
        
        # Create a summary turn
        summary_text = self._create_session_summary(older_turns)
        
        summary_turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_query="[SESSION SUMMARY]",
            system_response=summary_text,
            query_type="summary",
            sources_used=[],
            confidence=100,
            session_id=self.current_session_id,
            turn_id="summary"
        )
        
        # Replace old turns with summary + recent turns
        self.current_session_memory = [summary_turn] + recent_turns
        self._save_session()
    
    def _create_session_summary(self, turns: List[ConversationTurn]) -> str:
        """Create a summary of conversation turns"""
        if not turns:
            return "No conversation to summarize."
        
        topics = {}
        total_queries = len(turns)
        
        # Group by query type
        for turn in turns:
            query_type = turn.query_type
            if query_type not in topics:
                topics[query_type] = []
            topics[query_type].append(turn.user_query)
        
        summary_parts = [
            f"Conversation Summary ({total_queries} previous interactions):",
            f"Time period: {turns[0].timestamp} to {turns[-1].timestamp}",
            ""
        ]
        
        for query_type, queries in topics.items():
            summary_parts.append(f"{query_type.title()} queries ({len(queries)}):")
            for query in queries[:3]:  # Show first 3 queries of each type
                summary_parts.append(f"  - {query}")
            if len(queries) > 3:
                summary_parts.append(f"  ... and {len(queries) - 3} more")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session"""
        if not self.current_session_memory:
            return {"session_id": self.current_session_id, "turn_count": 0}
        
        query_types = {}
        confidence_scores = []
        
        for turn in self.current_session_memory:
            # Count query types
            query_type = turn.query_type
            query_types[query_type] = query_types.get(query_type, 0) + 1
            
            # Collect confidence scores
            confidence_scores.append(turn.confidence)
        
        return {
            "session_id": self.current_session_id,
            "turn_count": len(self.current_session_memory),
            "query_types": query_types,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "session_started": self.current_session_memory[0].timestamp,
            "last_interaction": self.current_session_memory[-1].timestamp
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions"""
        sessions = []
        
        try:
            for file in os.listdir(self.memory_dir):
                if file.endswith('.pkl'):
                    session_id = file[:-4]  # Remove .pkl extension
                    
                    try:
                        with open(os.path.join(self.memory_dir, file), 'rb') as f:
                            session_data = pickle.load(f)
                        
                        sessions.append({
                            'session_id': session_id,
                            'turn_count': session_data.get('turn_count', 0),
                            'created_at': session_data.get('created_at'),
                            'last_updated': session_data.get('last_updated')
                        })
                    except Exception as e:
                        print(f"Error reading session {session_id}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error listing sessions: {e}")
        
        return sorted(sessions, key=lambda x: x.get('last_updated', ''), reverse=True)
    
    def clear_session(self):
        """Clear the current session memory"""
        if self.current_session_memory:
            self._save_session()
        
        self.current_session_memory = []
        print(f"🧹 Cleared current session memory")
    
    def export_session(self, session_id: str, format: str = 'json') -> Optional[str]:
        """Export a session to a readable format"""
        if session_id == self.current_session_id:
            turns = self.current_session_memory
        else:
            # Load session
            session_file = os.path.join(self.memory_dir, f"{session_id}.pkl")
            try:
                with open(session_file, 'rb') as f:
                    session_data = pickle.load(f)
                turns = session_data['turns']
            except Exception as e:
                print(f"Error loading session for export: {e}")
                return None
        
        if format == 'json':
            export_data = {
                'session_id': session_id,
                'exported_at': datetime.now().isoformat(),
                'turn_count': len(turns),
                'turns': []
            }
            
            for turn in turns:
                export_data['turns'].append({
                    'timestamp': turn.timestamp,
                    'user_query': turn.user_query,
                    'system_response': turn.system_response,
                    'query_type': turn.query_type,
                    'confidence': turn.confidence,
                    'sources_count': len(turn.sources_used)
                })
            
            return json.dumps(export_data, indent=2)
        
        return None