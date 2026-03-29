import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.memory.base import MemoryBase


class EpisodicMemory(MemoryBase):
    """
    Episodic Memory - Stores user-specific history and preferences.
    
    Stores:
    - Books user has read
    - Ratings given
    - Preferences derived from history
    """
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.memory_file = Path(__file__).parent.parent.parent / "data" / f"memory_{user_id}.json"
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load memory from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file) as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load memory: {e}")
        
        # Default structure
        return {
            "user_id": self.user_id,
            "read_books": [],
            "ratings": {},
            "preferences": {
                "liked_topics": [],
                "disliked_topics": [],
                "favorite_authors": [],
                "last_query": None
            },
            "session_history": []
        }
    
    def _save(self):
        """Save memory to file"""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save memory: {e}")
    
    def add_read_book(self, title: str, rating: int = None):
        """Record that user read a book"""
        book_entry = {
            "title": title,
            "date": datetime.now().isoformat()
        }
        if rating:
            book_entry["rating"] = rating
            self.data["ratings"][title] = rating
        
        if title not in [b["title"] for b in self.data["read_books"]]:
            self.data["read_books"].append(book_entry)
        
        # Update derived preferences
        self._update_preferences(title, rating)
        
        self._save()
    
    def add_to_history(self, query: str, recommendations: List[Dict], feedback: str = None):
        """Record a session interaction"""
        self.data["session_history"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "recommendations": [r.get("title") for r in recommendations[:3]],
            "feedback": feedback
        })
        
        # Keep last 20 sessions
        if len(self.data["session_history"]) > 20:
            self.data["session_history"] = self.data["session_history"][-20:]
        
        self.data["preferences"]["last_query"] = query
        self._save()
    
    def _update_preferences(self, title: str, rating: int):
        """Update user preferences based on ratings"""
        if rating and rating >= 4:
            # Liked book - could extract topics in production
            if title not in self.data["preferences"]["liked_topics"]:
                self.data["preferences"]["liked_topics"].append(title)
        elif rating and rating <= 2:
            # Disliked book
            if title not in self.data["preferences"]["disliked_topics"]:
                self.data["preferences"]["disliked_topics"].append(title)
    
    def get_preferences(self) -> Dict:
        """Get user preferences"""
        return self.data["preferences"]
    
    def get_read_books(self) -> List[str]:
        """Get list of books user has read"""
        return [b["title"] for b in self.data["read_books"]]
    
    def has_read(self, title: str) -> bool:
        """Check if user has read a book"""
        return title in self.get_read_books()
    
    def get_rating(self, title: str) -> Optional[int]:
        """Get user's rating for a book"""
        return self.data["ratings"].get(title)
    
    def store(self, key: str, value: Any) -> bool:
        """Store arbitrary key-value"""
        self.data[key] = value
        self._save()
        return True
    
    def retrieve(self, key: str) -> Any:
        """Retrieve by key"""
        return self.data.get(key)
    
    def update(self, key: str, value: Any) -> bool:
        return self.store(key, value)
    
    def delete(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            self._save()
            return True
        return False