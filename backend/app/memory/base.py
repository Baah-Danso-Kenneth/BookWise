from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging


class MemoryBase(ABC):
    """Base class for all memory systems"""
    
    @abstractmethod
    def store(self, key: str, value: Any) -> bool:
        """Store data in memory"""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Any:
        """Retrieve data from memory"""
        pass
    
    @abstractmethod
    def update(self, key: str, value: Any) -> bool:
        """Update existing memory"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete from memory"""
        pass