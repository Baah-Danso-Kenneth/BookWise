from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass, field
import uuid
import logging
from datetime import datetime

from langchain_groq import ChatGroq
from app.config import config
from app.utils.acp_bus import ACPMessageType, acp_send_task



class A2ATaskStatus:
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class A2AAgentCard:
    """A2A Agent Card - declares agent identity and capabilities"""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    input_modes: List[str] = field(default_factory=list)
    output_modes: List[str] = field(default_factory=list)
    endpoint: str = ""
    skills: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": self.capabilities,
            "input_modes": self.input_modes,
            "output_modes": self.output_modes,
            "endpoint": self.endpoint,
            "skills": self.skills
        }


@dataclass
class A2ATask:
    """A2A Task object"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = ""
    to_agent: str = ""
    status: str = A2ATaskStatus.SUBMITTED
    objective: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def update_status(self, status: str):
        self.status = status
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "status": self.status,
            "objective": self.objective,
            "context": self.context,
            "result": self.result,
            "errors": self.errors,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


# ============================================================
# Base A2A Agent
# ============================================================

class A2AAgent(ABC):
    """Base class for all A2A-compliant agents"""
    
    def __init__(self):
        # LLM with retry + fallback
        primary_llm = ChatGroq(
            model=config.PRIMARY_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=500
        )
        
        fallback_llm = ChatGroq(
            model=config.FALLBACK_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=500
        )
        
        self.llm = primary_llm.with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=3
        ).with_fallbacks([fallback_llm])
        
        self.agent_card = self._register_card()
        self.task_history: List[A2ATask] = []
        
        logging.info(f"A2A Agent registered: {self.agent_card.name} [{self.agent_card.agent_id[:8]}...]")
    
    @abstractmethod
    def _register_card(self) -> A2AAgentCard:
        """Register agent capabilities"""
        pass
    
    @abstractmethod
    def process_task(self, task: A2ATask) -> A2ATask:
        """Process an A2A task"""
        pass
    
    def _invoke_llm(self, system: str, human: str) -> str:
        """Invoke LLM with system prompt"""
        response = self.llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": human}
        ])
        return response.content
    
    def _record_task(self, task: A2ATask):
        """Store task in history"""
        self.task_history.append(task)


    def send_task(self, target_agent, task, message_type=ACPMessageType.TASK_REQUEST, conversation_id=None):
        """Send a task to another agent via ACP"""
        return acp_send_task(self, target_agent, task, message_type, conversation_id)