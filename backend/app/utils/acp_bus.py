import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


class ACPMessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_DELEGATION = "task_delegation"
    CORRECTION_REQUEST = "correction_request"
    SYSTEM_EVENT = "system_event"
    ERROR_REPORT = "error_report"


class ACPPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ACPEnvelope:
    """ACP message envelope for inter-agent communication"""
    
    # Identity
    envelope_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Routing
    sender_id: str = ""
    sender_name: str = ""
    receiver_id: str = ""
    receiver_name: str = ""
    
    # Message classification
    message_type: ACPMessageType = ACPMessageType.TASK_REQUEST
    priority: ACPPriority = ACPPriority.NORMAL
    
    # Payload
    payload: Dict[str, Any] = field(default_factory=dict)
    a2a_task: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    reply_to: Optional[str] = None
    
    # Audit
    hop_count: int = 0
    trace: List[str] = field(default_factory=list)
    
    def stamp(self, agent_name: str):
        """Record an agent touching this envelope"""
        self.hop_count += 1
        self.trace.append(f"[{datetime.now().strftime('%H:%M:%S')}] {agent_name} (hop {self.hop_count})")
    
    def to_dict(self) -> Dict:
        return {
            "envelope_id": self.envelope_id,
            "conversation_id": self.conversation_id,
            "sender_name": self.sender_name,
            "receiver_name": self.receiver_name,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "a2a_task": self.a2a_task,
            "created_at": self.created_at,
            "hop_count": self.hop_count,
            "trace": self.trace
        }


class ACPMessageBus:
    """Central message bus for all inter-agent communication"""
    
    def __init__(self):
        self.message_log: List[Dict] = []
        self.conversation_map: Dict[str, List[str]] = {}
    
    def send(self, envelope: ACPEnvelope, log: bool = True) -> ACPEnvelope:
        """Route an ACP envelope"""
        envelope.stamp("message_bus")
        
        if log:
            self.message_log.append(envelope.to_dict())
            
            cid = envelope.conversation_id
            if cid not in self.conversation_map:
                self.conversation_map[cid] = []
            self.conversation_map[cid].append(envelope.envelope_id)
            
            logging.info(f"ACP [{envelope.message_type.value}] {envelope.sender_name} => {envelope.receiver_name} | hop {envelope.hop_count}")
        
        return envelope
    
    def get_conversation_thread(self, conversation_id: str) -> List[Dict]:
        """Retrieve all messages in a conversation"""
        envelope_ids = self.conversation_map.get(conversation_id, [])
        return [m for m in self.message_log if m["envelope_id"] in envelope_ids]
    
    def print_audit_log(self):
        """Print full audit trail"""
        print("\n" + "="*60)
        print("ACP MESSAGE BUS — AUDIT LOG")
        print("="*60)
        print(f"Total messages: {len(self.message_log)}")
        print(f"Active conversations: {len(self.conversation_map)}")
        print("-"*60)
        for i, msg in enumerate(self.message_log):
            print(f"  [{i+1:02d}] {msg['message_type']:<22} {msg['sender_name']:<20} => {msg['receiver_name']}")
        print("="*60)


# Global message bus instance
acp_bus = ACPMessageBus()


def acp_send_task(
    sender_agent,
    receiver_agent,
    a2a_task,
    message_type: ACPMessageType = ACPMessageType.TASK_REQUEST,
    conversation_id: str = None,
    reply_to: str = None
) -> ACPEnvelope:
    """Helper to send A2A task wrapped in ACP envelope"""
    
    envelope = ACPEnvelope(
        conversation_id=conversation_id or str(uuid.uuid4()),
        sender_id=sender_agent.agent_card.agent_id,
        sender_name=sender_agent.agent_card.name,
        receiver_id=receiver_agent.agent_card.agent_id,
        receiver_name=receiver_agent.agent_card.name,
        message_type=message_type,
        reply_to=reply_to,
        payload={
            "task_id": a2a_task.task_id,
            "objective": a2a_task.objective,
            "status": a2a_task.status
        },
        a2a_task=a2a_task.to_dict()
    )
    
    return acp_bus.send(envelope)