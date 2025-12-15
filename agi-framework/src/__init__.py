"""
AGI Framework
Multi-LLM Agent 기반 사용자 상호작용 및 능동학습 AGI 프레임워크
"""

__version__ = "0.1.0"
__author__ = "CU NLP Lab"

from .core.orchestrator import Orchestrator
from .core.message_bus import MessageBus, Message

__all__ = [
    "Orchestrator",
    "MessageBus", 
    "Message",
    "__version__",
]

