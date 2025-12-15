"""
오케스트레이터
에이전트 라이프사이클 관리 및 연계 조율
"""

import asyncio
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
from datetime import datetime
import yaml
from pathlib import Path

from .message_bus import MessageBus, Message, MessageType
from ..agents.base_agent import BaseAgent, AgentConfig
from ..utils.logging_utils import setup_logger


@dataclass
class OrchestratorConfig:
    """오케스트레이터 설정"""
    max_queue_size: int = 1000
    agent_timeout: float = 60.0
    retry_attempts: int = 3
    parallel_agents: bool = True
    log_level: str = "INFO"


class Orchestrator:
    """
    오케스트레이터
    
    Multi-LLM Agent 시스템의 중앙 관리자
    - 에이전트 라이프사이클 관리
    - 메시지 라우팅
    - 시스템 모니터링
    - 장애 복구
    """
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        
        # 메시지 버스
        self.message_bus = MessageBus(max_queue_size=self.config.max_queue_size)
        
        # 에이전트 레지스트리
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, str] = {}  # type -> agent_id
        
        # 로거
        self.logger = setup_logger(
            "agi_framework.orchestrator",
            level=self.config.log_level,
        )
        
        # 상태
        self._running = False
        self._started_at: Optional[datetime] = None
    
    def register_agent(
        self,
        agent: BaseAgent,
        agent_type: str = None
    ):
        """
        에이전트 등록
        
        Args:
            agent: 등록할 에이전트
            agent_type: 에이전트 유형 (모더레이터에서 사용)
        """
        self.agents[agent.agent_id] = agent
        
        agent_type = agent_type or agent.config.name
        self.agent_types[agent_type] = agent.agent_id
        
        self.logger.info(f"에이전트 등록: {agent.agent_id} (유형: {agent_type})")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """에이전트 조회"""
        return self.agents.get(agent_id)
    
    def get_agent_by_type(self, agent_type: str) -> Optional[BaseAgent]:
        """유형으로 에이전트 조회"""
        agent_id = self.agent_types.get(agent_type)
        if agent_id:
            return self.agents.get(agent_id)
        return None
    
    async def start(self):
        """시스템 시작"""
        self.logger.info("=" * 50)
        self.logger.info("AGI Framework 시작")
        self.logger.info("=" * 50)
        
        self._running = True
        self._started_at = datetime.now()
        
        # 모든 에이전트 시작
        start_tasks = []
        for agent in self.agents.values():
            start_tasks.append(agent.start())
        
        if self.config.parallel_agents:
            await asyncio.gather(*start_tasks)
        else:
            for task in start_tasks:
                await task
        
        # 모더레이터에 에이전트 등록
        moderator = self.get_agent_by_type("moderator")
        if moderator and hasattr(moderator, "register_agent"):
            for agent_type, agent_id in self.agent_types.items():
                if agent_type != "moderator":
                    moderator.register_agent(agent_type, agent_id)
        
        self.logger.info(f"시스템 시작 완료 - {len(self.agents)}개 에이전트 활성화")
        
        # 상태 모니터링 루프
        asyncio.create_task(self._monitoring_loop())
    
    async def stop(self):
        """시스템 중지"""
        self.logger.info("시스템 중지 중...")
        self._running = False
        
        # 모든 에이전트 중지
        stop_tasks = []
        for agent in self.agents.values():
            stop_tasks.append(agent.stop())
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.logger.info("시스템 중지 완료")
    
    async def _monitoring_loop(self):
        """상태 모니터링 루프"""
        while self._running:
            try:
                await asyncio.sleep(30)  # 30초마다 체크
                
                # 에이전트 상태 확인
                status = self.get_system_status()
                
                # 문제 감지
                for agent_status in status["agents"]:
                    if agent_status["state"] == "error":
                        self.logger.warning(
                            f"에이전트 오류 상태: {agent_status['agent_id']}"
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"모니터링 오류: {e}")
    
    async def process_request(
        self,
        query: str,
        context: Dict[str, Any] = None,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        사용자 요청 처리
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            session_id: 세션 ID
            
        Returns:
            처리 결과
        """
        self.logger.info(f"요청 수신: {query[:50]}...")
        
        # 컨텍스트 에이전트에 히스토리 추가
        context_agent = self.get_agent_by_type("context")
        if context_agent and hasattr(context_agent, "add_history"):
            context_agent.add_history(
                session_id=session_id,
                role="user",
                content=query,
            )
        
        # 모더레이터에 작업 할당
        moderator = self.get_agent_by_type("moderator")
        
        if moderator:
            # 메시지 생성
            message = Message(
                type=MessageType.TASK_ASSIGNMENT,
                sender="orchestrator",
                receiver=moderator.agent_id,
                payload={
                    "query": query,
                    "context": context or {},
                    "session_id": session_id,
                },
            )
            
            # 요청 전송 및 응답 대기
            response = await self.message_bus.request(
                message,
                timeout=self.config.agent_timeout,
            )
            
            if response:
                result = response.payload.get("result", {})
                
                # 응답을 히스토리에 추가
                if context_agent and hasattr(context_agent, "add_history"):
                    response_text = str(result.get("execution_result", {}).get("task_results", ""))
                    context_agent.add_history(
                        session_id=session_id,
                        role="assistant",
                        content=response_text[:1000],  # 길이 제한
                    )
                
                return {
                    "success": True,
                    "result": result,
                    "session_id": session_id,
                }
            else:
                return {
                    "success": False,
                    "error": "요청 타임아웃",
                    "session_id": session_id,
                }
        else:
            return {
                "success": False,
                "error": "모더레이터 에이전트 없음",
                "session_id": session_id,
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        uptime = None
        if self._started_at:
            uptime = (datetime.now() - self._started_at).total_seconds()
        
        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_seconds": uptime,
            "num_agents": len(self.agents),
            "agents": [
                agent.get_status() for agent in self.agents.values()
            ],
            "message_bus": {
                "queue_status": self.message_bus.get_queue_status(),
            },
        }
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "Orchestrator":
        """설정 파일에서 오케스트레이터 생성"""
        config_path = Path(config_path)
        
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        orchestrator_config = OrchestratorConfig(
            max_queue_size=config_data.get("orchestrator", {}).get("message_queue", {}).get("max_queue_size", 1000),
            agent_timeout=config_data.get("orchestrator", {}).get("agent_timeout", 60),
            retry_attempts=config_data.get("orchestrator", {}).get("retry_attempts", 3),
            parallel_agents=config_data.get("orchestrator", {}).get("parallel_agents", True),
            log_level=config_data.get("framework", {}).get("log_level", "INFO"),
        )
        
        return cls(config=orchestrator_config)


class AGIFramework:
    """
    AGI 프레임워크 팩토리
    
    전체 시스템을 쉽게 구성하기 위한 헬퍼 클래스
    """
    
    @staticmethod
    def create_default_system(config_path: str = None) -> Orchestrator:
        """
        기본 시스템 생성
        
        모든 에이전트가 포함된 기본 설정의 시스템 반환
        """
        from ..agents.moderator_agent import ModeratorAgent, ModeratorConfig
        from ..agents.knowledge_curation_agent import KnowledgeCurationAgent, KnowledgeCurationConfig
        from ..agents.context_agent import ContextAgent, ContextConfig
        from ..agents.reasoning_agent import ReasoningAgent, ReasoningConfig
        from ..agents.learning_agent import LearningAgent, LearningConfig
        
        # 오케스트레이터 생성
        if config_path:
            orchestrator = Orchestrator.from_config_file(config_path)
        else:
            orchestrator = Orchestrator()
        
        # 에이전트 생성 및 등록
        moderator = ModeratorAgent(
            message_bus=orchestrator.message_bus,
            config=ModeratorConfig(name="moderator", description="모더레이터")
        )
        orchestrator.register_agent(moderator, "moderator")
        
        knowledge_agent = KnowledgeCurationAgent(
            message_bus=orchestrator.message_bus,
            config=KnowledgeCurationConfig(name="knowledge_curation", description="지식 큐레이션")
        )
        orchestrator.register_agent(knowledge_agent, "knowledge_curation")
        
        context_agent = ContextAgent(
            message_bus=orchestrator.message_bus,
            config=ContextConfig(name="context", description="컨텍스트 관리")
        )
        orchestrator.register_agent(context_agent, "context")
        
        reasoning_agent = ReasoningAgent(
            message_bus=orchestrator.message_bus,
            config=ReasoningConfig(name="reasoning", description="추론", load_model=False)
        )
        orchestrator.register_agent(reasoning_agent, "reasoning")
        
        learning_agent = LearningAgent(
            message_bus=orchestrator.message_bus,
            config=LearningConfig(name="learning", description="학습", auto_learn=False)
        )
        orchestrator.register_agent(learning_agent, "learning")
        
        return orchestrator
    
    @staticmethod
    async def quick_start(config_path: str = None):
        """빠른 시작"""
        orchestrator = AGIFramework.create_default_system(config_path)
        await orchestrator.start()
        return orchestrator

