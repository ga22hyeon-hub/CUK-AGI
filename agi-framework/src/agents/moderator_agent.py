"""
모더레이터 에이전트
의도 분류, 계획 수립, 과업 수행 조율
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .base_agent import BaseAgent, AgentConfig, TaskResult
from ..core.message_bus import Message, MessageBus, MessageType


class IntentType(Enum):
    """사용자 의도 유형"""
    KNOWLEDGE_QUERY = "knowledge_query"      # 지식 검색
    TASK_EXECUTION = "task_execution"        # 작업 실행
    LEARNING_REQUEST = "learning_request"    # 학습 요청
    CONTEXT_RETRIEVAL = "context_retrieval"  # 컨텍스트 조회
    CLARIFICATION = "clarification"          # 명확화 요청
    FEEDBACK = "feedback"                    # 피드백
    UNKNOWN = "unknown"


@dataclass
class SubTask:
    """분해된 하위 작업"""
    task_id: str
    description: str
    agent_type: str  # 담당 에이전트
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    priority: int = 1


@dataclass 
class ExecutionPlan:
    """실행 계획"""
    plan_id: str
    original_query: str
    intent: IntentType
    subtasks: List[SubTask] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "created"  # created, executing, completed, failed
    
    def get_next_executable_tasks(self) -> List[SubTask]:
        """다음 실행 가능한 작업들 반환"""
        executable = []
        completed_ids = {
            t.task_id for t in self.subtasks if t.status == "completed"
        }
        
        for task in self.subtasks:
            if task.status == "pending":
                # 의존성 확인
                if all(dep in completed_ids for dep in task.dependencies):
                    executable.append(task)
        
        return sorted(executable, key=lambda t: t.priority, reverse=True)


@dataclass
class ModeratorConfig(AgentConfig):
    """모더레이터 에이전트 설정"""
    max_planning_steps: int = 10
    parallel_execution: bool = True
    max_subtasks: int = 5
    intent_confidence_threshold: float = 0.7


class ModeratorAgent(BaseAgent):
    """
    모더레이터 에이전트
    
    주요 기능:
    1. 사용자 의도 분류
    2. 과업 분해 및 계획 수립
    3. 에이전트 간 작업 조율
    4. 실행 흐름 관리
    """
    
    # 에이전트 유형별 ID 매핑 (오케스트레이터에서 설정)
    agent_registry: Dict[str, str] = {}
    
    def __init__(
        self,
        message_bus: MessageBus,
        config: ModeratorConfig = None,
        agent_id: str = None
    ):
        config = config or ModeratorConfig(
            name="moderator",
            description="의도 분류 및 작업 조율 에이전트"
        )
        super().__init__(config, message_bus, agent_id)
        
        self.mod_config = config
        
        # 실행 중인 계획들
        self.active_plans: Dict[str, ExecutionPlan] = {}
        
        # 의도 분류 키워드 (간단한 규칙 기반, 실제로는 LLM 사용)
        self.intent_keywords = {
            IntentType.KNOWLEDGE_QUERY: [
                "무엇", "왜", "어떻게", "설명", "알려", "what", "why", "how", "explain"
            ],
            IntentType.TASK_EXECUTION: [
                "생성", "만들", "작성", "계산", "변환", "create", "make", "generate"
            ],
            IntentType.LEARNING_REQUEST: [
                "학습", "훈련", "개선", "업데이트", "learn", "train", "improve"
            ],
            IntentType.CONTEXT_RETRIEVAL: [
                "이전", "기억", "히스토리", "맥락", "previous", "remember", "context"
            ],
        }
        
        # 핸들러 등록
        self.register_handler(
            MessageType.TASK_ASSIGNMENT,
            self._handle_task_assignment
        )
    
    async def on_start(self):
        """시작 시 초기화"""
        self.logger.info("모더레이터 에이전트 시작")
    
    def register_agent(self, agent_type: str, agent_id: str):
        """에이전트 등록"""
        self.agent_registry[agent_type] = agent_id
        self.logger.info(f"에이전트 등록: {agent_type} -> {agent_id}")
    
    async def handle_message(self, message: Message) -> TaskResult:
        """기본 메시지 처리"""
        return TaskResult(
            success=False,
            error=f"처리되지 않은 메시지 유형: {message.type.value}"
        )
    
    async def _handle_task_assignment(self, message: Message) -> TaskResult:
        """
        작업 할당 처리
        
        Payload:
            query: 사용자 쿼리
            context: 추가 컨텍스트 (선택)
        """
        query = message.payload.get("query", "")
        context = message.payload.get("context", {})
        
        self.logger.log_action("TASK_RECEIVED", {"query": query[:50]})
        
        try:
            # 1. 의도 분류
            intent, confidence = self.classify_intent(query)
            self.logger.info(f"의도 분류: {intent.value} (신뢰도: {confidence:.2f})")
            
            # 2. 계획 수립
            plan = await self.create_execution_plan(query, intent, context)
            self.active_plans[plan.plan_id] = plan
            self.logger.info(f"계획 수립 완료: {len(plan.subtasks)}개 하위 작업")
            
            # 3. 계획 실행
            result = await self.execute_plan(plan)
            
            return TaskResult(
                success=True,
                result={
                    "intent": intent.value,
                    "confidence": confidence,
                    "plan_id": plan.plan_id,
                    "execution_result": result,
                },
                metadata={
                    "num_subtasks": len(plan.subtasks),
                    "plan_status": plan.status,
                }
            )
            
        except Exception as e:
            self.logger.error(f"작업 처리 오류: {e}")
            return TaskResult(success=False, error=str(e))
    
    def classify_intent(self, query: str) -> Tuple[IntentType, float]:
        """
        의도 분류
        
        <think>
        사용자 쿼리를 분석하여 의도를 파악합니다.
        키워드 기반 규칙과 함께, 실제 시스템에서는 
        LLM을 활용한 의미론적 분류를 수행해야 합니다.
        </think>
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            (의도 유형, 신뢰도)
        """
        query_lower = query.lower()
        
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[intent] = score
        
        if not scores or max(scores.values()) == 0:
            return IntentType.UNKNOWN, 0.5
        
        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]
        total_keywords = len(self.intent_keywords[best_intent])
        confidence = min(max_score / max(total_keywords / 2, 1), 1.0)
        
        return best_intent, confidence
    
    async def create_execution_plan(
        self,
        query: str,
        intent: IntentType,
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        실행 계획 수립
        
        <think>
        쿼리와 의도를 기반으로 실행 계획을 수립합니다.
        - 의도에 따라 필요한 에이전트들을 결정
        - 작업 간 의존성을 고려하여 순서 결정
        - 병렬 실행 가능한 작업 식별
        </think>
        
        Args:
            query: 원본 쿼리
            intent: 분류된 의도
            context: 컨텍스트 정보
            
        Returns:
            실행 계획
        """
        import uuid
        
        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4())[:8],
            original_query=query,
            intent=intent,
        )
        
        # 의도별 작업 분해
        if intent == IntentType.KNOWLEDGE_QUERY:
            # 1. 컨텍스트 조회
            plan.subtasks.append(SubTask(
                task_id="ctx_1",
                description="관련 컨텍스트 조회",
                agent_type="context",
                priority=1,
            ))
            
            # 2. 지식 검색
            plan.subtasks.append(SubTask(
                task_id="kg_1",
                description="지식 그래프 검색",
                agent_type="knowledge_curation",
                dependencies=["ctx_1"],
                priority=2,
            ))
            
            # 3. 응답 생성
            plan.subtasks.append(SubTask(
                task_id="reason_1",
                description="응답 생성",
                agent_type="reasoning",
                dependencies=["kg_1"],
                priority=3,
            ))
            
        elif intent == IntentType.TASK_EXECUTION:
            # 1. 컨텍스트 조회
            plan.subtasks.append(SubTask(
                task_id="ctx_1",
                description="작업 컨텍스트 조회",
                agent_type="context",
                priority=1,
            ))
            
            # 2. 추론 및 작업 실행
            plan.subtasks.append(SubTask(
                task_id="reason_1",
                description="작업 실행",
                agent_type="reasoning",
                dependencies=["ctx_1"],
                priority=2,
            ))
            
        elif intent == IntentType.LEARNING_REQUEST:
            # 1. 학습 요청
            plan.subtasks.append(SubTask(
                task_id="learn_1",
                description="학습 프로세스 시작",
                agent_type="learning",
                priority=1,
            ))
            
        elif intent == IntentType.CONTEXT_RETRIEVAL:
            # 1. 히스토리 조회
            plan.subtasks.append(SubTask(
                task_id="ctx_1",
                description="히스토리 조회",
                agent_type="context",
                priority=1,
            ))
            
        else:
            # 기본: 지식 검색 + 추론
            plan.subtasks.append(SubTask(
                task_id="kg_1",
                description="관련 지식 검색",
                agent_type="knowledge_curation",
                priority=1,
            ))
            
            plan.subtasks.append(SubTask(
                task_id="reason_1",
                description="응답 생성",
                agent_type="reasoning",
                dependencies=["kg_1"],
                priority=2,
            ))
        
        plan.status = "created"
        return plan
    
    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        계획 실행
        
        Args:
            plan: 실행 계획
            
        Returns:
            실행 결과
        """
        plan.status = "executing"
        results = {}
        
        while True:
            # 다음 실행 가능한 작업들
            next_tasks = plan.get_next_executable_tasks()
            
            if not next_tasks:
                # 모든 작업 완료 또는 더 이상 실행 불가
                break
            
            # 병렬 실행 여부에 따라 처리
            if self.mod_config.parallel_execution and len(next_tasks) > 1:
                # 병렬 실행
                task_results = await asyncio.gather(*[
                    self._execute_subtask(task, plan, results)
                    for task in next_tasks
                ], return_exceptions=True)
                
                for task, result in zip(next_tasks, task_results):
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                    else:
                        task.status = "completed"
                        task.result = result
                        results[task.task_id] = result
            else:
                # 순차 실행
                for task in next_tasks:
                    try:
                        result = await self._execute_subtask(task, plan, results)
                        task.status = "completed"
                        task.result = result
                        results[task.task_id] = result
                    except Exception as e:
                        task.status = "failed"
                        task.result = str(e)
                        self.logger.error(f"작업 실패: {task.task_id} - {e}")
        
        # 최종 상태 확인
        failed_tasks = [t for t in plan.subtasks if t.status == "failed"]
        if failed_tasks:
            plan.status = "failed"
        else:
            plan.status = "completed"
        
        return {
            "task_results": results,
            "final_status": plan.status,
            "failed_tasks": [t.task_id for t in failed_tasks],
        }
    
    async def _execute_subtask(
        self,
        task: SubTask,
        plan: ExecutionPlan,
        previous_results: Dict[str, Any]
    ) -> Any:
        """하위 작업 실행"""
        task.status = "running"
        self.logger.info(f"작업 실행: {task.task_id} ({task.agent_type})")
        
        # 에이전트 ID 조회
        agent_id = self.agent_registry.get(task.agent_type)
        
        if not agent_id:
            self.logger.warning(f"에이전트 미등록: {task.agent_type}")
            # 시뮬레이션 결과 반환
            return {"simulated": True, "task": task.description}
        
        # 메시지 유형 결정
        message_type_map = {
            "knowledge_curation": MessageType.KNOWLEDGE_REQUEST,
            "context": MessageType.CONTEXT_REQUEST,
            "reasoning": MessageType.REASONING_REQUEST,
            "learning": MessageType.LEARNING_REQUEST,
        }
        
        message_type = message_type_map.get(
            task.agent_type, 
            MessageType.TASK_ASSIGNMENT
        )
        
        # 페이로드 구성
        payload = {
            "query": plan.original_query,
            "task_description": task.description,
            "previous_results": {
                dep: previous_results.get(dep)
                for dep in task.dependencies
            },
        }
        
        # 메시지 전송 및 응답 대기
        response = await self.send_message(
            receiver=agent_id,
            message_type=message_type,
            payload=payload,
            priority=task.priority,
        )
        
        if response:
            return response.payload.get("result")
        
        return None
    
    # 외부 API
    
    async def process_query(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        쿼리 처리 (외부 API)
        
        Args:
            query: 사용자 쿼리
            context: 컨텍스트 정보
            
        Returns:
            처리 결과
        """
        intent, confidence = self.classify_intent(query)
        plan = await self.create_execution_plan(query, intent, context or {})
        result = await self.execute_plan(plan)
        
        return {
            "intent": intent.value,
            "confidence": confidence,
            "plan_id": plan.plan_id,
            "result": result,
        }
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """계획 상태 조회"""
        plan = self.active_plans.get(plan_id)
        if plan:
            return {
                "plan_id": plan.plan_id,
                "status": plan.status,
                "intent": plan.intent.value,
                "subtasks": [
                    {
                        "task_id": t.task_id,
                        "status": t.status,
                        "agent": t.agent_type,
                    }
                    for t in plan.subtasks
                ],
            }
        return None

