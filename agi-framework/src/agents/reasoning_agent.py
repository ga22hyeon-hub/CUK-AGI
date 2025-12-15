"""
추론 에이전트
LoRA Adapter 기반 태스크별 추론 수행
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base_agent import BaseAgent, AgentConfig, TaskResult
from ..core.message_bus import Message, MessageBus, MessageType
from ..models.lora_adapter import LoRAAdapter, AdapterConfig, AdapterPool


@dataclass
class ReasoningConfig(AgentConfig):
    """추론 에이전트 설정"""
    base_model: str = "meta-llama/Llama-2-7b-hf"
    default_adapter: str = None
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    load_model: bool = False  # 실제 모델 로드 여부


@dataclass
class TaskModule:
    """태스크 모듈"""
    name: str
    description: str
    adapter_name: str
    prompt_template: str
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    avg_score: float = 0.0


class ReasoningAgent(BaseAgent):
    """
    추론 에이전트
    
    주요 기능:
    1. LoRA Adapter 기반 태스크별 추론
    2. 교육자료 생성, 소견 생성 등 다양한 태스크 모듈
    3. 동적 어댑터 전환
    4. 불확실도 추정
    """
    
    def __init__(
        self,
        message_bus: MessageBus,
        config: ReasoningConfig = None,
        agent_id: str = None
    ):
        config = config or ReasoningConfig(
            name="reasoning",
            description="LoRA 기반 추론 에이전트"
        )
        super().__init__(config, message_bus, agent_id)
        
        self.reason_config = config
        
        # 어댑터 풀
        self.adapter_pool = AdapterPool(base_model_name=config.base_model)
        
        # 태스크 모듈 레지스트리
        self.task_modules: Dict[str, TaskModule] = {}
        self._register_default_modules()
        
        # 핸들러 등록
        self.register_handler(
            MessageType.REASONING_REQUEST,
            self._handle_reasoning_request
        )
        self.register_handler(
            MessageType.TASK_MODULE_REQUEST,
            self._handle_task_module_request
        )
    
    def _register_default_modules(self):
        """기본 태스크 모듈 등록"""
        self.task_modules["education_material"] = TaskModule(
            name="education_material",
            description="교육 자료 생성",
            adapter_name="education_material",
            prompt_template="""다음 주제에 대한 교육 자료를 생성하세요.

주제: {topic}
대상: {audience}
형식: {format}

{context}

교육 자료:""",
        )
        
        self.task_modules["opinion_generator"] = TaskModule(
            name="opinion_generator",
            description="전문 소견 생성",
            adapter_name="opinion_generator",
            prompt_template="""다음 내용에 대한 전문적인 소견을 작성하세요.

내용: {content}
관점: {perspective}

{context}

소견:""",
        )
        
        self.task_modules["general_qa"] = TaskModule(
            name="general_qa",
            description="일반 질의응답",
            adapter_name=None,  # 기본 모델 사용
            prompt_template="""다음 질문에 답변하세요.

질문: {question}

{context}

답변:""",
        )
        
        self.task_modules["summarizer"] = TaskModule(
            name="summarizer",
            description="요약 생성",
            adapter_name="summarizer",
            prompt_template="""다음 내용을 요약하세요.

내용:
{content}

요약:""",
        )
    
    async def on_start(self):
        """시작 시 초기화"""
        self.logger.info("추론 에이전트 초기화 중...")
        
        if self.reason_config.load_model:
            try:
                # 기본 어댑터 로드
                default_adapter = self.adapter_pool.get_adapter("general_qa")
                default_adapter.initialize(load_base_model=True)
                self.logger.info("기본 모델 로드 완료")
            except Exception as e:
                self.logger.warning(f"모델 로드 실패 (시뮬레이션 모드): {e}")
        else:
            self.logger.info("시뮬레이션 모드로 실행")
    
    async def handle_message(self, message: Message) -> TaskResult:
        """기본 메시지 처리"""
        return TaskResult(
            success=False,
            error=f"처리되지 않은 메시지 유형: {message.type.value}"
        )
    
    async def _handle_reasoning_request(self, message: Message) -> TaskResult:
        """
        추론 요청 처리
        
        Payload:
            query: 추론 요청
            task_type: 태스크 유형 (선택)
            previous_results: 이전 에이전트 결과 (선택)
            parameters: 추가 파라미터
        """
        query = message.payload.get("query", "")
        task_type = message.payload.get("task_type", "general_qa")
        previous_results = message.payload.get("previous_results", {})
        parameters = message.payload.get("parameters", {})
        
        self.logger.log_action("REASONING", {"task": task_type, "query": query[:50]})
        
        try:
            # 태스크 모듈 선택
            module = self.task_modules.get(task_type)
            if not module:
                module = self.task_modules["general_qa"]
            
            # 프롬프트 구성
            prompt = self._build_prompt(query, module, previous_results, parameters)
            
            # 추론 수행
            result = await self.generate(
                prompt=prompt,
                adapter_name=module.adapter_name,
            )
            
            # 불확실도 추정 (간단한 휴리스틱)
            uncertainty = self._estimate_uncertainty(result, previous_results)
            
            # 모듈 통계 업데이트
            module.usage_count += 1
            
            return TaskResult(
                success=True,
                result={
                    "response": result,
                    "task_type": task_type,
                    "uncertainty": uncertainty,
                },
                metadata={
                    "adapter_used": module.adapter_name,
                    "prompt_length": len(prompt),
                    "response_length": len(result),
                }
            )
            
        except Exception as e:
            self.logger.error(f"추론 오류: {e}")
            return TaskResult(success=False, error=str(e))
    
    async def _handle_task_module_request(self, message: Message) -> TaskResult:
        """
        태스크 모듈 요청 처리
        
        새로운 태스크 모듈 등록/업데이트
        """
        action = message.payload.get("action", "get")
        module_name = message.payload.get("module_name")
        
        try:
            if action == "get":
                if module_name:
                    module = self.task_modules.get(module_name)
                    if module:
                        return TaskResult(
                            success=True,
                            result={
                                "name": module.name,
                                "description": module.description,
                                "adapter": module.adapter_name,
                                "usage_count": module.usage_count,
                            }
                        )
                    return TaskResult(success=False, error="모듈 없음")
                else:
                    return TaskResult(
                        success=True,
                        result={
                            "modules": [
                                {
                                    "name": m.name,
                                    "description": m.description,
                                    "usage_count": m.usage_count,
                                }
                                for m in self.task_modules.values()
                            ]
                        }
                    )
            
            elif action == "register":
                module_data = message.payload.get("module", {})
                module = TaskModule(
                    name=module_data["name"],
                    description=module_data.get("description", ""),
                    adapter_name=module_data.get("adapter_name"),
                    prompt_template=module_data.get("prompt_template", "{query}"),
                )
                self.task_modules[module.name] = module
                
                return TaskResult(
                    success=True,
                    result={"registered": module.name}
                )
            
            else:
                return TaskResult(success=False, error=f"알 수 없는 액션: {action}")
                
        except Exception as e:
            self.logger.error(f"태스크 모듈 처리 오류: {e}")
            return TaskResult(success=False, error=str(e))
    
    def _build_prompt(
        self,
        query: str,
        module: TaskModule,
        previous_results: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """프롬프트 구성"""
        # 컨텍스트 구성
        context_parts = []
        
        # 이전 결과 포함
        if previous_results:
            for key, value in previous_results.items():
                if value:
                    context_parts.append(f"[{key}]: {str(value)[:500]}")
        
        context = "\n".join(context_parts) if context_parts else ""
        
        # 템플릿 변수 준비
        template_vars = {
            "query": query,
            "question": query,
            "topic": parameters.get("topic", query),
            "content": parameters.get("content", query),
            "audience": parameters.get("audience", "일반 독자"),
            "format": parameters.get("format", "설명문"),
            "perspective": parameters.get("perspective", "전문가"),
            "context": context,
        }
        
        try:
            prompt = module.prompt_template.format(**template_vars)
        except KeyError as e:
            # 누락된 변수는 빈 문자열로
            prompt = module.prompt_template
            for key, value in template_vars.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))
        
        return prompt
    
    def _estimate_uncertainty(
        self,
        result: str,
        previous_results: Dict[str, Any]
    ) -> float:
        """
        불확실도 추정
        
        <think>
        불확실도는 여러 요소를 고려하여 추정합니다:
        1. 응답 길이 - 너무 짧거나 긴 응답은 불확실
        2. 컨텍스트 부족 - 이전 결과가 없으면 불확실도 증가
        3. 불확실성 표현 - "아마", "것 같다" 등의 표현
        실제로는 모델의 로그 확률이나 앙상블을 사용해야 합니다.
        </think>
        """
        uncertainty = 0.3  # 기본값
        
        # 응답 길이 기반
        if len(result) < 50:
            uncertainty += 0.2
        elif len(result) > 2000:
            uncertainty += 0.1
        
        # 컨텍스트 유무
        if not previous_results:
            uncertainty += 0.1
        
        # 불확실성 표현 감지
        uncertain_phrases = ["아마", "것 같", "추측", "불확실", "maybe", "perhaps", "might"]
        for phrase in uncertain_phrases:
            if phrase in result.lower():
                uncertainty += 0.05
        
        return min(uncertainty, 1.0)
    
    async def generate(
        self,
        prompt: str,
        adapter_name: str = None,
        **kwargs
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            adapter_name: 사용할 어댑터 이름
            **kwargs: 추가 생성 파라미터
            
        Returns:
            생성된 텍스트
        """
        # 어댑터 획득
        task_name = adapter_name or "general_qa"
        adapter = self.adapter_pool.get_adapter(task_name)
        
        # 초기화 (필요시)
        if not adapter._initialized:
            adapter.initialize(load_base_model=self.reason_config.load_model)
        
        # 생성
        result = adapter.generate(
            prompt=prompt,
            max_length=kwargs.get("max_length", self.reason_config.max_length),
            temperature=kwargs.get("temperature", self.reason_config.temperature),
            top_p=kwargs.get("top_p", self.reason_config.top_p),
        )
        
        return result
    
    # 외부 API
    
    def register_task_module(
        self,
        name: str,
        description: str,
        prompt_template: str,
        adapter_name: str = None
    ):
        """태스크 모듈 등록"""
        self.task_modules[name] = TaskModule(
            name=name,
            description=description,
            adapter_name=adapter_name,
            prompt_template=prompt_template,
        )
        self.logger.info(f"태스크 모듈 등록: {name}")
    
    def get_available_modules(self) -> List[Dict[str, Any]]:
        """사용 가능한 모듈 목록"""
        return [
            {
                "name": m.name,
                "description": m.description,
                "adapter": m.adapter_name,
                "usage_count": m.usage_count,
            }
            for m in self.task_modules.values()
        ]
    
    async def infer(
        self,
        query: str,
        task_type: str = "general_qa",
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        추론 수행 (외부 API)
        
        Args:
            query: 쿼리
            task_type: 태스크 유형
            context: 컨텍스트
            
        Returns:
            추론 결과
        """
        module = self.task_modules.get(task_type, self.task_modules["general_qa"])
        prompt = self._build_prompt(query, module, context or {}, {})
        
        result = await self.generate(
            prompt=prompt,
            adapter_name=module.adapter_name,
        )
        
        return {
            "response": result,
            "task_type": task_type,
            "uncertainty": self._estimate_uncertainty(result, context or {}),
        }

