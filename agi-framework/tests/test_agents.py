"""
에이전트 단위 테스트
"""

import pytest
import asyncio
from datetime import datetime

# 테스트 대상 모듈
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.message_bus import MessageBus, Message, MessageType
from agents.base_agent import BaseAgent, AgentConfig, AgentState, TaskResult
from agents.moderator_agent import ModeratorAgent, ModeratorConfig, IntentType
from agents.knowledge_curation_agent import KnowledgeCurationAgent, KnowledgeCurationConfig
from agents.context_agent import ContextAgent, ContextConfig
from agents.reasoning_agent import ReasoningAgent, ReasoningConfig
from agents.learning_agent import LearningAgent, LearningConfig


class TestMessageBus:
    """메시지 버스 테스트"""
    
    def test_agent_registration(self):
        """에이전트 등록 테스트"""
        bus = MessageBus()
        queue = bus.register_agent("test_agent")
        
        assert "test_agent" in bus.queues
        assert queue is not None
    
    @pytest.mark.asyncio
    async def test_message_publish(self):
        """메시지 발행 테스트"""
        bus = MessageBus()
        bus.register_agent("receiver")
        
        message = Message(
            type=MessageType.TASK_ASSIGNMENT,
            sender="sender",
            receiver="receiver",
            payload={"test": "data"},
        )
        
        await bus.publish(message)
        
        # 메시지 수신 확인
        received = await bus.queues["receiver"].get()
        assert received.payload["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_request_response(self):
        """요청-응답 패턴 테스트"""
        bus = MessageBus()
        bus.register_agent("responder")
        
        async def mock_responder():
            """응답자 시뮬레이션"""
            msg = await bus.queues["responder"].get()
            response = msg.create_response(
                MessageType.KNOWLEDGE_RESPONSE,
                {"answer": "test_answer"}
            )
            await bus.respond(response)
        
        # 응답자 시작
        asyncio.create_task(mock_responder())
        
        # 요청 전송
        request = Message(
            type=MessageType.KNOWLEDGE_REQUEST,
            sender="requester",
            receiver="responder",
            payload={"question": "test"},
        )
        
        response = await bus.request(request, timeout=5.0)
        
        assert response is not None
        assert response.payload["answer"] == "test_answer"


class TestModeratorAgent:
    """모더레이터 에이전트 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.bus = MessageBus()
        self.agent = ModeratorAgent(
            message_bus=self.bus,
            config=ModeratorConfig(name="moderator", description="테스트")
        )
    
    def test_intent_classification_knowledge(self):
        """지식 쿼리 의도 분류 테스트"""
        intent, confidence = self.agent.classify_intent("인공지능이 무엇인가요?")
        
        assert intent == IntentType.KNOWLEDGE_QUERY
        assert confidence > 0
    
    def test_intent_classification_task(self):
        """작업 실행 의도 분류 테스트"""
        intent, confidence = self.agent.classify_intent("보고서를 생성해주세요")
        
        assert intent == IntentType.TASK_EXECUTION
        assert confidence > 0
    
    def test_intent_classification_learning(self):
        """학습 요청 의도 분류 테스트"""
        intent, confidence = self.agent.classify_intent("이 패턴을 학습해주세요")
        
        assert intent == IntentType.LEARNING_REQUEST
    
    @pytest.mark.asyncio
    async def test_execution_plan_creation(self):
        """실행 계획 생성 테스트"""
        plan = await self.agent.create_execution_plan(
            query="딥러닝에 대해 설명해주세요",
            intent=IntentType.KNOWLEDGE_QUERY,
            context={}
        )
        
        assert plan.plan_id is not None
        assert len(plan.subtasks) > 0
        assert plan.status == "created"


class TestKnowledgeCurationAgent:
    """지식 큐레이션 에이전트 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.bus = MessageBus()
        self.agent = KnowledgeCurationAgent(
            message_bus=self.bus,
            config=KnowledgeCurationConfig(name="knowledge", description="테스트")
        )
    
    @pytest.mark.asyncio
    async def test_sample_knowledge_creation(self):
        """샘플 지식 생성 테스트"""
        await self.agent.on_start()
        
        stats = self.agent.knowledge_graph.statistics()
        assert stats["num_nodes"] > 0
        assert stats["num_edges"] > 0
    
    def test_knowledge_addition(self):
        """지식 추가 테스트"""
        node_id = self.agent.add_knowledge(
            label="테스트 개념",
            node_type="concept",
            properties={"domain": "test"}
        )
        
        assert node_id is not None
        node = self.agent.knowledge_graph.get_node(node_id)
        assert node.label == "테스트 개념"
    
    @pytest.mark.asyncio
    async def test_knowledge_search(self):
        """지식 검색 테스트"""
        await self.agent.on_start()
        
        results = self.agent.search_knowledge("인공지능", top_k=3)
        
        assert len(results) > 0
        assert "similarity" in results[0]


class TestContextAgent:
    """컨텍스트 에이전트 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.bus = MessageBus()
        self.agent = ContextAgent(
            message_bus=self.bus,
            config=ContextConfig(name="context", description="테스트", db_path=":memory:")
        )
    
    @pytest.mark.asyncio
    async def test_history_management(self):
        """히스토리 관리 테스트"""
        await self.agent.on_start()
        
        # 히스토리 추가
        entry = self.agent.add_history(
            session_id="test_session",
            role="user",
            content="테스트 메시지입니다"
        )
        
        assert entry.entry_id is not None
        
        # 히스토리 조회
        history = self.agent.get_recent_history("test_session")
        assert len(history) == 1
        assert history[0].content == "테스트 메시지입니다"
    
    @pytest.mark.asyncio
    async def test_situation_graph(self):
        """상황 정보 그래프 테스트"""
        await self.agent.on_start()
        
        # 상황 정보 추가
        self.agent.update_situation(
            node_type="topic",
            label="AI",
            activation=1.0
        )
        
        situation = self.agent.get_current_situation()
        assert len(situation["nodes"]) > 0


class TestReasoningAgent:
    """추론 에이전트 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.bus = MessageBus()
        self.agent = ReasoningAgent(
            message_bus=self.bus,
            config=ReasoningConfig(name="reasoning", description="테스트", load_model=False)
        )
    
    def test_task_modules_registered(self):
        """태스크 모듈 등록 확인"""
        modules = self.agent.get_available_modules()
        
        assert len(modules) > 0
        module_names = [m["name"] for m in modules]
        assert "general_qa" in module_names
        assert "education_material" in module_names
    
    @pytest.mark.asyncio
    async def test_inference(self):
        """추론 테스트 (시뮬레이션)"""
        await self.agent.on_start()
        
        result = await self.agent.infer(
            query="인공지능에 대해 설명해주세요",
            task_type="general_qa"
        )
        
        assert "response" in result
        assert "uncertainty" in result
        assert 0 <= result["uncertainty"] <= 1


class TestLearningAgent:
    """학습 에이전트 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.bus = MessageBus()
        self.agent = LearningAgent(
            message_bus=self.bus,
            config=LearningConfig(name="learning", description="테스트", auto_learn=False)
        )
    
    @pytest.mark.asyncio
    async def test_response_evaluation(self):
        """응답 평가 테스트"""
        await self.agent.on_start()
        
        result = await self.agent.evaluate_response(
            prompt="딥러닝에 대해 설명해주세요",
            response="딥러닝은 인공신경망을 활용한 머신러닝의 한 분야입니다. 여러 층의 뉴런을 통해 복잡한 패턴을 학습할 수 있습니다."
        )
        
        assert 0 <= result.score <= 1
        assert result.feedback is not None
        assert "relevance" in result.aspects
    
    def test_preference_data_addition(self):
        """선호도 데이터 추가 테스트"""
        self.agent.add_preference(
            prompt="테스트 질문",
            chosen="좋은 응답입니다",
            rejected="나쁜 응답입니다"
        )
        
        assert len(self.agent.preference_dataset) == 1
    
    def test_learning_statistics(self):
        """학습 통계 테스트"""
        stats = self.agent.get_learning_statistics()
        
        assert "general" in stats
        assert "preference_dataset" in stats
        assert "ppo_buffer" in stats


# 실행
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

