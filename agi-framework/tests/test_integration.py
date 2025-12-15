"""
통합 테스트
에이전트 간 연계 흐름 검증
"""

import pytest
import asyncio
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.orchestrator import Orchestrator, OrchestratorConfig, AGIFramework
from core.message_bus import MessageBus, Message, MessageType
from agents.moderator_agent import ModeratorAgent, ModeratorConfig
from agents.knowledge_curation_agent import KnowledgeCurationAgent, KnowledgeCurationConfig
from agents.context_agent import ContextAgent, ContextConfig
from agents.reasoning_agent import ReasoningAgent, ReasoningConfig
from agents.learning_agent import LearningAgent, LearningConfig


class TestAgentIntegration:
    """에이전트 통합 테스트"""
    
    @pytest.fixture
    async def orchestrator(self):
        """오케스트레이터 픽스처"""
        orch = AGIFramework.create_default_system()
        await orch.start()
        yield orch
        await orch.stop()
    
    @pytest.mark.asyncio
    async def test_system_startup(self, orchestrator):
        """시스템 시작 테스트"""
        status = orchestrator.get_system_status()
        
        assert status["running"] == True
        assert status["num_agents"] == 5  # 5개 에이전트
    
    @pytest.mark.asyncio
    async def test_knowledge_query_flow(self, orchestrator):
        """지식 쿼리 흐름 테스트
        
        흐름:
        1. 사용자 쿼리 -> 모더레이터
        2. 모더레이터 -> 의도 분류 (knowledge_query)
        3. 모더레이터 -> 컨텍스트 에이전트 (컨텍스트 조회)
        4. 모더레이터 -> 지식 큐레이션 에이전트 (지식 검색)
        5. 모더레이터 -> 추론 에이전트 (응답 생성)
        6. 결과 반환
        """
        result = await orchestrator.process_request(
            query="딥러닝이란 무엇인가요?",
            session_id="test_session"
        )
        
        assert result["success"] == True
        assert "result" in result
    
    @pytest.mark.asyncio
    async def test_task_execution_flow(self, orchestrator):
        """작업 실행 흐름 테스트"""
        result = await orchestrator.process_request(
            query="AI 기술 동향에 대한 보고서를 생성해주세요",
            session_id="test_session"
        )
        
        assert result["success"] == True
    
    @pytest.mark.asyncio
    async def test_context_persistence(self, orchestrator):
        """컨텍스트 유지 테스트"""
        session_id = "persistence_test"
        
        # 첫 번째 요청
        await orchestrator.process_request(
            query="인공지능에 대해 알려주세요",
            session_id=session_id
        )
        
        # 컨텍스트 에이전트에서 히스토리 확인
        context_agent = orchestrator.get_agent_by_type("context")
        history = context_agent.get_recent_history(session_id)
        
        assert len(history) >= 1  # 최소 사용자 입력 저장
    
    @pytest.mark.asyncio
    async def test_multi_agent_communication(self, orchestrator):
        """다중 에이전트 통신 테스트"""
        # 지식 에이전트에 직접 메시지 전송
        knowledge_agent = orchestrator.get_agent_by_type("knowledge_curation")
        
        message = Message(
            type=MessageType.KNOWLEDGE_REQUEST,
            sender="test",
            receiver=knowledge_agent.agent_id,
            payload={"query": "트랜스포머", "top_k": 3}
        )
        
        response = await orchestrator.message_bus.request(message, timeout=10.0)
        
        assert response is not None
        assert response.type == MessageType.KNOWLEDGE_RESPONSE


class TestEndToEndScenarios:
    """End-to-End 시나리오 테스트"""
    
    @pytest.fixture
    async def system(self):
        """시스템 픽스처"""
        orch = AGIFramework.create_default_system()
        await orch.start()
        yield orch
        await orch.stop()
    
    @pytest.mark.asyncio
    async def test_educational_content_generation(self, system):
        """
        시나리오: 교육 콘텐츠 생성
        
        1. 사용자가 특정 주제에 대한 교육 자료 요청
        2. 지식 그래프에서 관련 정보 검색
        3. 추론 에이전트가 교육 자료 생성
        4. 학습 에이전트가 품질 평가
        """
        # 교육 자료 생성 요청
        result = await system.process_request(
            query="머신러닝의 기초에 대한 교육 자료를 생성해주세요",
            context={"audience": "초보자", "format": "설명문"},
            session_id="edu_test"
        )
        
        assert result["success"] == True
        
        # 학습 에이전트로 평가
        learning_agent = system.get_agent_by_type("learning")
        eval_result = await learning_agent.evaluate_response(
            prompt="머신러닝의 기초에 대한 교육 자료를 생성해주세요",
            response="머신러닝은 데이터로부터 패턴을 학습하는 AI 기술입니다..."
        )
        
        assert eval_result.score > 0
    
    @pytest.mark.asyncio
    async def test_self_improvement_loop(self, system):
        """
        시나리오: 자기 개선 루프
        
        1. 응답 생성
        2. 자기 평가
        3. 선호도 데이터 생성
        4. 학습 수행
        """
        learning_agent = system.get_agent_by_type("learning")
        
        # 두 개의 응답 생성 및 평가
        responses = [
            "딥러닝은 다층 신경망을 사용하는 머신러닝 기법입니다.",
            "잘 모르겠습니다."
        ]
        
        scores = []
        for resp in responses:
            eval_result = await learning_agent.evaluate_response(
                prompt="딥러닝이란?",
                response=resp
            )
            scores.append(eval_result.score)
        
        # 선호도 데이터 추가 (좋은 응답 vs 나쁜 응답)
        learning_agent.add_preference(
            prompt="딥러닝이란?",
            chosen=responses[0],  # 높은 점수
            rejected=responses[1],  # 낮은 점수
        )
        
        # 통계 확인
        stats = learning_agent.get_learning_statistics()
        assert stats["general"]["total_preferences"] >= 1
        assert stats["general"]["total_evaluations"] >= 2
    
    @pytest.mark.asyncio
    async def test_context_aware_conversation(self, system):
        """
        시나리오: 컨텍스트 인식 대화
        
        1. 첫 번째 질문
        2. 컨텍스트 저장
        3. 후속 질문 (이전 컨텍스트 활용)
        """
        session_id = "context_aware_test"
        
        # 첫 번째 질문
        await system.process_request(
            query="GPT란 무엇인가요?",
            session_id=session_id
        )
        
        # 두 번째 질문 (컨텍스트 필요)
        result = await system.process_request(
            query="그것의 장점은 무엇인가요?",
            session_id=session_id
        )
        
        # 컨텍스트 에이전트 상태 확인
        context_agent = system.get_agent_by_type("context")
        history = context_agent.get_recent_history(session_id)
        
        assert len(history) >= 2  # 두 개의 대화 기록


class TestFailureRecovery:
    """장애 복구 테스트"""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """타임아웃 처리 테스트"""
        bus = MessageBus()
        bus.register_agent("slow_agent")
        
        message = Message(
            type=MessageType.TASK_ASSIGNMENT,
            sender="test",
            receiver="slow_agent",
            payload={"test": "data"}
        )
        
        # 응답 없이 타임아웃
        response = await bus.request(message, timeout=0.1)
        
        assert response is None  # 타임아웃시 None 반환
    
    @pytest.mark.asyncio
    async def test_agent_error_state(self):
        """에이전트 오류 상태 테스트"""
        bus = MessageBus()
        agent = ModeratorAgent(
            message_bus=bus,
            config=ModeratorConfig(name="test_mod", description="테스트")
        )
        
        await agent.start()
        
        # 잘못된 메시지로 오류 유발
        invalid_message = Message(
            type=MessageType.ERROR,
            sender="test",
            receiver=agent.agent_id,
            payload={"error": "테스트 오류"}
        )
        
        await bus.publish(invalid_message)
        
        # 에이전트가 여전히 동작하는지 확인
        await asyncio.sleep(0.1)
        status = agent.get_status()
        
        assert status["state"] != "stopped"
        
        await agent.stop()


# 실행
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

