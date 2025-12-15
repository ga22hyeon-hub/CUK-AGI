#!/usr/bin/env python3
"""
기본 사용 예제
AGI Framework의 기본적인 사용 방법 데모
"""

import asyncio
import sys
from pathlib import Path

# 소스 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.orchestrator import Orchestrator, AGIFramework
from core.message_bus import MessageBus, Message, MessageType
from agents.moderator_agent import ModeratorAgent, ModeratorConfig
from agents.knowledge_curation_agent import KnowledgeCurationAgent
from agents.context_agent import ContextAgent
from agents.reasoning_agent import ReasoningAgent
from agents.learning_agent import LearningAgent


async def basic_query_example():
    """
    기본 쿼리 처리 예제
    
    사용자 쿼리를 받아 전체 에이전트 파이프라인을 거쳐 응답 생성
    """
    print("=" * 60)
    print("예제 1: 기본 쿼리 처리")
    print("=" * 60)
    
    # 시스템 생성 및 시작
    orchestrator = AGIFramework.create_default_system()
    await orchestrator.start()
    
    try:
        # 쿼리 처리
        queries = [
            "딥러닝이란 무엇인가요?",
            "트랜스포머 아키텍처에 대해 설명해주세요",
            "머신러닝 교육 자료를 생성해주세요",
        ]
        
        for query in queries:
            print(f"\n질문: {query}")
            print("-" * 40)
            
            result = await orchestrator.process_request(
                query=query,
                session_id="demo_session"
            )
            
            if result["success"]:
                print(f"처리 결과: {result['result']}")
            else:
                print(f"오류: {result.get('error', '알 수 없는 오류')}")
        
        # 시스템 상태 확인
        print("\n" + "=" * 60)
        print("시스템 상태:")
        status = orchestrator.get_system_status()
        print(f"  - 활성 에이전트: {status['num_agents']}개")
        print(f"  - 실행 시간: {status['uptime_seconds']:.1f}초")
        for agent in status["agents"]:
            print(f"    - {agent['name']}: {agent['state']} (작업 {agent['task_count']}개)")
    
    finally:
        await orchestrator.stop()


async def individual_agent_example():
    """
    개별 에이전트 사용 예제
    
    각 에이전트의 기능을 개별적으로 테스트
    """
    print("\n" + "=" * 60)
    print("예제 2: 개별 에이전트 사용")
    print("=" * 60)
    
    bus = MessageBus()
    
    # 1. 지식 큐레이션 에이전트
    print("\n[지식 큐레이션 에이전트]")
    knowledge_agent = KnowledgeCurationAgent(message_bus=bus)
    await knowledge_agent.on_start()
    
    # 지식 추가
    node_id = knowledge_agent.add_knowledge(
        label="Large Language Model",
        node_type="concept",
        properties={"acronym": "LLM", "domain": "NLP"}
    )
    print(f"  지식 추가: LLM (ID: {node_id})")
    
    # 지식 검색
    results = knowledge_agent.search_knowledge("언어 모델", top_k=3)
    print(f"  검색 결과: {len(results)}개 노드 발견")
    for r in results:
        print(f"    - {r['label']} (유사도: {r['similarity']:.3f})")
    
    # 2. 컨텍스트 에이전트
    print("\n[컨텍스트 에이전트]")
    context_agent = ContextAgent(message_bus=bus)
    await context_agent.on_start()
    
    # 히스토리 추가
    context_agent.add_history("session1", "user", "안녕하세요!")
    context_agent.add_history("session1", "assistant", "안녕하세요! 무엇을 도와드릴까요?")
    
    history = context_agent.get_recent_history("session1")
    print(f"  히스토리: {len(history)}개 항목")
    
    # 상황 정보 업데이트
    context_agent.update_situation("topic", "AI", activation=1.0)
    situation = context_agent.get_current_situation()
    print(f"  상황 그래프: {len(situation['nodes'])}개 노드")
    
    # 3. 추론 에이전트
    print("\n[추론 에이전트]")
    reasoning_agent = ReasoningAgent(message_bus=bus)
    await reasoning_agent.on_start()
    
    # 사용 가능한 모듈 확인
    modules = reasoning_agent.get_available_modules()
    print(f"  사용 가능한 모듈: {len(modules)}개")
    for m in modules:
        print(f"    - {m['name']}: {m['description']}")
    
    # 추론 수행
    result = await reasoning_agent.infer(
        query="GPT-4의 특징을 설명해주세요",
        task_type="general_qa"
    )
    print(f"  추론 결과 (불확실도: {result['uncertainty']:.2f}):")
    print(f"    {result['response'][:100]}...")
    
    # 4. 학습 에이전트
    print("\n[학습 에이전트]")
    learning_agent = LearningAgent(message_bus=bus)
    await learning_agent.on_start()
    
    # 응답 평가
    eval_result = await learning_agent.evaluate_response(
        prompt="AI란 무엇인가요?",
        response="AI(인공지능)는 인간의 학습능력, 추론능력, 지각능력을 컴퓨터로 구현한 기술입니다."
    )
    print(f"  평가 점수: {eval_result.score:.2f}")
    print(f"  피드백: {eval_result.feedback}")
    print(f"  세부 점수: {eval_result.aspects}")
    
    # 선호도 데이터 추가
    learning_agent.add_preference(
        prompt="딥러닝이란?",
        chosen="딥러닝은 다층 신경망을 사용하는 머신러닝의 한 분야입니다.",
        rejected="잘 모르겠습니다."
    )
    
    stats = learning_agent.get_learning_statistics()
    print(f"  학습 통계: 평가 {stats['general']['total_evaluations']}회, 선호도 {stats['general']['total_preferences']}개")


async def message_passing_example():
    """
    메시지 패싱 예제
    
    에이전트 간 메시지 통신 데모
    """
    print("\n" + "=" * 60)
    print("예제 3: 에이전트 간 메시지 패싱")
    print("=" * 60)
    
    bus = MessageBus()
    
    # 에이전트 생성
    knowledge_agent = KnowledgeCurationAgent(message_bus=bus)
    await knowledge_agent.start()
    
    # 메시지 전송
    request = Message(
        type=MessageType.KNOWLEDGE_REQUEST,
        sender="demo",
        receiver=knowledge_agent.agent_id,
        payload={"query": "인공지능", "top_k": 5}
    )
    
    print(f"  요청 전송: {request.type.value}")
    response = await bus.request(request, timeout=10.0)
    
    if response:
        print(f"  응답 수신: {response.type.value}")
        result = response.payload.get("result", {})
        print(f"  검색 결과: {len(result.get('results', []))}개 노드")
    else:
        print("  응답 타임아웃")
    
    await knowledge_agent.stop()


async def custom_module_example():
    """
    커스텀 태스크 모듈 예제
    
    새로운 태스크 모듈 등록 및 사용
    """
    print("\n" + "=" * 60)
    print("예제 4: 커스텀 태스크 모듈")
    print("=" * 60)
    
    bus = MessageBus()
    reasoning_agent = ReasoningAgent(message_bus=bus)
    await reasoning_agent.on_start()
    
    # 커스텀 모듈 등록
    reasoning_agent.register_task_module(
        name="code_generator",
        description="코드 생성 모듈",
        prompt_template="""다음 요구사항에 맞는 Python 코드를 생성하세요.

요구사항: {query}

```python
# 코드 시작
""",
        adapter_name=None  # 기본 모델 사용
    )
    
    print("  커스텀 모듈 등록: code_generator")
    
    # 커스텀 모듈로 추론
    result = await reasoning_agent.infer(
        query="리스트의 모든 요소를 두 배로 만드는 함수",
        task_type="code_generator"
    )
    
    print(f"  생성 결과:")
    print(f"    {result['response']}")


async def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("AGI Framework 데모")
    print("Multi-LLM Agent 기반 AGI 프레임워크")
    print("=" * 60)
    
    await basic_query_example()
    await individual_agent_example()
    await message_passing_example()
    await custom_module_example()
    
    print("\n" + "=" * 60)
    print("데모 완료!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

