#!/usr/bin/env python3
"""
시나리오 기반 데모
실제 사용 시나리오를 통한 AGI Framework 검증
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.orchestrator import AGIFramework


async def scenario_educational_assistant():
    """
    시나리오 1: 교육 어시스턴트
    
    학습자의 질문에 답하고, 교육 자료를 생성하며,
    학습 진행 상황을 추적하는 시나리오
    """
    print("\n" + "=" * 70)
    print("시나리오 1: AI 교육 어시스턴트")
    print("=" * 70)
    
    orchestrator = AGIFramework.create_default_system()
    await orchestrator.start()
    
    session_id = "education_session"
    
    try:
        # 대화 시뮬레이션
        conversations = [
            ("학습자", "머신러닝이 뭔가요? 쉽게 설명해주세요."),
            ("학습자", "지도학습과 비지도학습의 차이점은?"),
            ("학습자", "신경망의 기본 원리를 설명해주세요"),
            ("학습자", "이 내용들을 요약해서 학습 노트로 만들어주세요"),
        ]
        
        for role, query in conversations:
            print(f"\n[{role}] {query}")
            print("-" * 50)
            
            result = await orchestrator.process_request(
                query=query,
                session_id=session_id,
                context={"role": "learner", "level": "beginner"}
            )
            
            if result["success"]:
                intent = result["result"].get("intent", "unknown")
                print(f"[시스템] 의도: {intent}")
                
                # 실행 결과 표시
                exec_result = result["result"].get("execution_result", {})
                task_results = exec_result.get("task_results", {})
                
                if task_results:
                    for task_id, task_result in task_results.items():
                        if task_result:
                            print(f"  [{task_id}] {str(task_result)[:200]}...")
            else:
                print(f"[오류] {result.get('error')}")
        
        # 학습 에이전트 통계 확인
        learning_agent = orchestrator.get_agent_by_type("learning")
        if learning_agent:
            stats = learning_agent.get_learning_statistics()
            print(f"\n학습 통계:")
            print(f"  - 평가 횟수: {stats['general']['total_evaluations']}")
            
    finally:
        await orchestrator.stop()


async def scenario_knowledge_exploration():
    """
    시나리오 2: 지식 탐색
    
    지식 그래프를 탐색하며 관련 개념을 발견하고
    연결 관계를 이해하는 시나리오
    """
    print("\n" + "=" * 70)
    print("시나리오 2: 지식 그래프 탐색")
    print("=" * 70)
    
    orchestrator = AGIFramework.create_default_system()
    await orchestrator.start()
    
    try:
        # 지식 에이전트 접근
        knowledge_agent = orchestrator.get_agent_by_type("knowledge_curation")
        
        # 지식 그래프 확장
        print("\n[지식 그래프 확장]")
        
        # 추가 지식 등록
        additional_knowledge = [
            ("GPT", "concept", {"full_name": "Generative Pre-trained Transformer"}),
            ("BERT", "concept", {"full_name": "Bidirectional Encoder Representations"}),
            ("어텐션", "concept", {"type": "mechanism"}),
            ("자연어처리", "concept", {"domain": "AI"}),
        ]
        
        node_ids = {}
        for label, node_type, props in additional_knowledge:
            node_id = knowledge_agent.add_knowledge(
                label=label,
                node_type=node_type,
                properties=props
            )
            node_ids[label] = node_id
            print(f"  + {label} 추가 (ID: {node_id[:8]})")
        
        # 그래프 통계
        stats = knowledge_agent.knowledge_graph.statistics()
        print(f"\n[그래프 통계]")
        print(f"  - 총 노드: {stats['num_nodes']}개")
        print(f"  - 총 엣지: {stats['num_edges']}개")
        print(f"  - 평균 차수: {stats['avg_degree']:.2f}")
        
        # 지식 검색
        print("\n[지식 검색]")
        search_queries = ["언어 모델", "신경망", "학습"]
        
        for query in search_queries:
            print(f"\n  쿼리: '{query}'")
            results = knowledge_agent.search_knowledge(query, top_k=3)
            for r in results:
                print(f"    - {r['label']} (유사도: {r['similarity']:.3f})")
        
        # 서브그래프 추출
        print("\n[서브그래프 추출]")
        if node_ids:
            first_id = list(node_ids.values())[0]
            subgraph = knowledge_agent.get_knowledge_subgraph(first_id, max_depth=2)
            print(f"  중심 노드에서 2단계 이내: {len(subgraph['nodes'])}개 노드")
    
    finally:
        await orchestrator.stop()


async def scenario_self_improvement():
    """
    시나리오 3: 자기 개선 루프
    
    응답 품질을 평가하고, 피드백을 통해
    지속적으로 개선하는 시나리오
    """
    print("\n" + "=" * 70)
    print("시나리오 3: 자기 평가 기반 개선 루프")
    print("=" * 70)
    
    orchestrator = AGIFramework.create_default_system()
    await orchestrator.start()
    
    try:
        learning_agent = orchestrator.get_agent_by_type("learning")
        reasoning_agent = orchestrator.get_agent_by_type("reasoning")
        
        # 테스트 프롬프트
        test_prompt = "강화학습의 핵심 원리를 설명하세요"
        
        print(f"\n[테스트 프롬프트] {test_prompt}")
        
        # 여러 응답 생성 및 평가
        print("\n[응답 생성 및 평가]")
        responses = []
        
        for i in range(3):
            # 추론 수행
            inference_result = await reasoning_agent.infer(
                query=test_prompt,
                task_type="general_qa"
            )
            response = inference_result["response"]
            
            # 평가 수행
            eval_result = await learning_agent.evaluate_response(
                prompt=test_prompt,
                response=response
            )
            
            responses.append({
                "response": response,
                "score": eval_result.score,
                "aspects": eval_result.aspects,
                "feedback": eval_result.feedback
            })
            
            print(f"\n  응답 {i+1}:")
            print(f"    내용: {response[:100]}...")
            print(f"    점수: {eval_result.score:.3f}")
            print(f"    피드백: {eval_result.feedback}")
        
        # 최고/최저 응답으로 선호도 데이터 생성
        sorted_responses = sorted(responses, key=lambda x: x["score"], reverse=True)
        
        if len(sorted_responses) >= 2:
            best = sorted_responses[0]
            worst = sorted_responses[-1]
            
            print(f"\n[선호도 데이터 생성]")
            print(f"  최고 점수: {best['score']:.3f}")
            print(f"  최저 점수: {worst['score']:.3f}")
            
            learning_agent.add_preference(
                prompt=test_prompt,
                chosen=best["response"],
                rejected=worst["response"],
                metadata={
                    "best_score": best["score"],
                    "worst_score": worst["score"],
                    "score_diff": best["score"] - worst["score"]
                }
            )
        
        # 학습 통계
        stats = learning_agent.get_learning_statistics()
        print(f"\n[학습 통계]")
        print(f"  - 총 평가: {stats['general']['total_evaluations']}회")
        print(f"  - 선호도 데이터: {stats['general']['total_preferences']}개")
        print(f"  - PPO 버퍼: {stats['ppo_buffer'].get('buffer_size', 0)}개 경험")
        
        if stats['ppo_buffer'].get('buffer_size', 0) > 0:
            print(f"  - 평균 보상: {stats['ppo_buffer'].get('mean_reward', 0):.3f}")
    
    finally:
        await orchestrator.stop()


async def scenario_multi_turn_dialogue():
    """
    시나리오 4: 다중 턴 대화
    
    컨텍스트를 유지하며 연속적인 대화를 
    처리하는 시나리오
    """
    print("\n" + "=" * 70)
    print("시나리오 4: 다중 턴 대화 (컨텍스트 유지)")
    print("=" * 70)
    
    orchestrator = AGIFramework.create_default_system()
    await orchestrator.start()
    
    session_id = "multi_turn_session"
    
    try:
        context_agent = orchestrator.get_agent_by_type("context")
        
        # 다중 턴 대화
        dialogue = [
            "트랜스포머에 대해 알려주세요",
            "그것의 주요 구성요소는 무엇인가요?",
            "어텐션 메커니즘에 대해 더 자세히 설명해주세요",
            "지금까지 대화 내용을 정리해주세요",
        ]
        
        for turn, query in enumerate(dialogue, 1):
            print(f"\n[턴 {turn}] 사용자: {query}")
            
            result = await orchestrator.process_request(
                query=query,
                session_id=session_id
            )
            
            if result["success"]:
                print(f"[시스템] 처리 완료")
            
            # 현재 컨텍스트 상태
            history = context_agent.get_recent_history(session_id, limit=10)
            situation = context_agent.get_current_situation()
            
            print(f"  - 히스토리 항목: {len(history)}개")
            print(f"  - 활성 상황 노드: {len(situation['nodes'])}개")
            
            # 활성화된 토픽 표시
            active_topics = [
                n["label"] for n in situation["nodes"] 
                if n.get("activation", 0) > 0.5
            ]
            if active_topics:
                print(f"  - 활성 토픽: {', '.join(active_topics[:5])}")
    
    finally:
        await orchestrator.stop()


async def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("AGI Framework 시나리오 기반 검증")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 각 시나리오 실행
    await scenario_educational_assistant()
    await scenario_knowledge_exploration()
    await scenario_self_improvement()
    await scenario_multi_turn_dialogue()
    
    print("\n" + "=" * 70)
    print("모든 시나리오 검증 완료!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

