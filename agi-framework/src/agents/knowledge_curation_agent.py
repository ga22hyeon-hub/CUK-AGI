"""
지식 큐레이션 에이전트
GNN 기반 그래프 정렬 및 지식 검색
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np

from .base_agent import BaseAgent, AgentConfig, TaskResult
from ..core.message_bus import Message, MessageBus, MessageType
from ..knowledge.knowledge_graph import (
    KnowledgeGraph, KGNode, NodeType, EdgeType
)
from ..knowledge.graph_embeddings import GraphEmbedder, EmbeddingConfig


@dataclass
class KnowledgeCurationConfig(AgentConfig):
    """지식 큐레이션 에이전트 설정"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gnn_model: str = "GCN"
    hidden_dim: int = 256
    num_layers: int = 3
    top_k_results: int = 10
    similarity_threshold: float = 0.5


class KnowledgeCurationAgent(BaseAgent):
    """
    지식 큐레이션 에이전트
    
    주요 기능:
    1. Multi-modal Knowledge Graph 관리
    2. 쿼리 임베딩과 그래프 기반 정렬
    3. GNN을 통한 구조적 정보 활용
    4. 컨텍스트 기반 지식 검색
    """
    
    def __init__(
        self,
        message_bus: MessageBus,
        config: KnowledgeCurationConfig = None,
        agent_id: str = None
    ):
        config = config or KnowledgeCurationConfig(
            name="knowledge_curation",
            description="GNN 기반 지식 큐레이션 에이전트"
        )
        super().__init__(config, message_bus, agent_id)
        
        self.kg_config = config
        
        # 지식 그래프
        self.knowledge_graph = KnowledgeGraph(name="main_kg")
        
        # 그래프 임베더
        embedding_config = EmbeddingConfig(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            model_type=config.gnn_model,
        )
        self.embedder = GraphEmbedder(
            config=embedding_config,
            text_encoder=config.embedding_model
        )
        
        # 핸들러 등록
        self.register_handler(
            MessageType.KNOWLEDGE_REQUEST, 
            self._handle_knowledge_request
        )
    
    async def on_start(self):
        """시작 시 초기화"""
        self.logger.info("지식 큐레이션 에이전트 초기화 중...")
        
        # 임베더 초기화
        try:
            self.embedder.initialize()
            self.logger.info("임베더 초기화 완료")
        except Exception as e:
            self.logger.warning(f"임베더 초기화 실패 (폴백 모드): {e}")
        
        # 샘플 지식 그래프 생성 (데모용)
        self._create_sample_knowledge()
    
    def _create_sample_knowledge(self):
        """샘플 지식 그래프 생성"""
        # 노드 생성
        ai_node = self.knowledge_graph.add_node(
            label="인공지능",
            node_type=NodeType.CONCEPT,
            properties={"domain": "computer_science"}
        )
        
        ml_node = self.knowledge_graph.add_node(
            label="머신러닝",
            node_type=NodeType.CONCEPT,
            properties={"domain": "computer_science"}
        )
        
        dl_node = self.knowledge_graph.add_node(
            label="딥러닝",
            node_type=NodeType.CONCEPT,
            properties={"domain": "computer_science"}
        )
        
        llm_node = self.knowledge_graph.add_node(
            label="대규모 언어 모델",
            node_type=NodeType.CONCEPT,
            properties={"domain": "nlp"}
        )
        
        transformer_node = self.knowledge_graph.add_node(
            label="트랜스포머",
            node_type=NodeType.CONCEPT,
            properties={"domain": "nlp", "year": 2017}
        )
        
        attention_node = self.knowledge_graph.add_node(
            label="어텐션 메커니즘",
            node_type=NodeType.CONCEPT,
            properties={"domain": "nlp"}
        )
        
        # 엣지 생성
        self.knowledge_graph.add_edge(
            ml_node.node_id, ai_node.node_id,
            EdgeType.IS_A, weight=1.0
        )
        
        self.knowledge_graph.add_edge(
            dl_node.node_id, ml_node.node_id,
            EdgeType.IS_A, weight=1.0
        )
        
        self.knowledge_graph.add_edge(
            llm_node.node_id, dl_node.node_id,
            EdgeType.IS_A, weight=1.0
        )
        
        self.knowledge_graph.add_edge(
            transformer_node.node_id, dl_node.node_id,
            EdgeType.IS_A, weight=1.0
        )
        
        self.knowledge_graph.add_edge(
            llm_node.node_id, transformer_node.node_id,
            EdgeType.PART_OF, weight=0.9
        )
        
        self.knowledge_graph.add_edge(
            attention_node.node_id, transformer_node.node_id,
            EdgeType.PART_OF, weight=1.0
        )
        
        self.logger.info(f"샘플 지식 그래프 생성: {self.knowledge_graph.statistics()}")
    
    async def handle_message(self, message: Message) -> TaskResult:
        """기본 메시지 처리"""
        return TaskResult(
            success=False,
            error=f"처리되지 않은 메시지 유형: {message.type.value}"
        )
    
    async def _handle_knowledge_request(self, message: Message) -> TaskResult:
        """
        지식 요청 처리
        
        Payload:
            query: 검색 쿼리
            context: 컨텍스트 정보 (선택)
            top_k: 반환할 결과 수 (선택)
        """
        query = message.payload.get("query", "")
        context = message.payload.get("context", {})
        top_k = message.payload.get("top_k", self.kg_config.top_k_results)
        
        self.logger.log_action("KNOWLEDGE_SEARCH", {"query": query[:50]})
        
        try:
            # 쿼리 기반 그래프 정렬
            aligned_nodes = self.embedder.align_query_to_graph(
                query=query,
                kg=self.knowledge_graph,
                top_k=top_k
            )
            
            # 결과 구성
            results = []
            for node_id, similarity in aligned_nodes:
                node = self.knowledge_graph.get_node(node_id)
                if node:
                    # 관련 노드 정보도 포함
                    neighbors = self.knowledge_graph.get_neighbors(node_id, "both")
                    neighbor_info = [
                        {
                            "label": n.label,
                            "relation": e.edge_type.value,
                            "weight": e.weight
                        }
                        for n, e in neighbors[:5]  # 상위 5개만
                    ]
                    
                    results.append({
                        "node_id": node_id,
                        "label": node.label,
                        "type": node.node_type.value,
                        "properties": node.properties,
                        "similarity": similarity,
                        "neighbors": neighbor_info,
                    })
            
            return TaskResult(
                success=True,
                result={
                    "query": query,
                    "results": results,
                    "graph_stats": self.knowledge_graph.statistics(),
                },
                metadata={
                    "num_results": len(results),
                    "threshold": self.kg_config.similarity_threshold,
                }
            )
            
        except Exception as e:
            self.logger.error(f"지식 검색 오류: {e}")
            return TaskResult(success=False, error=str(e))
    
    # 외부 API
    
    def add_knowledge(
        self,
        label: str,
        node_type: str = "concept",
        properties: Dict[str, Any] = None,
        relations: List[Dict[str, Any]] = None
    ) -> str:
        """
        지식 추가
        
        Args:
            label: 노드 레이블
            node_type: 노드 유형
            properties: 속성
            relations: 관계 목록 [{"target": id, "type": str, "weight": float}]
            
        Returns:
            생성된 노드 ID
        """
        node = self.knowledge_graph.add_node(
            label=label,
            node_type=NodeType(node_type),
            properties=properties or {},
        )
        
        # 관계 추가
        if relations:
            for rel in relations:
                self.knowledge_graph.add_edge(
                    source_id=node.node_id,
                    target_id=rel["target"],
                    edge_type=EdgeType(rel.get("type", "related_to")),
                    weight=rel.get("weight", 1.0),
                )
        
        self.logger.info(f"지식 추가: {label} (ID: {node.node_id})")
        return node.node_id
    
    def search_knowledge(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        지식 검색 (동기 API)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        aligned_nodes = self.embedder.align_query_to_graph(
            query=query,
            kg=self.knowledge_graph,
            top_k=top_k
        )
        
        results = []
        for node_id, similarity in aligned_nodes:
            node = self.knowledge_graph.get_node(node_id)
            if node:
                results.append({
                    "node_id": node_id,
                    "label": node.label,
                    "type": node.node_type.value,
                    "similarity": similarity,
                })
        
        return results
    
    def get_knowledge_subgraph(
        self,
        center_node_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """중심 노드 기준 서브그래프 반환"""
        subgraph = self.knowledge_graph.get_subgraph(center_node_id, max_depth)
        
        return {
            "nodes": [n.to_dict() for n in subgraph.nodes.values()],
            "edges": [e.to_dict() for e in subgraph.edges.values()],
            "statistics": subgraph.statistics(),
        }

