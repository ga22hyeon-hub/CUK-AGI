"""
지식 그래프
Multi-modal Knowledge Graph 구현
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import numpy as np


class NodeType(Enum):
    """노드 유형"""
    ENTITY = "entity"
    CONCEPT = "concept"
    EVENT = "event"
    DOCUMENT = "document"
    RELATION = "relation"


class EdgeType(Enum):
    """엣지 유형"""
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSES = "causes"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"


@dataclass
class KGNode:
    """지식 그래프 노드"""
    
    node_id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if isinstance(other, KGNode):
            return self.node_id == other.node_id
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "properties": self.properties,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class KGEdge:
    """지식 그래프 엣지"""
    
    edge_id: str
    edge_type: EdgeType
    source_id: str
    target_id: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        return hash(self.edge_id)
    
    def __eq__(self, other):
        if isinstance(other, KGEdge):
            return self.edge_id == other.edge_id
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
        }


class KnowledgeGraph:
    """
    Multi-modal 지식 그래프
    
    노드와 엣지로 구성된 그래프 구조로 지식 저장 및 검색
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.nodes: Dict[str, KGNode] = {}
        self.edges: Dict[str, KGEdge] = {}
        
        # 인덱스
        self._adjacency: Dict[str, Set[str]] = {}  # node_id -> set of edge_ids
        self._reverse_adjacency: Dict[str, Set[str]] = {}  # node_id -> set of incoming edge_ids
        self._type_index: Dict[NodeType, Set[str]] = {t: set() for t in NodeType}
    
    def add_node(
        self,
        label: str,
        node_type: NodeType,
        properties: Dict[str, Any] = None,
        embedding: np.ndarray = None,
        node_id: str = None
    ) -> KGNode:
        """
        노드 추가
        
        Args:
            label: 노드 레이블
            node_type: 노드 유형
            properties: 추가 속성
            embedding: 임베딩 벡터
            node_id: 노드 ID (선택)
            
        Returns:
            생성된 노드
        """
        node_id = node_id or str(uuid.uuid4())
        
        node = KGNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            properties=properties or {},
            embedding=embedding,
        )
        
        self.nodes[node_id] = node
        self._adjacency[node_id] = set()
        self._reverse_adjacency[node_id] = set()
        self._type_index[node_type].add(node_id)
        
        return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        properties: Dict[str, Any] = None,
        edge_id: str = None
    ) -> Optional[KGEdge]:
        """
        엣지 추가
        
        Args:
            source_id: 소스 노드 ID
            target_id: 타겟 노드 ID
            edge_type: 엣지 유형
            weight: 가중치
            properties: 추가 속성
            edge_id: 엣지 ID (선택)
            
        Returns:
            생성된 엣지 또는 None (노드가 없는 경우)
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        edge_id = edge_id or str(uuid.uuid4())
        
        edge = KGEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            properties=properties or {},
        )
        
        self.edges[edge_id] = edge
        self._adjacency[source_id].add(edge_id)
        self._reverse_adjacency[target_id].add(edge_id)
        
        return edge
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """노드 조회"""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[KGEdge]:
        """엣지 조회"""
        return self.edges.get(edge_id)
    
    def get_neighbors(
        self, 
        node_id: str, 
        direction: str = "outgoing"
    ) -> List[Tuple[KGNode, KGEdge]]:
        """
        이웃 노드 조회
        
        Args:
            node_id: 노드 ID
            direction: "outgoing", "incoming", "both"
            
        Returns:
            (이웃 노드, 연결 엣지) 튜플 리스트
        """
        neighbors = []
        
        if direction in ("outgoing", "both"):
            for edge_id in self._adjacency.get(node_id, set()):
                edge = self.edges[edge_id]
                neighbor = self.nodes[edge.target_id]
                neighbors.append((neighbor, edge))
        
        if direction in ("incoming", "both"):
            for edge_id in self._reverse_adjacency.get(node_id, set()):
                edge = self.edges[edge_id]
                neighbor = self.nodes[edge.source_id]
                neighbors.append((neighbor, edge))
        
        return neighbors
    
    def find_nodes_by_type(self, node_type: NodeType) -> List[KGNode]:
        """유형별 노드 검색"""
        return [self.nodes[nid] for nid in self._type_index[node_type]]
    
    def find_nodes_by_label(self, label: str, fuzzy: bool = False) -> List[KGNode]:
        """레이블로 노드 검색"""
        if fuzzy:
            label_lower = label.lower()
            return [n for n in self.nodes.values() if label_lower in n.label.lower()]
        return [n for n in self.nodes.values() if n.label == label]
    
    def get_subgraph(
        self, 
        center_node_id: str, 
        max_depth: int = 2
    ) -> "KnowledgeGraph":
        """
        중심 노드 기준 서브그래프 추출
        
        Args:
            center_node_id: 중심 노드 ID
            max_depth: 최대 탐색 깊이
            
        Returns:
            서브그래프
        """
        subgraph = KnowledgeGraph(name=f"{self.name}_subgraph")
        
        visited = set()
        queue = [(center_node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # 노드 복사
            if current_id in self.nodes:
                node = self.nodes[current_id]
                subgraph.add_node(
                    label=node.label,
                    node_type=node.node_type,
                    properties=node.properties.copy(),
                    embedding=node.embedding.copy() if node.embedding is not None else None,
                    node_id=node.node_id,
                )
            
            # 이웃 탐색
            if depth < max_depth:
                for neighbor, edge in self.get_neighbors(current_id, "both"):
                    queue.append((neighbor.node_id, depth + 1))
        
        # 엣지 복사
        for edge in self.edges.values():
            if edge.source_id in visited and edge.target_id in visited:
                subgraph.add_edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    properties=edge.properties.copy(),
                    edge_id=edge.edge_id,
                )
        
        return subgraph
    
    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """인접 행렬로 변환"""
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        matrix = np.zeros((n, n))
        
        for edge in self.edges.values():
            i = id_to_idx[edge.source_id]
            j = id_to_idx[edge.target_id]
            matrix[i, j] = edge.weight
        
        return matrix, node_ids
    
    def to_edge_index(self) -> Tuple[np.ndarray, List[str]]:
        """PyTorch Geometric 형식의 edge_index로 변환"""
        node_ids = list(self.nodes.keys())
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        sources = []
        targets = []
        
        for edge in self.edges.values():
            sources.append(id_to_idx[edge.source_id])
            targets.append(id_to_idx[edge.target_id])
        
        edge_index = np.array([sources, targets])
        
        return edge_index, node_ids
    
    def get_node_features(self, node_ids: List[str] = None) -> np.ndarray:
        """노드 특징 행렬 반환"""
        if node_ids is None:
            node_ids = list(self.nodes.keys())
        
        features = []
        for nid in node_ids:
            node = self.nodes[nid]
            if node.embedding is not None:
                features.append(node.embedding)
            else:
                # 기본 임베딩 (영벡터)
                features.append(np.zeros(384))
        
        return np.array(features)
    
    def statistics(self) -> Dict[str, Any]:
        """그래프 통계"""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_types": {t.value: len(ids) for t, ids in self._type_index.items()},
            "avg_degree": sum(len(e) for e in self._adjacency.values()) / max(len(self.nodes), 1),
        }
    
    def merge(self, other: "KnowledgeGraph"):
        """다른 그래프와 병합"""
        # 노드 병합
        for node in other.nodes.values():
            if node.node_id not in self.nodes:
                self.add_node(
                    label=node.label,
                    node_type=node.node_type,
                    properties=node.properties.copy(),
                    embedding=node.embedding.copy() if node.embedding is not None else None,
                    node_id=node.node_id,
                )
        
        # 엣지 병합
        for edge in other.edges.values():
            if edge.edge_id not in self.edges:
                self.add_edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    properties=edge.properties.copy(),
                    edge_id=edge.edge_id,
                )

