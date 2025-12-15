"""
컨텍스트 에이전트
장기 문맥 관리 및 상황 정보 그래프 구성
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import uuid

from .base_agent import BaseAgent, AgentConfig, TaskResult
from ..core.message_bus import Message, MessageBus, MessageType


@dataclass
class HistoryEntry:
    """히스토리 항목"""
    entry_id: str
    session_id: str
    role: str  # user, assistant, system
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SituationNode:
    """상황 정보 그래프 노드"""
    node_id: str
    node_type: str  # entity, event, topic, emotion
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    activation: float = 1.0  # 활성화 수준 (시간에 따라 감쇠)
    last_activated: datetime = field(default_factory=datetime.now)
    
    def decay(self, decay_rate: float = 0.95) -> float:
        """활성화 수준 감쇠"""
        elapsed = (datetime.now() - self.last_activated).total_seconds()
        decay_factor = decay_rate ** (elapsed / 60)  # 분 단위 감쇠
        self.activation *= decay_factor
        return self.activation


class SituationGraph:
    """
    상황 정보 그래프
    현재 대화 문맥에서 활성화된 개념/엔티티 관리
    """
    
    def __init__(self, max_nodes: int = 500, decay_rate: float = 0.95):
        self.max_nodes = max_nodes
        self.decay_rate = decay_rate
        self.nodes: Dict[str, SituationNode] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = {}  # node_id -> [(target_id, weight)]
    
    def add_node(
        self,
        node_type: str,
        label: str,
        properties: Dict[str, Any] = None,
        activation: float = 1.0
    ) -> SituationNode:
        """노드 추가 또는 활성화 업데이트"""
        # 기존 노드 확인 (레이블 기준)
        for node in self.nodes.values():
            if node.label == label and node.node_type == node_type:
                node.activation = min(node.activation + activation, 2.0)
                node.last_activated = datetime.now()
                return node
        
        # 새 노드 생성
        node_id = str(uuid.uuid4())[:8]
        node = SituationNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            properties=properties or {},
            activation=activation,
        )
        
        self.nodes[node_id] = node
        self.edges[node_id] = []
        
        # 노드 수 제한
        self._prune_inactive_nodes()
        
        return node
    
    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0):
        """엣지 추가"""
        if source_id in self.nodes and target_id in self.nodes:
            self.edges[source_id].append((target_id, weight))
    
    def get_active_context(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """활성화된 컨텍스트 반환"""
        # 감쇠 적용
        for node in self.nodes.values():
            node.decay(self.decay_rate)
        
        # 임계값 이상인 노드들
        active_nodes = [
            {
                "node_id": n.node_id,
                "type": n.node_type,
                "label": n.label,
                "activation": n.activation,
                "properties": n.properties,
            }
            for n in self.nodes.values()
            if n.activation >= threshold
        ]
        
        return sorted(active_nodes, key=lambda x: x["activation"], reverse=True)
    
    def _prune_inactive_nodes(self):
        """비활성 노드 제거"""
        if len(self.nodes) <= self.max_nodes:
            return
        
        # 활성화 수준으로 정렬
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: x[1].activation,
            reverse=True
        )
        
        # 상위 max_nodes개만 유지
        keep_ids = {nid for nid, _ in sorted_nodes[:self.max_nodes]}
        
        for nid in list(self.nodes.keys()):
            if nid not in keep_ids:
                del self.nodes[nid]
                del self.edges[nid]
    
    def to_dict(self) -> Dict[str, Any]:
        """직렬화"""
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "type": n.node_type,
                    "label": n.label,
                    "activation": n.activation,
                    "properties": n.properties,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": src, "target": tgt, "weight": w}
                for src, edges in self.edges.items()
                for tgt, w in edges
            ],
        }


@dataclass
class ContextConfig(AgentConfig):
    """컨텍스트 에이전트 설정"""
    db_path: str = ":memory:"  # SQLite DB 경로
    max_history_length: int = 1000
    temporal_decay: float = 0.95
    retrieval_top_k: int = 10


class ContextAgent(BaseAgent):
    """
    컨텍스트 에이전트
    
    주요 기능:
    1. 대화 히스토리 관리 (SQLite 기반)
    2. 상황 정보 그래프 구성 및 관리
    3. 장기 문맥 검색 및 회수
    4. 시간적 감쇠를 통한 관련성 관리
    """
    
    def __init__(
        self,
        message_bus: MessageBus,
        config: ContextConfig = None,
        agent_id: str = None
    ):
        config = config or ContextConfig(
            name="context",
            description="장기 문맥 및 상황 정보 관리 에이전트"
        )
        super().__init__(config, message_bus, agent_id)
        
        self.ctx_config = config
        
        # 상황 정보 그래프
        self.situation_graph = SituationGraph(
            decay_rate=config.temporal_decay
        )
        
        # 세션별 히스토리 캐시
        self.session_cache: Dict[str, List[HistoryEntry]] = {}
        
        # 데이터베이스 연결
        self.db_conn: Optional[sqlite3.Connection] = None
        
        # 핸들러 등록
        self.register_handler(
            MessageType.CONTEXT_REQUEST,
            self._handle_context_request
        )
        self.register_handler(
            MessageType.HISTORY_REQUEST,
            self._handle_history_request
        )
    
    async def on_start(self):
        """시작 시 초기화"""
        self.logger.info("컨텍스트 에이전트 초기화 중...")
        self._init_database()
    
    async def on_stop(self):
        """종료 시 정리"""
        if self.db_conn:
            self.db_conn.close()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        self.db_conn = sqlite3.connect(self.ctx_config.db_path)
        cursor = self.db_conn.cursor()
        
        # 히스토리 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                entry_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                embedding BLOB
            )
        """)
        
        # 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session 
            ON history(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON history(timestamp)
        """)
        
        self.db_conn.commit()
        self.logger.info("데이터베이스 초기화 완료")
    
    async def handle_message(self, message: Message) -> TaskResult:
        """기본 메시지 처리"""
        return TaskResult(
            success=False,
            error=f"처리되지 않은 메시지 유형: {message.type.value}"
        )
    
    async def _handle_context_request(self, message: Message) -> TaskResult:
        """
        컨텍스트 요청 처리
        
        Payload:
            session_id: 세션 ID
            query: 검색 쿼리 (선택)
            include_situation: 상황 그래프 포함 여부
        """
        session_id = message.payload.get("session_id", "default")
        query = message.payload.get("query", "")
        include_situation = message.payload.get("include_situation", True)
        
        self.logger.log_action("CONTEXT_RETRIEVAL", {"session": session_id})
        
        try:
            # 최근 히스토리 조회
            recent_history = self.get_recent_history(
                session_id,
                limit=self.ctx_config.retrieval_top_k
            )
            
            # 상황 정보
            situation = None
            if include_situation:
                situation = self.situation_graph.get_active_context()
            
            # 쿼리 기반 관련 히스토리 검색
            relevant_history = []
            if query:
                relevant_history = self.search_history(query, session_id)
            
            return TaskResult(
                success=True,
                result={
                    "session_id": session_id,
                    "recent_history": [h.to_dict() for h in recent_history],
                    "relevant_history": [h.to_dict() for h in relevant_history],
                    "situation_graph": situation,
                },
                metadata={
                    "history_count": len(recent_history),
                    "relevant_count": len(relevant_history),
                }
            )
            
        except Exception as e:
            self.logger.error(f"컨텍스트 조회 오류: {e}")
            return TaskResult(success=False, error=str(e))
    
    async def _handle_history_request(self, message: Message) -> TaskResult:
        """
        히스토리 요청 처리
        
        Payload:
            action: add, get, search
            session_id: 세션 ID
            entry: 추가할 항목 (add 시)
            query: 검색 쿼리 (search 시)
        """
        action = message.payload.get("action", "get")
        session_id = message.payload.get("session_id", "default")
        
        try:
            if action == "add":
                entry_data = message.payload.get("entry", {})
                entry = self.add_history(
                    session_id=session_id,
                    role=entry_data.get("role", "user"),
                    content=entry_data.get("content", ""),
                    metadata=entry_data.get("metadata", {}),
                )
                
                # 상황 그래프 업데이트
                self._update_situation_from_content(entry.content)
                
                return TaskResult(
                    success=True,
                    result={"entry_id": entry.entry_id}
                )
                
            elif action == "get":
                limit = message.payload.get("limit", 20)
                history = self.get_recent_history(session_id, limit)
                return TaskResult(
                    success=True,
                    result={"history": [h.to_dict() for h in history]}
                )
                
            elif action == "search":
                query = message.payload.get("query", "")
                results = self.search_history(query, session_id)
                return TaskResult(
                    success=True,
                    result={"results": [h.to_dict() for h in results]}
                )
                
            else:
                return TaskResult(
                    success=False,
                    error=f"알 수 없는 액션: {action}"
                )
                
        except Exception as e:
            self.logger.error(f"히스토리 처리 오류: {e}")
            return TaskResult(success=False, error=str(e))
    
    def _update_situation_from_content(self, content: str):
        """컨텐츠에서 상황 정보 추출 및 업데이트"""
        # 간단한 키워드 추출 (실제로는 NER, 토픽 모델링 등 사용)
        words = content.split()
        
        # 명사 추출 (간단한 휴리스틱)
        for word in words:
            if len(word) > 2 and word[0].isupper():
                self.situation_graph.add_node(
                    node_type="entity",
                    label=word,
                    activation=0.5,
                )
        
        # 토픽 키워드
        topic_keywords = ["AI", "머신러닝", "딥러닝", "학습", "모델"]
        for kw in topic_keywords:
            if kw.lower() in content.lower():
                self.situation_graph.add_node(
                    node_type="topic",
                    label=kw,
                    activation=0.8,
                )
    
    # 외부 API
    
    def add_history(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> HistoryEntry:
        """히스토리 항목 추가"""
        entry = HistoryEntry(
            entry_id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )
        
        # 데이터베이스에 저장
        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT INTO history (entry_id, session_id, role, content, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry.entry_id,
            entry.session_id,
            entry.role,
            entry.content,
            json.dumps(entry.metadata),
            entry.timestamp.isoformat(),
        ))
        self.db_conn.commit()
        
        # 캐시 업데이트
        if session_id not in self.session_cache:
            self.session_cache[session_id] = []
        self.session_cache[session_id].append(entry)
        
        # 캐시 크기 제한
        if len(self.session_cache[session_id]) > 100:
            self.session_cache[session_id] = self.session_cache[session_id][-100:]
        
        return entry
    
    def get_recent_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[HistoryEntry]:
        """최근 히스토리 조회"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT entry_id, session_id, role, content, metadata, timestamp
            FROM history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        entries = []
        for row in cursor.fetchall():
            entries.append(HistoryEntry(
                entry_id=row[0],
                session_id=row[1],
                role=row[2],
                content=row[3],
                metadata=json.loads(row[4]) if row[4] else {},
                timestamp=datetime.fromisoformat(row[5]),
            ))
        
        return list(reversed(entries))  # 시간순 정렬
    
    def search_history(
        self,
        query: str,
        session_id: str = None,
        limit: int = 10
    ) -> List[HistoryEntry]:
        """히스토리 검색"""
        cursor = self.db_conn.cursor()
        
        # 간단한 LIKE 검색 (실제로는 임베딩 기반 검색 사용)
        if session_id:
            cursor.execute("""
                SELECT entry_id, session_id, role, content, metadata, timestamp
                FROM history
                WHERE session_id = ? AND content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, f"%{query}%", limit))
        else:
            cursor.execute("""
                SELECT entry_id, session_id, role, content, metadata, timestamp
                FROM history
                WHERE content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (f"%{query}%", limit))
        
        entries = []
        for row in cursor.fetchall():
            entries.append(HistoryEntry(
                entry_id=row[0],
                session_id=row[1],
                role=row[2],
                content=row[3],
                metadata=json.loads(row[4]) if row[4] else {},
                timestamp=datetime.fromisoformat(row[5]),
            ))
        
        return entries
    
    def update_situation(
        self,
        node_type: str,
        label: str,
        properties: Dict[str, Any] = None,
        activation: float = 1.0
    ):
        """상황 정보 업데이트"""
        self.situation_graph.add_node(
            node_type=node_type,
            label=label,
            properties=properties,
            activation=activation,
        )
    
    def get_current_situation(self) -> Dict[str, Any]:
        """현재 상황 정보 반환"""
        return self.situation_graph.to_dict()

