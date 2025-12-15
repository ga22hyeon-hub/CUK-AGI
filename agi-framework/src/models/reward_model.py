"""
리워드 모델
선호도 기반 보상 모델 및 강화학습 지원
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random
import json
from pathlib import Path

import numpy as np

# PyTorch 선택적 import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PreferenceData:
    """선호도 데이터"""
    prompt: str
    chosen: str  # 선호 응답
    rejected: str  # 비선호 응답
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class PreferenceDataset:
    """
    선호도 데이터셋
    
    사용자 피드백 기반 선호도 데이터 관리
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path) if data_path else None
        self.data: List[PreferenceData] = []
        
        if self.data_path and self.data_path.exists():
            self.load()
    
    def add(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        metadata: Dict[str, Any] = None
    ):
        """선호도 데이터 추가"""
        self.data.append(PreferenceData(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata or {},
        ))
    
    def get_batch(self, batch_size: int = 32) -> List[PreferenceData]:
        """배치 샘플링"""
        if len(self.data) <= batch_size:
            return self.data
        return random.sample(self.data, batch_size)
    
    def save(self, path: str = None):
        """데이터셋 저장"""
        save_path = Path(path) if path else self.data_path
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([d.to_dict() for d in self.data], f, ensure_ascii=False, indent=2)
    
    def load(self, path: str = None):
        """데이터셋 로드"""
        load_path = Path(path) if path else self.data_path
        if load_path and load_path.exists():
            with open(load_path, encoding="utf-8") as f:
                data = json.load(f)
            self.data = [
                PreferenceData(
                    prompt=d["prompt"],
                    chosen=d["chosen"],
                    rejected=d["rejected"],
                    metadata=d.get("metadata", {}),
                    timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.now(),
                )
                for d in data
            ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def statistics(self) -> Dict[str, Any]:
        """데이터셋 통계"""
        return {
            "total_samples": len(self.data),
            "avg_prompt_length": np.mean([len(d.prompt) for d in self.data]) if self.data else 0,
            "avg_chosen_length": np.mean([len(d.chosen) for d in self.data]) if self.data else 0,
        }


class RewardModelNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    리워드 모델 네트워크
    
    응답의 품질 점수를 예측하는 신경망
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다: pip install torch")
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.reward_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """순전파 - 보상 점수 예측"""
        features = self.encoder(x)
        reward = self.reward_head(features)
        return reward.squeeze(-1)


class RewardModel:
    """
    리워드 모델 관리자
    
    선호도 학습 및 보상 점수 예측
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        learning_rate: float = 1e-5,
        device: str = "auto"
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        if TORCH_AVAILABLE:
            self.device = (
                "cuda" if device == "auto" and torch.cuda.is_available()
                else device if device != "auto" else "cpu"
            )
        else:
            self.device = "cpu"
        
        self.model: Optional[RewardModelNetwork] = None
        self.optimizer = None
        self.text_encoder = None
        
        # 학습 이력
        self.training_history: List[Dict[str, float]] = []
        
        self._initialized = False
    
    def initialize(self):
        """모델 초기화"""
        if self._initialized:
            return
        
        if TORCH_AVAILABLE:
            self.model = RewardModelNetwork(
                input_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
            ).to(self.device)
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        
        # 텍스트 인코더
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            if TORCH_AVAILABLE:
                self.text_encoder = self.text_encoder.to(self.device)
        except ImportError:
            print("Warning: sentence-transformers not available")
        
        self._initialized = True
    
    def encode_text(self, text: str) -> np.ndarray:
        """텍스트 인코딩"""
        if self.text_encoder:
            return self.text_encoder.encode(text, convert_to_numpy=True)
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """
        보상 점수 계산
        
        Args:
            prompt: 프롬프트
            response: 응답
            
        Returns:
            보상 점수 (-1 ~ 1)
        """
        if not self._initialized:
            self.initialize()
        
        # 텍스트 결합 및 인코딩
        combined = f"{prompt}\n\n{response}"
        embedding = self.encode_text(combined)
        
        if self.model is not None and TORCH_AVAILABLE:
            with torch.no_grad():
                x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
                reward = self.model(x).item()
                return float(np.tanh(reward))  # -1 ~ 1 범위로 정규화
        
        # 시뮬레이션 (휴리스틱 기반)
        return self._simulate_reward(prompt, response)
    
    def _simulate_reward(self, prompt: str, response: str) -> float:
        """시뮬레이션 보상 계산"""
        score = 0.5
        
        # 길이 기반
        if 100 < len(response) < 1000:
            score += 0.1
        elif len(response) < 50:
            score -= 0.2
        
        # 관련성 (단순 단어 겹침)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
        score += overlap * 0.2
        
        # 품질 지표 (간단한 휴리스틱)
        if "설명" in response or "다음과 같" in response:
            score += 0.1
        if "모르" in response or "불가능" in response:
            score -= 0.1
        
        return max(-1, min(1, score))
    
    def train_step(self, batch: List[PreferenceData]) -> Dict[str, float]:
        """
        
        Args:
            batch: 선호도 데이터 배치
            
        Returns:
            손실 및 메트릭
        """
        if not self._initialized:
            self.initialize()
        
        if self.model is None or not TORCH_AVAILABLE:
            # 시뮬레이션
            return {"loss": random.uniform(0.1, 0.5), "accuracy": random.uniform(0.6, 0.9)}
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # 데이터 준비
        chosen_embeddings = []
        rejected_embeddings = []
        
        for data in batch:
            chosen_text = f"{data.prompt}\n\n{data.chosen}"
            rejected_text = f"{data.prompt}\n\n{data.rejected}"
            
            chosen_embeddings.append(self.encode_text(chosen_text))
            rejected_embeddings.append(self.encode_text(rejected_text))
        
        chosen_tensor = torch.tensor(np.array(chosen_embeddings), dtype=torch.float32).to(self.device)
        rejected_tensor = torch.tensor(np.array(rejected_embeddings), dtype=torch.float32).to(self.device)
        
        # 보상 예측
        chosen_rewards = self.model(chosen_tensor)
        rejected_rewards = self.model(rejected_tensor)
        
        # Bradley-Terry 손실
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # 역전파
        loss.backward()
        self.optimizer.step()
        
        # 정확도 계산
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item(),
        }
        
        self.training_history.append(metrics)
        
        return metrics
    
    def train(
        self,
        dataset: PreferenceDataset,
        epochs: int = 10,
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """
        전체 학습
        
        Args:
            dataset: 선호도 데이터셋
            epochs: 에포크 수
            batch_size: 배치 크기
            
        Returns:
            에포크별 메트릭
        """
        epoch_metrics = []
        
        for epoch in range(epochs):
            batch = dataset.get_batch(batch_size)
            metrics = self.train_step(batch)
            metrics["epoch"] = epoch + 1
            epoch_metrics.append(metrics)
            
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
        
        return epoch_metrics
    
    def save(self, path: str):
        """모델 저장"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None and TORCH_AVAILABLE:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_history": self.training_history,
            }, save_path)
    
    def load(self, path: str):
        """모델 로드"""
        if not self._initialized:
            self.initialize()
        
        if self.model is not None and TORCH_AVAILABLE:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.training_history = checkpoint.get("training_history", [])


class PPOTrainer:
    """
    PPO 강화학습 트레이너
    
    리워드 모델을 사용한 정책 최적화
    """
    
    def __init__(
        self,
        reward_model: RewardModel,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        learning_rate: float = 3e-4,
        n_epochs: int = 4
    ):
        self.reward_model = reward_model
        self.gamma = gamma
        self.clip_range = clip_range
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        # 경험 버퍼
        self.experience_buffer: List[Dict[str, Any]] = []
    
    def add_experience(
        self,
        prompt: str,
        response: str,
        reward: float = None
    ):
        """경험 추가"""
        if reward is None:
            reward = self.reward_model.compute_reward(prompt, response)
        
        self.experience_buffer.append({
            "prompt": prompt,
            "response": response,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        })
    
    def compute_advantages(self) -> List[float]:
        """어드밴티지 계산 (간단한 버전)"""
        rewards = [exp["reward"] for exp in self.experience_buffer]
        
        # 기준선 (평균 보상)
        baseline = np.mean(rewards) if rewards else 0
        
        # 어드밴티지 = 보상 - 기준선
        advantages = [r - baseline for r in rewards]
        
        return advantages
    
    def update_policy(self) -> Dict[str, float]:
        """
        정책 업데이트 (시뮬레이션)
        
        실제 구현에서는 LLM의 파라미터를 업데이트해야 함
        """
        if not self.experience_buffer:
            return {"status": "no_data"}
        
        advantages = self.compute_advantages()
        
        # 메트릭 계산
        metrics = {
            "num_experiences": len(self.experience_buffer),
            "mean_reward": np.mean([exp["reward"] for exp in self.experience_buffer]),
            "mean_advantage": np.mean(advantages),
            "std_advantage": np.std(advantages) if len(advantages) > 1 else 0,
        }
        
        # 버퍼 클리어
        self.experience_buffer = []
        
        return metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 반환"""
        if not self.experience_buffer:
            return {"buffer_size": 0}
        
        rewards = [exp["reward"] for exp in self.experience_buffer]
        
        return {
            "buffer_size": len(self.experience_buffer),
            "mean_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
        }

