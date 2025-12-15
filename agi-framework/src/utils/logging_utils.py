"""
로깅 유틸리티
에이전트 및 프레임워크 로깅을 위한 헬퍼 함수
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """컬러 로그 포맷터"""
    
    COLORS = {
        'DEBUG': '\033[94m',     # 파랑
        'INFO': '\033[92m',      # 초록
        'WARNING': '\033[93m',   # 노랑
        'ERROR': '\033[91m',     # 빨강
        'CRITICAL': '\033[95m',  # 보라
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(
    name: str = "agi_framework",
    level: str = "INFO",
    log_file: Optional[str] = None,
    colored: bool = True
) -> logging.Logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로 (선택)
        colored: 컬러 출력 여부
        
    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 기존 핸들러 제거
    logger.handlers = []
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if colored:
        formatter = ColoredFormatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    기존 로거 가져오기
    
    Args:
        name: 로거 이름
        
    Returns:
        로거 인스턴스
    """
    return logging.getLogger(name)


class AgentLogger:
    """에이전트 전용 로거"""
    
    def __init__(self, agent_name: str, agent_id: str):
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.logger = get_logger(f"agi_framework.{agent_name}")
    
    def _format_message(self, message: str) -> str:
        return f"[{self.agent_id}] {message}"
    
    def debug(self, message: str):
        self.logger.debug(self._format_message(message))
    
    def info(self, message: str):
        self.logger.info(self._format_message(message))
    
    def warning(self, message: str):
        self.logger.warning(self._format_message(message))
    
    def error(self, message: str):
        self.logger.error(self._format_message(message))
    
    def critical(self, message: str):
        self.logger.critical(self._format_message(message))
    
    def log_action(self, action: str, details: dict = None):
        """에이전트 액션 로깅"""
        details_str = f" | {details}" if details else ""
        self.info(f"ACTION: {action}{details_str}")
    
    def log_message_received(self, message_type: str, sender: str):
        """메시지 수신 로깅"""
        self.debug(f"RECEIVED: {message_type} from {sender}")
    
    def log_message_sent(self, message_type: str, receiver: str):
        """메시지 발신 로깅"""
        self.debug(f"SENT: {message_type} to {receiver}")

