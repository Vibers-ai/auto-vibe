"""Token Overflow Guard - 토큰 오버플로우 방지 및 모니터링 시스템"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import threading

logger = logging.getLogger(__name__)


class TokenUsageLevel(Enum):
    """토큰 사용량 레벨"""
    SAFE = "safe"           # < 70%
    WARNING = "warning"     # 70-85%
    CRITICAL = "critical"   # 85-95%
    EMERGENCY = "emergency" # 95-100%
    OVERFLOW = "overflow"   # > 100%


class CompressionUrgency(Enum):
    """압축 긴급도"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"


@dataclass
class TokenUsageStats:
    """토큰 사용량 통계"""
    current_tokens: int = 0
    max_tokens: int = 128000
    utilization_rate: float = 0.0
    usage_level: TokenUsageLevel = TokenUsageLevel.SAFE
    last_updated: datetime = field(default_factory=datetime.now)
    
    # 압축 통계
    total_compressions: int = 0
    successful_compressions: int = 0
    failed_compressions: int = 0
    emergency_compressions: int = 0
    
    # 성능 통계
    avg_compression_time: float = 0.0
    memory_usage_mb: float = 0.0
    compression_ratio: float = 0.0


@dataclass
class CompressionAttempt:
    """압축 시도 기록"""
    timestamp: datetime
    pre_compression_tokens: int
    post_compression_tokens: int
    duration: float
    strategy: str
    success: bool
    error: Optional[str] = None


class TokenOverflowGuard:
    """토큰 오버플로우 방지 및 모니터링 시스템"""
    
    def __init__(self, max_tokens: int = 128000, reserved_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.available_tokens = max_tokens - reserved_tokens
        
        # 임계값 설정
        self.warning_threshold = self.available_tokens * 0.7   # 70%
        self.critical_threshold = self.available_tokens * 0.85  # 85%
        self.emergency_threshold = self.available_tokens * 0.95 # 95%
        
        # 통계 및 상태
        self.stats = TokenUsageStats(max_tokens=max_tokens)
        self.compression_history: List[CompressionAttempt] = []
        self.compression_in_progress = False
        self.compression_lock = asyncio.Lock()
        
        # 무한 루프 방지
        self.max_compression_attempts = 3
        self.compression_cooldown = 10  # 10초 쿨다운
        self.last_compression_time = 0
        
        # 모니터링
        self.monitoring_enabled = True
        self.monitoring_interval = 5  # 5초마다
        self.alert_callbacks: List[callable] = []
        
        # 백그라운드 모니터링 태스크
        self.monitoring_task = None
        self.shutdown_event = asyncio.Event()
    
    def estimate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 추정 (개선된 알고리즘)"""
        if not text:
            return 0
        
        # 다양한 언어와 특수 문자를 고려한 토큰 추정
        # 영어: 1 토큰 ≈ 4 문자
        # 한국어: 1 토큰 ≈ 1.5-2 문자  
        # 코드: 1 토큰 ≈ 3-4 문자
        
        char_count = len(text)
        
        # 한국어 문자 비율 계산
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        korean_ratio = korean_chars / char_count if char_count > 0 else 0
        
        # 코드 패턴 감지 (중괄호, 세미콜론, 함수명 등)
        code_indicators = text.count('{') + text.count('}') + text.count(';') + text.count('def ') + text.count('function ')
        code_ratio = min(code_indicators / (char_count / 100), 1.0) if char_count > 0 else 0
        
        # 토큰 비율 계산
        if korean_ratio > 0.3:  # 한국어 텍스트
            token_ratio = 1.7
        elif code_ratio > 0.1:  # 코드
            token_ratio = 3.5
        else:  # 영어 텍스트
            token_ratio = 4.0
        
        estimated_tokens = int(char_count / token_ratio)
        return max(estimated_tokens, 1)  # 최소 1 토큰
    
    def update_token_usage(self, current_tokens: int) -> TokenUsageLevel:
        """현재 토큰 사용량 업데이트 및 레벨 결정"""
        self.stats.current_tokens = current_tokens
        self.stats.utilization_rate = current_tokens / self.available_tokens
        self.stats.last_updated = datetime.now()
        self.stats.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 사용량 레벨 결정
        if current_tokens >= self.available_tokens:
            self.stats.usage_level = TokenUsageLevel.OVERFLOW
        elif current_tokens >= self.emergency_threshold:
            self.stats.usage_level = TokenUsageLevel.EMERGENCY
        elif current_tokens >= self.critical_threshold:
            self.stats.usage_level = TokenUsageLevel.CRITICAL
        elif current_tokens >= self.warning_threshold:
            self.stats.usage_level = TokenUsageLevel.WARNING
        else:
            self.stats.usage_level = TokenUsageLevel.SAFE
        
        return self.stats.usage_level
    
    def get_compression_urgency(self, projected_tokens: int = 0) -> CompressionUrgency:
        """압축 긴급도 평가"""
        total_tokens = self.stats.current_tokens + projected_tokens
        utilization = total_tokens / self.available_tokens
        
        if utilization >= 1.0:
            return CompressionUrgency.EMERGENCY
        elif utilization >= 0.95:
            return CompressionUrgency.HIGH
        elif utilization >= 0.85:
            return CompressionUrgency.MEDIUM
        elif utilization >= 0.7:
            return CompressionUrgency.LOW
        else:
            return CompressionUrgency.NONE
    
    async def check_and_compress_if_needed(self, context_manager, additional_tokens: int = 0) -> bool:
        """필요시 압축 수행 (무한 루프 방지 포함)"""
        
        # 현재 토큰 사용량 확인
        current_tokens = sum(cw.token_count for cw in context_manager.context_windows)
        usage_level = self.update_token_usage(current_tokens + additional_tokens)
        
        # 압축이 필요한지 확인
        urgency = self.get_compression_urgency(additional_tokens)
        
        if urgency == CompressionUrgency.NONE:
            return True  # 압축 불필요
        
        # 무한 루프 방지 체크
        if self._is_compression_blocked():
            logger.warning("Compression blocked due to cooldown or max attempts")
            return False
        
        # 압축 수행
        return await self._perform_safe_compression(context_manager, urgency)
    
    def _is_compression_blocked(self) -> bool:
        """압축이 차단되었는지 확인"""
        current_time = time.time()
        
        # 쿨다운 체크
        if current_time - self.last_compression_time < self.compression_cooldown:
            return True
        
        # 최근 실패한 압축 시도 체크
        recent_attempts = [
            attempt for attempt in self.compression_history[-self.max_compression_attempts:]
            if (datetime.now() - attempt.timestamp).total_seconds() < 60  # 1분 이내
        ]
        
        failed_attempts = [attempt for attempt in recent_attempts if not attempt.success]
        
        return len(failed_attempts) >= self.max_compression_attempts
    
    async def _perform_safe_compression(self, context_manager, urgency: CompressionUrgency) -> bool:
        """안전한 압축 수행"""
        
        if self.compression_in_progress:
            logger.warning("Compression already in progress, skipping")
            return False
        
        async with self.compression_lock:
            self.compression_in_progress = True
            
            try:
                start_time = time.time()
                pre_tokens = sum(cw.token_count for cw in context_manager.context_windows)
                
                logger.info(f"Starting compression with urgency: {urgency.value}, tokens: {pre_tokens}")
                
                # 긴급도에 따른 압축 전략 선택
                success = await self._execute_compression_strategy(context_manager, urgency)
                
                # 압축 후 토큰 수 확인
                post_tokens = sum(cw.token_count for cw in context_manager.context_windows)
                duration = time.time() - start_time
                
                # 압축 기록
                attempt = CompressionAttempt(
                    timestamp=datetime.now(),
                    pre_compression_tokens=pre_tokens,
                    post_compression_tokens=post_tokens,
                    duration=duration,
                    strategy=urgency.value,
                    success=success
                )
                
                self.compression_history.append(attempt)
                self._update_compression_stats(attempt)
                
                # 긴급 압축인 경우 통계 업데이트
                if urgency == CompressionUrgency.EMERGENCY:
                    self.stats.emergency_compressions += 1
                
                self.last_compression_time = time.time()
                
                if success:
                    compression_ratio = (pre_tokens - post_tokens) / pre_tokens if pre_tokens > 0 else 0
                    logger.info(f"Compression successful: {pre_tokens} -> {post_tokens} tokens "
                              f"(ratio: {compression_ratio:.2f}) in {duration:.2f}s")
                else:
                    logger.error(f"Compression failed after {duration:.2f}s")
                
                return success
                
            except Exception as e:
                logger.error(f"Compression error: {e}")
                
                # 실패 기록
                attempt = CompressionAttempt(
                    timestamp=datetime.now(),
                    pre_compression_tokens=pre_tokens if 'pre_tokens' in locals() else 0,
                    post_compression_tokens=0,
                    duration=time.time() - start_time if 'start_time' in locals() else 0,
                    strategy=urgency.value,
                    success=False,
                    error=str(e)
                )
                
                self.compression_history.append(attempt)
                self._update_compression_stats(attempt)
                
                return False
                
            finally:
                self.compression_in_progress = False
    
    async def _execute_compression_strategy(self, context_manager, urgency: CompressionUrgency) -> bool:
        """긴급도에 따른 압축 전략 실행"""
        
        try:
            if urgency == CompressionUrgency.EMERGENCY:
                # 긴급: 가장 적극적인 압축
                await context_manager._sliding_window_compression()
                
                # 여전히 부족하면 강제 정리
                if self._still_over_threshold(context_manager):
                    await self._emergency_cleanup(context_manager)
                    
            elif urgency == CompressionUrgency.HIGH:
                # 높음: 하이브리드 압축
                await context_manager._hybrid_compression()
                
            elif urgency == CompressionUrgency.MEDIUM:
                # 중간: 의미적 필터링
                await context_manager._semantic_filtering_compression()
                
            elif urgency == CompressionUrgency.LOW:
                # 낮음: 오래된 컨텍스트 정리
                context_manager._remove_stale_context()
            
            return True
            
        except Exception as e:
            logger.error(f"Compression strategy execution failed: {e}")
            return False
    
    def _still_over_threshold(self, context_manager) -> bool:
        """여전히 임계값을 초과하는지 확인"""
        current_tokens = sum(cw.token_count for cw in context_manager.context_windows)
        return current_tokens > self.emergency_threshold
    
    async def _emergency_cleanup(self, context_manager):
        """긴급 상황에서의 강제 정리"""
        logger.warning("Performing emergency context cleanup")
        
        # 컨텍스트 윈도우를 중요도 순으로 정렬
        sorted_windows = sorted(
            context_manager.context_windows,
            key=lambda cw: (cw.importance_score, cw.timestamp),
            reverse=True
        )
        
        # 가장 중요한 것들만 유지 (50% 까지)
        keep_count = len(sorted_windows) // 2
        context_manager.context_windows = sorted_windows[:keep_count]
        
        logger.warning(f"Emergency cleanup: kept {keep_count} out of {len(sorted_windows)} contexts")
    
    def _update_compression_stats(self, attempt: CompressionAttempt):
        """압축 통계 업데이트"""
        self.stats.total_compressions += 1
        
        if attempt.success:
            self.stats.successful_compressions += 1
            
            # 평균 압축 시간 업데이트
            total_time = self.stats.avg_compression_time * (self.stats.successful_compressions - 1) + attempt.duration
            self.stats.avg_compression_time = total_time / self.stats.successful_compressions
            
            # 압축 비율 업데이트
            if attempt.pre_compression_tokens > 0:
                current_ratio = (attempt.pre_compression_tokens - attempt.post_compression_tokens) / attempt.pre_compression_tokens
                self.stats.compression_ratio = (self.stats.compression_ratio + current_ratio) / 2
                
        else:
            self.stats.failed_compressions += 1
    
    async def start_monitoring(self):
        """백그라운드 모니터링 시작"""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Token monitoring started")
    
    async def stop_monitoring(self):
        """백그라운드 모니터링 중지"""
        self.shutdown_event.set()
        if self.monitoring_task and not self.monitoring_task.done():
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.monitoring_task.cancel()
        logger.info("Token monitoring stopped")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while not self.shutdown_event.is_set():
            try:
                # 메모리 사용량 업데이트
                self.stats.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
                
                # 알림 체크
                await self._check_and_send_alerts()
                
                # 오래된 압축 기록 정리
                self._cleanup_old_compression_history()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_and_send_alerts(self):
        """알림 확인 및 전송"""
        if self.stats.usage_level in [TokenUsageLevel.CRITICAL, TokenUsageLevel.EMERGENCY, TokenUsageLevel.OVERFLOW]:
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self.stats)
                    else:
                        callback(self.stats)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def _cleanup_old_compression_history(self):
        """오래된 압축 기록 정리"""
        cutoff_time = datetime.now() - timedelta(hours=1)  # 1시간 이전 기록 제거
        self.compression_history = [
            attempt for attempt in self.compression_history
            if attempt.timestamp > cutoff_time
        ]
    
    def add_alert_callback(self, callback: callable):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """상세 통계 정보 반환"""
        recent_attempts = [
            attempt for attempt in self.compression_history
            if (datetime.now() - attempt.timestamp).total_seconds() < 300  # 5분 이내
        ]
        
        return {
            'current_stats': {
                'current_tokens': self.stats.current_tokens,
                'max_tokens': self.stats.max_tokens,
                'utilization_rate': round(self.stats.utilization_rate * 100, 2),
                'usage_level': self.stats.usage_level.value,
                'memory_usage_mb': round(self.stats.memory_usage_mb, 2)
            },
            'compression_stats': {
                'total_compressions': self.stats.total_compressions,
                'successful_compressions': self.stats.successful_compressions,
                'failed_compressions': self.stats.failed_compressions,
                'emergency_compressions': self.stats.emergency_compressions,
                'success_rate': round(
                    self.stats.successful_compressions / max(self.stats.total_compressions, 1) * 100, 2
                ),
                'avg_compression_time': round(self.stats.avg_compression_time, 2),
                'avg_compression_ratio': round(self.stats.compression_ratio * 100, 2)
            },
            'recent_activity': {
                'recent_compressions': len(recent_attempts),
                'compression_in_progress': self.compression_in_progress,
                'last_compression': self.compression_history[-1].timestamp.isoformat() if self.compression_history else None
            },
            'thresholds': {
                'warning_threshold': self.warning_threshold,
                'critical_threshold': self.critical_threshold,
                'emergency_threshold': self.emergency_threshold
            }
        }


# 전역 토큰 가드 인스턴스
global_token_guard = TokenOverflowGuard()