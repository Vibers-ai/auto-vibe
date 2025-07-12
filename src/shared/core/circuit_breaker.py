"""Circuit Breaker Pattern - API 호출 안정성 보장 시스템"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import random
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """서킷 브레이커 상태"""
    CLOSED = "closed"        # 정상 동작
    OPEN = "open"           # 차단 상태
    HALF_OPEN = "half_open" # 반열림 상태 (테스트 중)


class FailureType(Enum):
    """실패 유형"""
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    AUTHENTICATION_ERROR = "auth_error"
    QUOTA_EXCEEDED = "quota_exceeded"


@dataclass
class CircuitBreakerConfig:
    """서킷 브레이커 설정"""
    failure_threshold: int = 5              # 실패 임계값
    recovery_timeout: float = 300.0         # 복구 타임아웃 (초)
    expected_exception: tuple = (Exception,) # 예상되는 예외 타입
    success_threshold: int = 3              # 성공 임계값 (HALF_OPEN -> CLOSED)
    timeout: float = 30.0                   # API 호출 타임아웃
    
    # 진보된 설정
    sliding_window_size: int = 10           # 슬라이딩 윈도우 크기
    minimum_requests: int = 5               # 최소 요청 수
    error_rate_threshold: float = 0.5       # 에러율 임계값 (50%)
    
    # 백오프 설정
    exponential_backoff: bool = True        # 지수 백오프 사용
    max_backoff_time: float = 3600.0        # 최대 백오프 시간 (1시간)
    jitter: bool = True                     # 지터 적용


@dataclass
class CallResult:
    """API 호출 결과"""
    success: bool
    timestamp: datetime
    duration: float
    failure_type: Optional[FailureType] = None
    error_message: Optional[str] = None
    response_data: Any = None


@dataclass
class CircuitBreakerStats:
    """서킷 브레이커 통계"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    
    current_failure_count: int = 0
    current_success_count: int = 0
    
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    state_changes: int = 0
    time_in_open_state: float = 0.0


class CircuitBreakerOpenError(Exception):
    """서킷 브레이커가 열린 상태에서 발생하는 예외"""
    pass


class CircuitBreaker:
    """API 호출을 보호하는 서킷 브레이커"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # 상태 관리
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        self.state_changed_time = datetime.now()
        
        # 통계
        self.stats = CircuitBreakerStats()
        
        # 호출 기록 (슬라이딩 윈도우용)
        self.call_history: List[CallResult] = []
        
        # 백오프 계산용
        self.consecutive_failures = 0
        
        # 비동기 락
        self.lock = asyncio.Lock()
    
    def _classify_exception(self, exception: Exception) -> FailureType:
        """예외를 실패 유형으로 분류"""
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        if 'timeout' in exception_str or 'timeout' in exception_type:
            return FailureType.TIMEOUT
        elif 'rate limit' in exception_str or 'too many requests' in exception_str:
            return FailureType.RATE_LIMIT
        elif 'quota' in exception_str or 'exceeded' in exception_str:
            return FailureType.QUOTA_EXCEEDED
        elif 'authentication' in exception_str or 'unauthorized' in exception_str:
            return FailureType.AUTHENTICATION_ERROR
        elif 'network' in exception_str or 'connection' in exception_str:
            return FailureType.NETWORK_ERROR
        elif 'server error' in exception_str or '500' in exception_str:
            return FailureType.SERVER_ERROR
        else:
            return FailureType.API_ERROR
    
    def _calculate_backoff_time(self) -> float:
        """백오프 시간 계산"""
        if not self.config.exponential_backoff:
            return self.config.recovery_timeout
        
        # 지수 백오프 계산
        backoff_time = min(
            self.config.recovery_timeout * (2 ** self.consecutive_failures),
            self.config.max_backoff_time
        )
        
        # 지터 적용
        if self.config.jitter:
            jitter_factor = random.uniform(0.8, 1.2)
            backoff_time *= jitter_factor
        
        return backoff_time
    
    def _should_open_circuit(self) -> bool:
        """서킷을 열어야 하는지 판단"""
        # 기본 실패 카운트 체크
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # 슬라이딩 윈도우 기반 에러율 체크
        if len(self.call_history) >= self.config.minimum_requests:
            recent_calls = self.call_history[-self.config.sliding_window_size:]
            failed_calls = sum(1 for call in recent_calls if not call.success)
            error_rate = failed_calls / len(recent_calls)
            
            if error_rate >= self.config.error_rate_threshold:
                return True
        
        return False
    
    def _should_close_circuit(self) -> bool:
        """서킷을 닫아야 하는지 판단"""
        return self.success_count >= self.config.success_threshold
    
    def _can_attempt_call(self) -> bool:
        """호출을 시도할 수 있는지 확인"""
        current_time = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.HALF_OPEN:
            return True
        elif self.state == CircuitState.OPEN:
            if self.next_attempt_time and current_time >= self.next_attempt_time:
                return True
            return False
        
        return False
    
    def _change_state(self, new_state: CircuitState):
        """상태 변경"""
        old_state = self.state
        self.state = new_state
        self.state_changed_time = datetime.now()
        self.stats.state_changes += 1
        
        # 상태별 초기화
        if new_state == CircuitState.HALF_OPEN:
            self.success_count = 0
        elif new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.consecutive_failures = 0
            self.next_attempt_time = None
        elif new_state == CircuitState.OPEN:
            backoff_time = self._calculate_backoff_time()
            self.next_attempt_time = datetime.now() + timedelta(seconds=backoff_time)
            self.consecutive_failures += 1
        
        logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
    
    def _record_success(self, duration: float, response_data: Any = None):
        """성공 기록"""
        call_result = CallResult(
            success=True,
            timestamp=datetime.now(),
            duration=duration,
            response_data=response_data
        )
        
        self.call_history.append(call_result)
        self._cleanup_old_records()
        
        # 통계 업데이트
        self.stats.total_calls += 1
        self.stats.successful_calls += 1
        self.stats.last_success_time = call_result.timestamp
        
        # 평균 응답 시간 업데이트
        total_time = self.stats.avg_response_time * (self.stats.successful_calls - 1) + duration
        self.stats.avg_response_time = total_time / self.stats.successful_calls
        
        # 상태별 처리
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self._should_close_circuit():
                self._change_state(CircuitState.CLOSED)
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # 성공 시 실패 카운트 리셋
        
        self._update_error_rate()
    
    def _record_failure(self, exception: Exception, duration: float = 0.0):
        """실패 기록"""
        failure_type = self._classify_exception(exception)
        
        call_result = CallResult(
            success=False,
            timestamp=datetime.now(),
            duration=duration,
            failure_type=failure_type,
            error_message=str(exception)
        )
        
        self.call_history.append(call_result)
        self._cleanup_old_records()
        
        # 통계 업데이트
        self.stats.total_calls += 1
        self.stats.failed_calls += 1
        self.stats.current_failure_count += 1
        self.stats.last_failure_time = call_result.timestamp
        
        # 상태별 처리
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self._should_open_circuit():
                self._change_state(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            self._change_state(CircuitState.OPEN)
        
        self._update_error_rate()
    
    def _record_rejection(self):
        """거부 기록"""
        self.stats.rejected_calls += 1
        logger.debug(f"Circuit breaker '{self.name}' rejected call (state: {self.state.value})")
    
    def _cleanup_old_records(self):
        """오래된 기록 정리"""
        max_records = self.config.sliding_window_size * 2
        if len(self.call_history) > max_records:
            self.call_history = self.call_history[-max_records:]
    
    def _update_error_rate(self):
        """에러율 업데이트"""
        if self.stats.total_calls > 0:
            self.stats.error_rate = self.stats.failed_calls / self.stats.total_calls
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """보호된 함수 호출"""
        async with self.lock:
            # 호출 가능성 확인
            if not self._can_attempt_call():
                self._record_rejection()
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is {self.state.value}. "
                    f"Next attempt time: {self.next_attempt_time}"
                )
            
            # CLOSED에서 HALF_OPEN으로 전환
            if self.state == CircuitState.OPEN and self._can_attempt_call():
                self._change_state(CircuitState.HALF_OPEN)
        
        # 실제 함수 호출 (락 외부에서)
        start_time = time.time()
        try:
            # 타임아웃 적용
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            async with self.lock:
                self._record_success(duration, result)
            
            return result
            
        except asyncio.TimeoutError as e:
            duration = time.time() - start_time
            async with self.lock:
                self._record_failure(e, duration)
            raise
            
        except Exception as e:
            duration = time.time() - start_time
            if isinstance(e, self.config.expected_exception):
                async with self.lock:
                    self._record_failure(e, duration)
            raise
    
    def get_state(self) -> CircuitState:
        """현재 상태 반환"""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        current_time = datetime.now()
        
        # 열린 상태 시간 계산
        time_in_open = 0.0
        if self.state == CircuitState.OPEN and self.state_changed_time:
            time_in_open = (current_time - self.state_changed_time).total_seconds()
        
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'next_attempt_time': self.next_attempt_time.isoformat() if self.next_attempt_time else None,
            'stats': {
                'total_calls': self.stats.total_calls,
                'successful_calls': self.stats.successful_calls,
                'failed_calls': self.stats.failed_calls,
                'rejected_calls': self.stats.rejected_calls,
                'error_rate': round(self.stats.error_rate * 100, 2),
                'avg_response_time': round(self.stats.avg_response_time, 3),
                'state_changes': self.stats.state_changes,
                'time_in_open_state': round(time_in_open, 2)
            },
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }
    
    def force_open(self):
        """강제로 서킷 열기"""
        with asyncio.Lock():
            self._change_state(CircuitState.OPEN)
        logger.warning(f"Circuit breaker '{self.name}' forced open")
    
    def force_close(self):
        """강제로 서킷 닫기"""
        with asyncio.Lock():
            self._change_state(CircuitState.CLOSED)
        logger.info(f"Circuit breaker '{self.name}' forced closed")
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = CircuitBreakerStats()
        self.call_history.clear()
        logger.info(f"Circuit breaker '{self.name}' stats reset")


class CircuitBreakerManager:
    """여러 서킷 브레이커를 관리하는 매니저"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """서킷 브레이커 인스턴스 가져오기 (없으면 생성)"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                config=config or self.default_config
            )
        return self.circuit_breakers[name]
    
    def remove_circuit_breaker(self, name: str):
        """서킷 브레이커 제거"""
        if name in self.circuit_breakers:
            del self.circuit_breakers[name]
            logger.info(f"Circuit breaker '{name}' removed")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """모든 서킷 브레이커 통계"""
        return {
            name: breaker.get_stats()
            for name, breaker in self.circuit_breakers.items()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """전체 헬스 요약"""
        total_breakers = len(self.circuit_breakers)
        open_breakers = sum(
            1 for breaker in self.circuit_breakers.values()
            if breaker.get_state() == CircuitState.OPEN
        )
        half_open_breakers = sum(
            1 for breaker in self.circuit_breakers.values()
            if breaker.get_state() == CircuitState.HALF_OPEN
        )
        
        overall_health = "healthy"
        if open_breakers > 0:
            if open_breakers == total_breakers:
                overall_health = "critical"
            elif open_breakers >= total_breakers * 0.5:
                overall_health = "degraded"
            else:
                overall_health = "warning"
        
        return {
            'overall_health': overall_health,
            'total_circuit_breakers': total_breakers,
            'open_circuit_breakers': open_breakers,
            'half_open_circuit_breakers': half_open_breakers,
            'healthy_circuit_breakers': total_breakers - open_breakers - half_open_breakers
        }


# 전역 서킷 브레이커 매니저
global_circuit_breaker_manager = CircuitBreakerManager()


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """서킷 브레이커 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            breaker = global_circuit_breaker_manager.get_circuit_breaker(name, config)
            return await breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            breaker = global_circuit_breaker_manager.get_circuit_breaker(name, config)
            return asyncio.run(breaker.call(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# 사전 정의된 설정들
GEMINI_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=180.0,  # 3분
    success_threshold=2,
    timeout=60.0,
    error_rate_threshold=0.4,
    exponential_backoff=True,
    max_backoff_time=1800.0  # 30분
)

CLAUDE_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=120.0,  # 2분
    success_threshold=2,
    timeout=45.0,
    error_rate_threshold=0.3,
    exponential_backoff=True,
    max_backoff_time=900.0   # 15분
)

FILE_OPERATION_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0,   # 30초
    success_threshold=3,
    timeout=10.0,
    error_rate_threshold=0.6,
    exponential_backoff=False
)