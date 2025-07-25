"""Smart Retry Engine for VIBE

This module provides intelligent retry strategies based on failure types,
with exponential backoff, partial result preservation, and error isolation.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
import re

from shared.core.enhanced_logger import get_logger, LogCategory

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur."""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    DEPENDENCY_ERROR = "dependency_error"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies for different failure types."""
    IMMEDIATE = "immediate"              # Retry immediately
    EXPONENTIAL_BACKOFF = "exponential"  # Exponential backoff
    LINEAR_BACKOFF = "linear"           # Linear backoff
    FIXED_DELAY = "fixed"               # Fixed delay between retries
    PROGRESSIVE = "progressive"         # Progressive strategy based on failure count
    NO_RETRY = "no_retry"               # Don't retry


@dataclass
class RetryPolicy:
    """Policy for retrying failed operations."""
    strategy: RetryStrategy
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0     # seconds
    backoff_factor: float = 2.0
    jitter: bool = True         # Add random jitter to delays
    
    # Conditions
    retry_on: Set[FailureType] = field(default_factory=lambda: {
        FailureType.TIMEOUT,
        FailureType.NETWORK_ERROR,
        FailureType.RESOURCE_LIMIT
    })
    
    # Partial success handling
    preserve_partial_results: bool = True
    aggregate_partial_results: bool = False


@dataclass
class FailureContext:
    """Context information about a failure."""
    failure_type: FailureType
    error_message: str
    error_details: Dict[str, Any]
    timestamp: datetime
    stack_trace: Optional[str] = None
    
    # Failure location
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    
    # Related data
    partial_result: Optional[Any] = None
    affected_resources: List[str] = field(default_factory=list)
    
    # Recovery hints
    suggested_fixes: List[str] = field(default_factory=list)
    can_retry: bool = True


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    failure_context: Optional[FailureContext] = None
    result: Optional[Any] = None
    delay_before_retry: float = 0.0


@dataclass
class RetrySession:
    """Complete retry session information."""
    session_id: str
    operation_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Attempts
    attempts: List[RetryAttempt] = field(default_factory=list)
    current_attempt: int = 0
    
    # Results
    final_result: Optional[Any] = None
    partial_results: List[Any] = field(default_factory=list)
    success: bool = False
    
    # Policy
    retry_policy: Optional[RetryPolicy] = None
    
    @property
    def total_duration(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


class SmartRetryEngine:
    """Intelligent retry engine with adaptive strategies."""
    
    def __init__(self):
        # Retry policies by failure type
        self.default_policies = self._initialize_default_policies()
        
        # Active retry sessions
        self.active_sessions: Dict[str, RetrySession] = {}
        
        # Historical data for learning
        self.retry_history: List[RetrySession] = []
        
        # Statistics
        self.stats = {
            'total_retries': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'partial_successes': 0,
            'average_attempts': 0.0,
            'failure_type_counts': {}
        }
        
        # Error patterns for classification
        self.error_patterns = self._initialize_error_patterns()
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Enhanced logger
        self.logger = get_logger(
            component="smart_retry",
            session_id=f"retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    async def retry_with_policy(self, operation: Callable, 
                               operation_name: str,
                               policy: Optional[RetryPolicy] = None,
                               context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
        """Execute operation with retry policy."""
        session_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        session = RetrySession(
            session_id=session_id,
            operation_name=operation_name,
            started_at=datetime.now(),
            retry_policy=policy
        )
        
        self.active_sessions[session_id] = session
        
        try:
            result = await self._execute_with_retries(operation, session, context)
            session.success = result[0]
            session.final_result = result[1]
            
            return result
            
        finally:
            session.completed_at = datetime.now()
            del self.active_sessions[session_id]
            self.retry_history.append(session)
            self._update_statistics(session)
    
    async def retry_with_fallback(self, operations: List[Callable],
                                 operation_name: str,
                                 context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
        """Try multiple operations in sequence until one succeeds."""
        for i, operation in enumerate(operations):
            op_name = f"{operation_name}_fallback_{i}"
            success, result = await self.retry_with_policy(operation, op_name, context=context)
            
            if success:
                return True, result
        
        return False, None
    
    def analyze_failure(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> FailureContext:
        """Analyze an error to determine failure type and recovery options."""
        error_str = str(error)
        error_type = type(error).__name__
        
        # Classify failure type
        failure_type = self._classify_failure(error, error_str)
        
        # Extract error details
        failure_context = FailureContext(
            failure_type=failure_type,
            error_message=error_str,
            error_details={'type': error_type, 'context': context or {}},
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc()
        )
        
        # Extract location information
        tb = traceback.extract_tb(error.__traceback__)
        if tb:
            last_frame = tb[-1]
            failure_context.file_path = last_frame.filename
            failure_context.line_number = last_frame.lineno
            failure_context.function_name = last_frame.name
        
        # Suggest fixes based on failure type
        failure_context.suggested_fixes = self._suggest_fixes(failure_type, error_str)
        
        # Determine if retryable
        failure_context.can_retry = failure_type in {
            FailureType.TIMEOUT,
            FailureType.NETWORK_ERROR,
            FailureType.RESOURCE_LIMIT
        }
        
        return failure_context
    
    def get_retry_policy(self, failure_type: FailureType, 
                        custom_overrides: Optional[Dict[str, Any]] = None) -> RetryPolicy:
        """Get appropriate retry policy for failure type."""
        policy = self.default_policies.get(failure_type, self.default_policies[FailureType.UNKNOWN])
        
        # Apply custom overrides
        if custom_overrides:
            for key, value in custom_overrides.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
        
        return policy
    
    def create_isolated_operation(self, operation: Callable, 
                                 isolation_level: str = "function") -> Callable:
        """Create an isolated version of an operation for safer retry."""
        async def isolated_operation(*args, **kwargs):
            # Create isolated execution context
            if isolation_level == "process":
                # Would use multiprocessing for complete isolation
                return await operation(*args, **kwargs)
            elif isolation_level == "thread":
                # Would use threading for thread-level isolation
                return await operation(*args, **kwargs)
            else:
                # Function-level isolation with error boundaries
                try:
                    return await operation(*args, **kwargs)
                except Exception as e:
                    # Log but contain the error
                    logger.error(f"Isolated operation failed: {e}")
                    raise
        
        return isolated_operation
    
    def merge_partial_results(self, partial_results: List[Any], 
                            merge_strategy: str = "last") -> Any:
        """Merge partial results from multiple attempts."""
        if not partial_results:
            return None
        
        if merge_strategy == "last":
            return partial_results[-1]
        elif merge_strategy == "first":
            return partial_results[0]
        elif merge_strategy == "combine":
            # Combine dictionaries
            if all(isinstance(r, dict) for r in partial_results):
                merged = {}
                for result in partial_results:
                    merged.update(result)
                return merged
            # Combine lists
            elif all(isinstance(r, list) for r in partial_results):
                merged = []
                for result in partial_results:
                    merged.extend(result)
                return merged
        
        return partial_results[-1]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry engine statistics."""
        return {
            **self.stats,
            'active_sessions': len(self.active_sessions),
            'success_rate': (
                self.stats['successful_retries'] / self.stats['total_retries']
                if self.stats['total_retries'] > 0 else 0.0
            ),
            'recent_failures': self._get_recent_failures(),
            'retry_patterns': self._analyze_retry_patterns()
        }
    
    # Private methods
    
    def _initialize_default_policies(self) -> Dict[FailureType, RetryPolicy]:
        """Initialize default retry policies for each failure type."""
        return {
            FailureType.SYNTAX_ERROR: RetryPolicy(
                strategy=RetryStrategy.NO_RETRY,
                max_attempts=1
            ),
            FailureType.IMPORT_ERROR: RetryPolicy(
                strategy=RetryStrategy.FIXED_DELAY,
                max_attempts=2,
                initial_delay=2.0
            ),
            FailureType.RUNTIME_ERROR: RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                initial_delay=1.0
            ),
            FailureType.TIMEOUT: RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=5,
                initial_delay=2.0,
                max_delay=30.0
            ),
            FailureType.NETWORK_ERROR: RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=5,
                initial_delay=1.0,
                backoff_factor=2.0,
                jitter=True
            ),
            FailureType.RESOURCE_LIMIT: RetryPolicy(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=3,
                initial_delay=5.0
            ),
            FailureType.PERMISSION_ERROR: RetryPolicy(
                strategy=RetryStrategy.NO_RETRY,
                max_attempts=1
            ),
            FailureType.UNKNOWN: RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                initial_delay=1.0
            )
        }
    
    def _initialize_error_patterns(self) -> Dict[FailureType, List[re.Pattern]]:
        """Initialize regex patterns for error classification."""
        return {
            FailureType.SYNTAX_ERROR: [
                re.compile(r'SyntaxError:', re.I),
                re.compile(r'IndentationError:', re.I),
                re.compile(r'unexpected token', re.I)
            ],
            FailureType.IMPORT_ERROR: [
                re.compile(r'ImportError:', re.I),
                re.compile(r'ModuleNotFoundError:', re.I),
                re.compile(r'cannot import name', re.I)
            ],
            FailureType.TIMEOUT: [
                re.compile(r'timeout', re.I),
                re.compile(r'timed out', re.I),
                re.compile(r'deadline exceeded', re.I)
            ],
            FailureType.NETWORK_ERROR: [
                re.compile(r'ConnectionError:', re.I),
                re.compile(r'URLError:', re.I),
                re.compile(r'connection refused', re.I),
                re.compile(r'network unreachable', re.I)
            ],
            FailureType.RESOURCE_LIMIT: [
                re.compile(r'MemoryError:', re.I),
                re.compile(r'out of memory', re.I),
                re.compile(r'resource exhausted', re.I),
                re.compile(r'quota exceeded', re.I)
            ],
            FailureType.PERMISSION_ERROR: [
                re.compile(r'PermissionError:', re.I),
                re.compile(r'permission denied', re.I),
                re.compile(r'access denied', re.I)
            ]
        }
    
    def _initialize_recovery_strategies(self) -> Dict[FailureType, List[str]]:
        """Initialize recovery strategies for each failure type."""
        return {
            FailureType.SYNTAX_ERROR: [
                "Check code syntax",
                "Validate generated code before execution",
                "Use AST parser to detect issues"
            ],
            FailureType.IMPORT_ERROR: [
                "Install missing dependencies",
                "Check import paths",
                "Verify module availability"
            ],
            FailureType.TIMEOUT: [
                "Increase timeout duration",
                "Optimize operation performance",
                "Break into smaller operations"
            ],
            FailureType.NETWORK_ERROR: [
                "Check network connectivity",
                "Retry with exponential backoff",
                "Use alternative endpoints"
            ],
            FailureType.RESOURCE_LIMIT: [
                "Free up resources",
                "Increase resource limits",
                "Optimize resource usage"
            ]
        }
    
    async def _execute_with_retries(self, operation: Callable,
                                   session: RetrySession,
                                   context: Optional[Dict[str, Any]]) -> Tuple[bool, Any]:
        """Execute operation with retries based on policy."""
        policy = session.retry_policy or self.default_policies[FailureType.UNKNOWN]
        
        for attempt_num in range(policy.max_attempts):
            attempt = RetryAttempt(
                attempt_number=attempt_num + 1,
                started_at=datetime.now()
            )
            
            try:
                # Calculate delay for this attempt
                if attempt_num > 0:
                    delay = self._calculate_delay(attempt_num, policy)
                    attempt.delay_before_retry = delay
                    
                    self.logger.log(
                        level=logging.INFO,
                        category=LogCategory.ERROR_RECOVERY,
                        message=f"Retrying {session.operation_name} after {delay:.2f}s",
                        metadata={'attempt': attempt_num + 1, 'delay': delay}
                    )
                    
                    await asyncio.sleep(delay)
                
                # Execute operation
                result = await operation(**(context or {}))
                
                attempt.success = True
                attempt.result = result
                attempt.completed_at = datetime.now()
                session.attempts.append(attempt)
                
                return True, result
                
            except Exception as e:
                # Analyze failure
                failure_context = self.analyze_failure(e, context)
                attempt.failure_context = failure_context
                attempt.completed_at = datetime.now()
                session.attempts.append(attempt)
                
                # Check if we should retry
                if failure_context.failure_type not in policy.retry_on:
                    self.logger.log(
                        level=logging.WARNING,
                        category=LogCategory.ERROR_RECOVERY,
                        message=f"Failure type {failure_context.failure_type.value} not retryable",
                        metadata={'error': str(e)}
                    )
                    return False, failure_context.partial_result
                
                # Preserve partial results if available
                if policy.preserve_partial_results and failure_context.partial_result:
                    session.partial_results.append(failure_context.partial_result)
                
                # Log retry decision
                self.logger.log(
                    level=logging.INFO,
                    category=LogCategory.ERROR_RECOVERY,
                    message=f"Attempt {attempt_num + 1} failed: {failure_context.error_message}",
                    metadata={
                        'failure_type': failure_context.failure_type.value,
                        'can_retry': failure_context.can_retry,
                        'remaining_attempts': policy.max_attempts - attempt_num - 1
                    }
                )
                
                # If last attempt, return aggregated partial results
                if attempt_num == policy.max_attempts - 1:
                    if policy.aggregate_partial_results and session.partial_results:
                        merged = self.merge_partial_results(session.partial_results)
                        return False, merged
                    return False, None
                
                self.stats['total_retries'] += 1
        
        return False, None
    
    def _calculate_delay(self, attempt_num: int, policy: RetryPolicy) -> float:
        """Calculate delay before retry based on strategy."""
        if policy.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0
        elif policy.strategy == RetryStrategy.FIXED_DELAY:
            delay = policy.initial_delay
        elif policy.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = policy.initial_delay * attempt_num
        elif policy.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = policy.initial_delay * (policy.backoff_factor ** (attempt_num - 1))
        else:
            delay = policy.initial_delay
        
        # Apply max delay cap
        delay = min(delay, policy.max_delay)
        
        # Add jitter if enabled
        if policy.jitter and delay > 0:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.0, delay)
    
    def _classify_failure(self, error: Exception, error_str: str) -> FailureType:
        """Classify failure type based on error."""
        # Check error patterns
        for failure_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.search(error_str):
                    return failure_type
        
        # Check exception type
        error_type = type(error).__name__
        
        if error_type in ['SyntaxError', 'IndentationError']:
            return FailureType.SYNTAX_ERROR
        elif error_type in ['ImportError', 'ModuleNotFoundError']:
            return FailureType.IMPORT_ERROR
        elif error_type in ['TimeoutError', 'asyncio.TimeoutError']:
            return FailureType.TIMEOUT
        elif error_type in ['ConnectionError', 'NetworkError']:
            return FailureType.NETWORK_ERROR
        elif error_type == 'PermissionError':
            return FailureType.PERMISSION_ERROR
        elif error_type == 'MemoryError':
            return FailureType.RESOURCE_LIMIT
        
        return FailureType.UNKNOWN
    
    def _suggest_fixes(self, failure_type: FailureType, error_message: str) -> List[str]:
        """Suggest fixes based on failure type and error message."""
        fixes = self.recovery_strategies.get(failure_type, [])
        
        # Add specific fixes based on error message
        if "No module named" in error_message:
            module_match = re.search(r"No module named '(\w+)'", error_message)
            if module_match:
                module_name = module_match.group(1)
                fixes.append(f"Run: pip install {module_name}")
        
        if "connection refused" in error_message.lower():
            fixes.append("Check if the service is running")
            fixes.append("Verify the connection URL and port")
        
        return fixes
    
    def _update_statistics(self, session: RetrySession):
        """Update statistics based on completed session."""
        if session.success:
            self.stats['successful_retries'] += 1
        else:
            self.stats['failed_retries'] += 1
        
        if session.partial_results:
            self.stats['partial_successes'] += 1
        
        # Update failure type counts
        for attempt in session.attempts:
            if attempt.failure_context:
                failure_type = attempt.failure_context.failure_type.value
                self.stats['failure_type_counts'][failure_type] = \
                    self.stats['failure_type_counts'].get(failure_type, 0) + 1
        
        # Update average attempts
        total_sessions = len(self.retry_history)
        if total_sessions > 0:
            total_attempts = sum(len(s.attempts) for s in self.retry_history)
            self.stats['average_attempts'] = total_attempts / total_sessions
    
    def _get_recent_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent failure information."""
        failures = []
        
        for session in reversed(self.retry_history[-limit:]):
            if not session.success:
                last_attempt = session.attempts[-1] if session.attempts else None
                if last_attempt and last_attempt.failure_context:
                    failures.append({
                        'operation': session.operation_name,
                        'timestamp': session.completed_at.isoformat() if session.completed_at else None,
                        'failure_type': last_attempt.failure_context.failure_type.value,
                        'error': last_attempt.failure_context.error_message,
                        'attempts': len(session.attempts)
                    })
        
        return failures
    
    def _analyze_retry_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in retry behavior."""
        patterns = {
            'most_common_failures': {},
            'average_attempts_by_type': {},
            'success_rate_by_type': {}
        }
        
        # Analyze by failure type
        type_stats = {}
        
        for session in self.retry_history:
            for attempt in session.attempts:
                if attempt.failure_context:
                    failure_type = attempt.failure_context.failure_type.value
                    
                    if failure_type not in type_stats:
                        type_stats[failure_type] = {
                            'total': 0,
                            'successful': 0,
                            'total_attempts': 0
                        }
                    
                    type_stats[failure_type]['total'] += 1
                    type_stats[failure_type]['total_attempts'] += len(session.attempts)
                    
                    if session.success:
                        type_stats[failure_type]['successful'] += 1
        
        # Calculate statistics
        for failure_type, stats in type_stats.items():
            if stats['total'] > 0:
                patterns['average_attempts_by_type'][failure_type] = \
                    stats['total_attempts'] / stats['total']
                patterns['success_rate_by_type'][failure_type] = \
                    stats['successful'] / stats['total']
        
        # Most common failures
        patterns['most_common_failures'] = dict(
            sorted(
                self.stats['failure_type_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        )
        
        return patterns


# Singleton instance
_smart_retry_engine: Optional[SmartRetryEngine] = None


def get_smart_retry_engine() -> SmartRetryEngine:
    """Get or create singleton retry engine."""
    global _smart_retry_engine
    if _smart_retry_engine is None:
        _smart_retry_engine = SmartRetryEngine()
    return _smart_retry_engine