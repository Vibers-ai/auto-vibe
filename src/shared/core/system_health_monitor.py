"""System Health Monitor - 시스템 헬스체크 및 자동 복구 시스템"""

import asyncio
import logging
import time
import psutil
import os
import signal
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import threading

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """헬스 상태"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class RecoveryAction(Enum):
    """복구 액션"""
    NONE = "none"
    RESTART_COMPONENT = "restart_component"
    CLEAR_CACHE = "clear_cache"
    FORCE_CLEANUP = "force_cleanup"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    SCALE_RESOURCES = "scale_resources"


@dataclass
class HealthCheckResult:
    """헬스체크 결과"""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_free_gb: float = 0.0
    open_files_count: int = 0
    network_connections: int = 0
    process_count: int = 0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class BaseHealthCheck:
    """기본 헬스체크 클래스"""
    
    def __init__(self, name: str, critical: bool = True):
        self.name = name
        self.critical = critical
        self.last_check_time = None
        self.consecutive_failures = 0
        self.max_failures = 3
    
    async def check_health(self) -> HealthCheckResult:
        """헬스체크 수행 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def reset_failure_count(self):
        """실패 카운트 리셋"""
        self.consecutive_failures = 0


class MemoryHealthCheck(BaseHealthCheck):
    """메모리 사용량 헬스체크"""
    
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0):
        super().__init__("memory", critical=True)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    async def check_health(self) -> HealthCheckResult:
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            if memory_percent >= self.critical_threshold:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory_percent:.1f}%"
                recovery_actions = [RecoveryAction.FORCE_CLEANUP, RecoveryAction.CLEAR_CACHE]
                suggestions = [
                    "Force garbage collection",
                    "Clear context cache",
                    "Terminate non-essential processes"
                ]
            elif memory_percent >= self.warning_threshold:
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_percent:.1f}%"
                recovery_actions = [RecoveryAction.CLEAR_CACHE]
                suggestions = [
                    "Clear context cache",
                    "Reduce parallel tasks"
                ]
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
                recovery_actions = []
                suggestions = []
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'memory_percent': memory_percent,
                    'memory_available_gb': round(memory_available_gb, 2),
                    'memory_total_gb': round(memory.total / (1024**3), 2)
                },
                suggestions=suggestions,
                recovery_actions=recovery_actions
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.FAILED,
                message=f"Memory check failed: {e}",
                timestamp=datetime.now()
            )


class TokenUsageHealthCheck(BaseHealthCheck):
    """토큰 사용량 헬스체크"""
    
    def __init__(self, token_guard):
        super().__init__("token_usage", critical=True)
        self.token_guard = token_guard
    
    async def check_health(self) -> HealthCheckResult:
        try:
            stats = self.token_guard.stats
            utilization = stats.utilization_rate
            
            if utilization >= 0.95:
                status = HealthStatus.CRITICAL
                message = f"Critical token usage: {utilization*100:.1f}%"
                recovery_actions = [RecoveryAction.FORCE_CLEANUP]
                suggestions = [
                    "Force context compression",
                    "Clear old context windows",
                    "Reduce context window size"
                ]
            elif utilization >= 0.85:
                status = HealthStatus.WARNING
                message = f"High token usage: {utilization*100:.1f}%"
                recovery_actions = [RecoveryAction.CLEAR_CACHE]
                suggestions = [
                    "Trigger context compression",
                    "Review context retention policy"
                ]
            else:
                status = HealthStatus.HEALTHY
                message = f"Token usage normal: {utilization*100:.1f}%"
                recovery_actions = []
                suggestions = []
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'current_tokens': stats.current_tokens,
                    'max_tokens': stats.max_tokens,
                    'utilization_percent': round(utilization * 100, 2),
                    'compression_ratio': round(stats.compression_ratio * 100, 2)
                },
                suggestions=suggestions,
                recovery_actions=recovery_actions
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.FAILED,
                message=f"Token usage check failed: {e}",
                timestamp=datetime.now()
            )


class FileSystemHealthCheck(BaseHealthCheck):
    """파일 시스템 헬스체크"""
    
    def __init__(self, workspace_path: str, warning_threshold: float = 85.0, critical_threshold: float = 95.0):
        super().__init__("filesystem", critical=True)
        self.workspace_path = workspace_path
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    async def check_health(self) -> HealthCheckResult:
        try:
            disk_usage = psutil.disk_usage(self.workspace_path)
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            disk_free_gb = disk_usage.free / (1024**3)
            
            # 열린 파일 수 확인
            process = psutil.Process()
            open_files = len(process.open_files())
            
            issues = []
            recovery_actions = []
            suggestions = []
            
            # 디스크 사용량 체크
            if disk_percent >= self.critical_threshold:
                status = HealthStatus.CRITICAL
                issues.append(f"Critical disk usage: {disk_percent:.1f}%")
                recovery_actions.extend([RecoveryAction.FORCE_CLEANUP])
                suggestions.extend([
                    "Clean up temporary files",
                    "Remove old log files",
                    "Clear workspace cache"
                ])
            elif disk_percent >= self.warning_threshold:
                status = HealthStatus.WARNING
                issues.append(f"High disk usage: {disk_percent:.1f}%")
                suggestions.extend([
                    "Monitor disk usage",
                    "Consider cleanup"
                ])
            else:
                status = HealthStatus.HEALTHY
            
            # 열린 파일 수 체크
            if open_files > 1000:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"High open files count: {open_files}")
                suggestions.append("Check for file handle leaks")
            
            message = "; ".join(issues) if issues else f"Filesystem healthy: {disk_percent:.1f}% used"
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'disk_usage_percent': round(disk_percent, 2),
                    'disk_free_gb': round(disk_free_gb, 2),
                    'disk_total_gb': round(disk_usage.total / (1024**3), 2),
                    'open_files_count': open_files
                },
                suggestions=suggestions,
                recovery_actions=recovery_actions
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.FAILED,
                message=f"Filesystem check failed: {e}",
                timestamp=datetime.now()
            )


class DeadlockHealthCheck(BaseHealthCheck):
    """데드락 상태 헬스체크"""
    
    def __init__(self, deadlock_detector):
        super().__init__("deadlock", critical=True)
        self.deadlock_detector = deadlock_detector
    
    async def check_health(self) -> HealthCheckResult:
        try:
            status_report = self.deadlock_detector.get_status_report()
            recent_deadlocks = status_report.get('recent_deadlocks', [])
            
            # 최근 5분 내 데드락 확인
            current_time = datetime.now()
            recent_critical = []
            
            for deadlock in recent_deadlocks:
                detected_time = datetime.fromisoformat(deadlock['detected_at'])
                if (current_time - detected_time).total_seconds() < 300:  # 5분
                    if deadlock['severity'] in ['critical', 'high']:
                        recent_critical.append(deadlock)
            
            if recent_critical:
                status = HealthStatus.CRITICAL
                message = f"Recent critical deadlocks detected: {len(recent_critical)}"
                recovery_actions = [RecoveryAction.RESTART_COMPONENT]
                suggestions = [
                    "Review task dependencies",
                    "Check worker pool status",
                    "Consider reducing parallel tasks"
                ]
            elif recent_deadlocks:
                status = HealthStatus.WARNING
                message = f"Recent deadlocks detected: {len(recent_deadlocks)}"
                recovery_actions = []
                suggestions = [
                    "Monitor deadlock patterns",
                    "Review task scheduling"
                ]
            else:
                status = HealthStatus.HEALTHY
                message = "No recent deadlocks detected"
                recovery_actions = []
                suggestions = []
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'recent_deadlocks': len(recent_deadlocks),
                    'critical_deadlocks': len(recent_critical),
                    'monitoring_active': status_report['monitoring_status']['active']
                },
                suggestions=suggestions,
                recovery_actions=recovery_actions
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.FAILED,
                message=f"Deadlock check failed: {e}",
                timestamp=datetime.now()
            )


class ProcessManagerHealthCheck(BaseHealthCheck):
    """프로세스 매니저 상태 헬스체크"""
    
    def __init__(self, process_manager, warning_threshold: int = 30, critical_threshold: int = 40):
        super().__init__("process_manager", critical=True)
        self.process_manager = process_manager
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    async def check_health(self) -> HealthCheckResult:
        try:
            stats = self.process_manager.get_process_stats()
            current_processes = stats['current_processes']
            max_processes = stats['max_processes']
            
            if current_processes >= self.critical_threshold:
                status = HealthStatus.CRITICAL
                message = f"Critical process count: {current_processes} processes"
                recovery_actions = [RecoveryAction.FORCE_CLEANUP]
                suggestions = [
                    "Force cleanup of orphaned processes",
                    "Review process creation patterns",
                    "Check for process leaks"
                ]
            elif current_processes >= self.warning_threshold:
                status = HealthStatus.WARNING
                message = f"High process count: {current_processes} processes"
                recovery_actions = []
                suggestions = [
                    "Monitor process creation",
                    "Consider process cleanup"
                ]
            else:
                status = HealthStatus.HEALTHY
                message = f"Process count normal: {current_processes} processes"
                recovery_actions = []
                suggestions = []
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'current_processes': current_processes,
                    'max_processes': max_processes,
                    'running_processes': stats['running_processes'],
                    'oldest_process_age': stats['oldest_process_age'],
                    'creator_breakdown': stats['creator_breakdown']
                },
                suggestions=suggestions,
                recovery_actions=recovery_actions
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.FAILED,
                message=f"Process manager check failed: {e}",
                timestamp=datetime.now()
            )


class CircuitBreakerHealthCheck(BaseHealthCheck):
    """서킷 브레이커 상태 헬스체크"""
    
    def __init__(self, circuit_manager):
        super().__init__("circuit_breakers", critical=False)
        self.circuit_manager = circuit_manager
    
    async def check_health(self) -> HealthCheckResult:
        try:
            health_summary = self.circuit_manager.get_health_summary()
            overall_health = health_summary['overall_health']
            
            open_breakers = health_summary['open_circuit_breakers']
            total_breakers = health_summary['total_circuit_breakers']
            
            if overall_health == 'critical':
                status = HealthStatus.CRITICAL
                message = f"All circuit breakers open ({total_breakers})"
                recovery_actions = [RecoveryAction.RESTART_COMPONENT]
                suggestions = [
                    "Check API connectivity",
                    "Review API quotas",
                    "Consider fallback mechanisms"
                ]
            elif overall_health == 'degraded':
                status = HealthStatus.WARNING
                message = f"Multiple circuit breakers open ({open_breakers}/{total_breakers})"
                recovery_actions = []
                suggestions = [
                    "Monitor API status",
                    "Check error patterns"
                ]
            else:
                status = HealthStatus.HEALTHY
                message = f"Circuit breakers healthy ({total_breakers - open_breakers}/{total_breakers} closed)"
                recovery_actions = []
                suggestions = []
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=health_summary,
                suggestions=suggestions,
                recovery_actions=recovery_actions
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.FAILED,
                message=f"Circuit breaker check failed: {e}",
                timestamp=datetime.now()
            )


class SystemHealthMonitor:
    """종합적인 시스템 헬스 모니터링"""
    
    def __init__(self, workspace_path: str, check_interval: int = 30):
        self.workspace_path = workspace_path
        self.check_interval = check_interval
        
        # 헬스체크 목록
        self.health_checks: List[BaseHealthCheck] = []
        
        # 복구 액션 핸들러
        self.recovery_handlers: Dict[RecoveryAction, Callable] = {}
        
        # 상태 추적
        self.monitoring_active = False
        self.monitoring_task = None
        self.shutdown_event = asyncio.Event()
        
        # 결과 기록
        self.health_history: List[HealthCheckResult] = []
        self.system_metrics_history: List[SystemMetrics] = []
        
        # 알림 콜백
        self.alert_callbacks: List[Callable] = []
        
        # 통계
        self.stats = {
            'total_checks': 0,
            'total_recoveries': 0,
            'total_alerts': 0,
            'uptime_start': datetime.now()
        }
    
    def add_health_check(self, health_check: BaseHealthCheck):
        """헬스체크 추가"""
        self.health_checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    def add_recovery_handler(self, action: RecoveryAction, handler: Callable):
        """복구 핸들러 추가"""
        self.recovery_handlers[action] = handler
        logger.info(f"Added recovery handler for: {action.value}")
    
    def add_alert_callback(self, callback: Callable):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self):
        """모니터링 시작"""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Health monitoring started (interval: {self.check_interval}s)")
    
    async def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        self.shutdown_event.set()
        
        if self.monitoring_task and not self.monitoring_task.done():
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.monitoring_task.cancel()
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # 헬스체크 수행
                health_results = await self._perform_health_checks()
                
                # 시스템 메트릭 수집
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # 결과 분석 및 복구 액션 수행
                await self._analyze_and_recover(health_results)
                
                # 알림 발송
                await self._send_alerts_if_needed(health_results)
                
                # 기록 정리
                self._cleanup_old_records()
                
                self.stats['total_checks'] += 1
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self) -> List[HealthCheckResult]:
        """모든 헬스체크 수행"""
        results = []
        
        for health_check in self.health_checks:
            try:
                result = await health_check.check_health()
                results.append(result)
                self.health_history.append(result)
                
                # 성공 시 실패 카운트 리셋
                if result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]:
                    health_check.reset_failure_count()
                else:
                    health_check.consecutive_failures += 1
                
            except Exception as e:
                logger.error(f"Health check failed for {health_check.name}: {e}")
                
                # 실패 결과 생성
                failed_result = HealthCheckResult(
                    component=health_check.name,
                    status=HealthStatus.FAILED,
                    message=f"Health check exception: {e}",
                    timestamp=datetime.now()
                )
                results.append(failed_result)
                self.health_history.append(failed_result)
                
                health_check.consecutive_failures += 1
        
        return results
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # 디스크 정보
            disk = psutil.disk_usage(self.workspace_path)
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # 프로세스 정보
            process = psutil.Process()
            open_files_count = len(process.open_files())
            
            # 네트워크 연결
            network_connections = len(psutil.net_connections())
            
            # 프로세스 수
            process_count = len(psutil.pids())
            
            # 로드 평균 (Unix 계열만)
            try:
                load_average = os.getloadavg()
            except (OSError, AttributeError):
                load_average = (0.0, 0.0, 0.0)
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                open_files_count=open_files_count,
                network_connections=network_connections,
                process_count=process_count,
                load_average=load_average
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics()
    
    async def _analyze_and_recover(self, health_results: List[HealthCheckResult]):
        """결과 분석 및 복구 액션 수행"""
        critical_results = [r for r in health_results if r.status == HealthStatus.CRITICAL]
        
        for result in critical_results:
            for action in result.recovery_actions:
                if action in self.recovery_handlers:
                    try:
                        logger.warning(f"Executing recovery action: {action.value} for {result.component}")
                        await self.recovery_handlers[action](result)
                        self.stats['total_recoveries'] += 1
                    except Exception as e:
                        logger.error(f"Recovery action {action.value} failed: {e}")
    
    async def _send_alerts_if_needed(self, health_results: List[HealthCheckResult]):
        """필요시 알림 발송"""
        critical_results = [r for r in health_results if r.status == HealthStatus.CRITICAL]
        warning_results = [r for r in health_results if r.status == HealthStatus.WARNING]
        
        if critical_results or warning_results:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'critical_issues': len(critical_results),
                'warning_issues': len(warning_results),
                'critical_components': [r.component for r in critical_results],
                'warning_components': [r.component for r in warning_results],
                'details': health_results
            }
            
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert_data)
                    else:
                        callback(alert_data)
                    self.stats['total_alerts'] += 1
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def _cleanup_old_records(self):
        """오래된 기록 정리"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        # 헬스 기록 정리
        self.health_history = [
            result for result in self.health_history
            if result.timestamp > cutoff_time
        ]
        
        # 시스템 메트릭 정리 (더 긴 보관)
        metric_cutoff = datetime.now() - timedelta(hours=6)
        self.system_metrics_history = [
            metrics for metrics in self.system_metrics_history[-360:]  # 최대 360개 (6시간)
        ]
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """현재 헬스 상태 반환"""
        if not self.health_history:
            return {'overall_status': 'unknown', 'components': {}}
        
        # 각 컴포넌트의 최신 상태
        latest_results = {}
        for result in reversed(self.health_history):
            if result.component not in latest_results:
                latest_results[result.component] = result
        
        # 전체 상태 결정
        statuses = [result.status for result in latest_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = 'critical'
        elif HealthStatus.FAILED in statuses:
            overall_status = 'failed'
        elif HealthStatus.WARNING in statuses:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        # 컴포넌트별 상태
        components = {}
        for component, result in latest_results.items():
            components[component] = {
                'status': result.status.value,
                'message': result.message,
                'last_check': result.timestamp.isoformat(),
                'metrics': result.metrics,
                'suggestions': result.suggestions
            }
        
        return {
            'overall_status': overall_status,
            'components': components,
            'monitoring_active': self.monitoring_active,
            'last_check': max(r.timestamp for r in latest_results.values()).isoformat(),
            'stats': self.stats
        }
    
    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """시스템 메트릭 요약"""
        if not self.system_metrics_history:
            return {}
        
        latest = self.system_metrics_history[-1]
        
        # 최근 1시간 평균 (최대 120개 샘플)
        recent_metrics = self.system_metrics_history[-120:]
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            'current': {
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'memory_available_gb': latest.memory_available_gb,
                'disk_usage_percent': latest.disk_usage_percent,
                'disk_free_gb': latest.disk_free_gb,
                'open_files_count': latest.open_files_count,
                'network_connections': latest.network_connections,
                'process_count': latest.process_count,
                'load_average': latest.load_average
            },
            'averages_1h': {
                'cpu_percent': round(avg_cpu, 2),
                'memory_percent': round(avg_memory, 2)
            },
            'samples_count': len(recent_metrics)
        }
    
    async def force_health_check(self) -> Dict[str, Any]:
        """즉시 헬스체크 수행"""
        health_results = await self._perform_health_checks()
        system_metrics = self._collect_system_metrics()
        
        return {
            'health_results': [
                {
                    'component': r.component,
                    'status': r.status.value,
                    'message': r.message,
                    'timestamp': r.timestamp.isoformat(),
                    'metrics': r.metrics,
                    'suggestions': r.suggestions
                }
                for r in health_results
            ],
            'system_metrics': system_metrics.__dict__,
            'timestamp': datetime.now().isoformat()
        }


# 기본 복구 핸들러들
async def default_force_cleanup_handler(result: HealthCheckResult):
    """기본 강제 정리 핸들러"""
    logger.info(f"Executing force cleanup for {result.component}")
    
    # 가비지 컬렉션
    import gc
    gc.collect()
    
    # 임시 파일 정리
    try:
        import tempfile
        import shutil
        temp_dir = tempfile.gettempdir()
        for item in os.listdir(temp_dir):
            if item.startswith('vibe_') or item.startswith('tmp_'):
                item_path = os.path.join(temp_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except OSError:
                    pass
    except Exception as e:
        logger.warning(f"Temp file cleanup failed: {e}")


async def default_clear_cache_handler(result: HealthCheckResult):
    """기본 캐시 정리 핸들러"""
    logger.info(f"Executing cache clear for {result.component}")
    
    # 가비지 컬렉션
    import gc
    gc.collect()


async def default_restart_component_handler(result: HealthCheckResult):
    """기본 컴포넌트 재시작 핸들러"""
    logger.warning(f"Component restart requested for {result.component}")
    # 실제 재시작은 상위 시스템에서 처리


async def default_process_cleanup_handler(result: HealthCheckResult):
    """프로세스 정리 핸들러"""
    logger.info(f"Executing process cleanup for {result.component}")
    
    try:
        from shared.core.process_manager import global_process_manager
        
        # 고아 프로세스 정리
        orphaned_count = await global_process_manager.cleanup_orphaned_processes()
        logger.info(f"Cleaned up {orphaned_count} orphaned processes")
        
        # 오래 실행되는 프로세스 체크 (1시간 이상)
        stats = global_process_manager.get_process_stats()
        if stats['oldest_process_age'] > 3600:
            logger.warning(f"Long-running process detected: {stats['oldest_process_age']:.0f}s old")
        
    except Exception as e:
        logger.error(f"Process cleanup failed: {e}")
    

# 전역 헬스 모니터 인스턴스 생성을 위한 팩토리 함수
def create_system_health_monitor(workspace_path: str, 
                                token_guard=None, 
                                deadlock_detector=None, 
                                circuit_manager=None,
                                process_manager=None) -> SystemHealthMonitor:
    """시스템 헬스 모니터 생성"""
    monitor = SystemHealthMonitor(workspace_path)
    
    # 기본 헬스체크 추가
    monitor.add_health_check(MemoryHealthCheck())
    monitor.add_health_check(FileSystemHealthCheck(workspace_path))
    
    # 의존성이 있는 헬스체크들
    if token_guard:
        monitor.add_health_check(TokenUsageHealthCheck(token_guard))
    
    if deadlock_detector:
        monitor.add_health_check(DeadlockHealthCheck(deadlock_detector))
    
    if circuit_manager:
        monitor.add_health_check(CircuitBreakerHealthCheck(circuit_manager))
    
    if process_manager:
        monitor.add_health_check(ProcessManagerHealthCheck(process_manager))
    
    # 기본 복구 핸들러 등록
    monitor.add_recovery_handler(RecoveryAction.FORCE_CLEANUP, default_process_cleanup_handler)
    monitor.add_recovery_handler(RecoveryAction.CLEAR_CACHE, default_clear_cache_handler)
    monitor.add_recovery_handler(RecoveryAction.RESTART_COMPONENT, default_restart_component_handler)
    
    return monitor