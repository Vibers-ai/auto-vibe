"""Deadlock Detection and Recovery System - 데드락 감지 및 복구 시스템"""

import asyncio
import logging
import time
import threading
from typing import Dict, Set, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DeadlockType(Enum):
    """데드락 유형"""
    RESOURCE_DEADLOCK = "resource"      # 리소스 경합
    DEPENDENCY_DEADLOCK = "dependency"  # 의존성 순환
    WORKER_DEADLOCK = "worker"          # 워커 풀 데드락
    MIXED_DEADLOCK = "mixed"            # 복합 데드락


class DeadlockSeverity(Enum):
    """데드락 심각도"""
    LOW = "low"           # 일부 작업 지연
    MEDIUM = "medium"     # 워커 풀 일부 블록
    HIGH = "high"         # 대부분 작업 중단
    CRITICAL = "critical" # 전체 시스템 멈춤


@dataclass
class TaskNode:
    """작업 노드 정보"""
    task_id: str
    worker_id: Optional[str] = None
    status: str = "pending"  # pending, running, waiting, blocked, completed, failed
    dependencies: Set[str] = field(default_factory=set)
    waiting_for: Set[str] = field(default_factory=set)  # 대기 중인 리소스/작업
    resources_held: Set[str] = field(default_factory=set)  # 보유 중인 리소스
    start_time: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    timeout: Optional[float] = None


@dataclass
class DeadlockInfo:
    """데드락 정보"""
    deadlock_id: str
    detected_at: datetime
    deadlock_type: DeadlockType
    severity: DeadlockSeverity
    involved_tasks: Set[str]
    involved_workers: Set[str]
    involved_resources: Set[str]
    cycle_path: List[str]
    resolution_strategy: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_successful: bool = False


class DeadlockDetector:
    """데드락 감지 및 복구 시스템"""
    
    def __init__(self, detection_interval: int = 10, timeout_threshold: int = 300):
        self.detection_interval = detection_interval  # 감지 주기 (초)
        self.timeout_threshold = timeout_threshold    # 작업 타임아웃 임계값 (초)
        
        # 그래프 구조
        self.task_dependency_graph = nx.DiGraph()  # 작업 의존성 그래프
        self.resource_allocation_graph = nx.DiGraph()  # 리소스 할당 그래프
        self.worker_task_graph = nx.DiGraph()  # 워커-작업 그래프
        
        # 노드 정보
        self.task_nodes: Dict[str, TaskNode] = {}
        self.worker_status: Dict[str, Dict[str, Any]] = {}
        self.resource_locks: Dict[str, str] = {}  # resource_id -> task_id
        
        # 데드락 기록
        self.detected_deadlocks: List[DeadlockInfo] = []
        self.deadlock_counter = 0
        
        # 모니터링 상태
        self.monitoring_active = False
        self.monitoring_task = None
        self.shutdown_event = asyncio.Event()
        
        # 통계
        self.stats = {
            'total_deadlocks_detected': 0,
            'total_deadlocks_resolved': 0,
            'false_positives': 0,
            'avg_detection_time': 0.0,
            'avg_resolution_time': 0.0
        }
    
    def register_task(self, task_id: str, dependencies: Set[str] = None, timeout: float = None):
        """작업 등록"""
        if dependencies is None:
            dependencies = set()
            
        task_node = TaskNode(
            task_id=task_id,
            dependencies=dependencies,
            timeout=timeout or self.timeout_threshold,
            start_time=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.task_nodes[task_id] = task_node
        
        # 의존성 그래프 업데이트
        self.task_dependency_graph.add_node(task_id)
        for dep_id in dependencies:
            if dep_id in self.task_nodes:
                self.task_dependency_graph.add_edge(dep_id, task_id)
        
        logger.debug(f"Registered task: {task_id} with dependencies: {dependencies}")
    
    def update_task_status(self, task_id: str, status: str, worker_id: str = None):
        """작업 상태 업데이트"""
        if task_id not in self.task_nodes:
            logger.warning(f"Task {task_id} not registered")
            return
        
        task_node = self.task_nodes[task_id]
        old_status = task_node.status
        task_node.status = status
        task_node.last_activity = datetime.now()
        
        if worker_id:
            task_node.worker_id = worker_id
            
            # 워커-작업 그래프 업데이트
            if status == "running":
                self.worker_task_graph.add_edge(worker_id, task_id)
            elif status in ["completed", "failed"]:
                if self.worker_task_graph.has_edge(worker_id, task_id):
                    self.worker_task_graph.remove_edge(worker_id, task_id)
        
        logger.debug(f"Task {task_id} status: {old_status} -> {status}")
    
    def add_resource_wait(self, task_id: str, resource_id: str):
        """리소스 대기 상태 추가"""
        if task_id not in self.task_nodes:
            return
        
        task_node = self.task_nodes[task_id]
        task_node.waiting_for.add(resource_id)
        
        # 리소스 할당 그래프 업데이트
        self.resource_allocation_graph.add_edge(task_id, resource_id, edge_type="waiting")
        
        logger.debug(f"Task {task_id} waiting for resource: {resource_id}")
    
    def add_resource_lock(self, task_id: str, resource_id: str):
        """리소스 락 획득"""
        if task_id not in self.task_nodes:
            return
        
        task_node = self.task_nodes[task_id]
        task_node.resources_held.add(resource_id)
        task_node.waiting_for.discard(resource_id)
        
        self.resource_locks[resource_id] = task_id
        
        # 리소스 할당 그래프 업데이트
        if self.resource_allocation_graph.has_edge(task_id, resource_id):
            self.resource_allocation_graph.remove_edge(task_id, resource_id)
        self.resource_allocation_graph.add_edge(resource_id, task_id, edge_type="allocated")
        
        logger.debug(f"Task {task_id} acquired resource: {resource_id}")
    
    def remove_resource_lock(self, task_id: str, resource_id: str):
        """리소스 락 해제"""
        if task_id not in self.task_nodes:
            return
        
        task_node = self.task_nodes[task_id]
        task_node.resources_held.discard(resource_id)
        
        if resource_id in self.resource_locks and self.resource_locks[resource_id] == task_id:
            del self.resource_locks[resource_id]
        
        # 리소스 할당 그래프 업데이트
        if self.resource_allocation_graph.has_edge(resource_id, task_id):
            self.resource_allocation_graph.remove_edge(resource_id, task_id)
        
        logger.debug(f"Task {task_id} released resource: {resource_id}")
    
    async def start_monitoring(self):
        """데드락 모니터링 시작"""
        if self.monitoring_active:
            logger.warning("Deadlock monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Deadlock monitoring started (interval: {self.detection_interval}s)")
    
    async def stop_monitoring(self):
        """데드락 모니터링 중지"""
        self.monitoring_active = False
        self.shutdown_event.set()
        
        if self.monitoring_task and not self.monitoring_task.done():
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.monitoring_task.cancel()
        
        logger.info("Deadlock monitoring stopped")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # 데드락 감지
                deadlocks = await self._detect_deadlocks()
                
                # 감지된 데드락 처리
                for deadlock in deadlocks:
                    await self._handle_deadlock(deadlock)
                
                # 타임아웃 작업 확인
                await self._check_timeouts()
                
                # 오래된 기록 정리
                self._cleanup_old_records()
                
                await asyncio.sleep(self.detection_interval)
                
            except Exception as e:
                logger.error(f"Deadlock monitoring error: {e}")
                await asyncio.sleep(self.detection_interval)
    
    async def _detect_deadlocks(self) -> List[DeadlockInfo]:
        """데드락 감지"""
        detected = []
        
        try:
            # 1. 의존성 순환 감지
            dependency_cycles = self._detect_dependency_cycles()
            for cycle in dependency_cycles:
                deadlock = self._create_deadlock_info(
                    DeadlockType.DEPENDENCY_DEADLOCK,
                    cycle,
                    "Circular dependency detected"
                )
                detected.append(deadlock)
            
            # 2. 리소스 데드락 감지
            resource_cycles = self._detect_resource_cycles()
            for cycle in resource_cycles:
                deadlock = self._create_deadlock_info(
                    DeadlockType.RESOURCE_DEADLOCK,
                    cycle,
                    "Resource allocation deadlock"
                )
                detected.append(deadlock)
            
            # 3. 워커 풀 데드락 감지
            worker_deadlocks = self._detect_worker_deadlocks()
            for deadlock_info in worker_deadlocks:
                detected.append(deadlock_info)
            
        except Exception as e:
            logger.error(f"Deadlock detection error: {e}")
        
        return detected
    
    def _detect_dependency_cycles(self) -> List[List[str]]:
        """의존성 순환 감지"""
        cycles = []
        
        try:
            # NetworkX를 사용한 순환 감지
            for cycle in nx.simple_cycles(self.task_dependency_graph):
                if len(cycle) > 1:  # 자기 자신 순환 제외
                    cycles.append(cycle)
        except Exception as e:
            logger.error(f"Dependency cycle detection error: {e}")
        
        return cycles
    
    def _detect_resource_cycles(self) -> List[List[str]]:
        """리소스 할당 순환 감지"""
        cycles = []
        
        try:
            # 대기-보유 그래프 구성
            wait_for_graph = nx.DiGraph()
            
            for task_id, task_node in self.task_nodes.items():
                if task_node.status in ["running", "waiting"]:
                    # 이 작업이 대기하는 리소스들
                    for resource_id in task_node.waiting_for:
                        # 그 리소스를 보유한 작업 찾기
                        holder = self.resource_locks.get(resource_id)
                        if holder and holder != task_id:
                            wait_for_graph.add_edge(task_id, holder)
            
            # 순환 감지
            for cycle in nx.simple_cycles(wait_for_graph):
                if len(cycle) > 1:
                    cycles.append(cycle)
                    
        except Exception as e:
            logger.error(f"Resource cycle detection error: {e}")
        
        return cycles
    
    def _detect_worker_deadlocks(self) -> List[DeadlockInfo]:
        """워커 풀 데드락 감지"""
        deadlocks = []
        
        try:
            # 모든 워커가 블록된 상태 확인
            blocked_workers = 0
            total_workers = len(self.worker_status)
            
            if total_workers == 0:
                return deadlocks
            
            current_time = datetime.now()
            
            for worker_id, status in self.worker_status.items():
                if status.get('status') == 'blocked':
                    last_activity = status.get('last_activity')
                    if last_activity and (current_time - last_activity).total_seconds() > 60:  # 1분 이상 블록
                        blocked_workers += 1
            
            # 대부분의 워커가 블록된 경우
            if blocked_workers >= total_workers * 0.8:  # 80% 이상
                deadlock = DeadlockInfo(
                    deadlock_id=f"worker_deadlock_{self.deadlock_counter}",
                    detected_at=current_time,
                    deadlock_type=DeadlockType.WORKER_DEADLOCK,
                    severity=DeadlockSeverity.CRITICAL,
                    involved_tasks=set(),
                    involved_workers=set(self.worker_status.keys()),
                    involved_resources=set(),
                    cycle_path=[]
                )
                deadlocks.append(deadlock)
                self.deadlock_counter += 1
                
        except Exception as e:
            logger.error(f"Worker deadlock detection error: {e}")
        
        return deadlocks
    
    def _create_deadlock_info(self, deadlock_type: DeadlockType, cycle: List[str], description: str) -> DeadlockInfo:
        """데드락 정보 생성"""
        
        # 관련 작업, 워커, 리소스 추출
        involved_tasks = set()
        involved_workers = set()
        involved_resources = set()
        
        for node_id in cycle:
            if node_id in self.task_nodes:
                involved_tasks.add(node_id)
                task_node = self.task_nodes[node_id]
                if task_node.worker_id:
                    involved_workers.add(task_node.worker_id)
                involved_resources.update(task_node.resources_held)
                involved_resources.update(task_node.waiting_for)
        
        # 심각도 결정
        severity = self._determine_severity(involved_tasks, involved_workers)
        
        deadlock = DeadlockInfo(
            deadlock_id=f"{deadlock_type.value}_deadlock_{self.deadlock_counter}",
            detected_at=datetime.now(),
            deadlock_type=deadlock_type,
            severity=severity,
            involved_tasks=involved_tasks,
            involved_workers=involved_workers,
            involved_resources=involved_resources,
            cycle_path=cycle
        )
        
        self.deadlock_counter += 1
        return deadlock
    
    def _determine_severity(self, involved_tasks: Set[str], involved_workers: Set[str]) -> DeadlockSeverity:
        """데드락 심각도 결정"""
        total_tasks = len(self.task_nodes)
        total_workers = len(self.worker_status)
        
        task_ratio = len(involved_tasks) / max(total_tasks, 1)
        worker_ratio = len(involved_workers) / max(total_workers, 1)
        
        if task_ratio >= 0.8 or worker_ratio >= 0.8:
            return DeadlockSeverity.CRITICAL
        elif task_ratio >= 0.5 or worker_ratio >= 0.5:
            return DeadlockSeverity.HIGH
        elif task_ratio >= 0.2 or worker_ratio >= 0.2:
            return DeadlockSeverity.MEDIUM
        else:
            return DeadlockSeverity.LOW
    
    async def _handle_deadlock(self, deadlock: DeadlockInfo):
        """데드락 처리"""
        logger.critical(f"Deadlock detected: {deadlock.deadlock_id} "
                       f"(type: {deadlock.deadlock_type.value}, severity: {deadlock.severity.value})")
        
        self.detected_deadlocks.append(deadlock)
        self.stats['total_deadlocks_detected'] += 1
        
        try:
            # 해결 전략 선택
            strategy = self._select_resolution_strategy(deadlock)
            deadlock.resolution_strategy = strategy
            
            # 해결 시도
            success = await self._execute_resolution_strategy(deadlock, strategy)
            
            deadlock.resolved_at = datetime.now()
            deadlock.resolution_successful = success
            
            if success:
                self.stats['total_deadlocks_resolved'] += 1
                logger.info(f"Deadlock {deadlock.deadlock_id} resolved using strategy: {strategy}")
            else:
                logger.error(f"Failed to resolve deadlock {deadlock.deadlock_id}")
                
        except Exception as e:
            logger.error(f"Deadlock resolution error: {e}")
            deadlock.resolved_at = datetime.now()
            deadlock.resolution_successful = False
    
    def _select_resolution_strategy(self, deadlock: DeadlockInfo) -> str:
        """해결 전략 선택"""
        
        if deadlock.deadlock_type == DeadlockType.DEPENDENCY_DEADLOCK:
            return "break_dependency_cycle"
        elif deadlock.deadlock_type == DeadlockType.RESOURCE_DEADLOCK:
            return "force_resource_release"
        elif deadlock.deadlock_type == DeadlockType.WORKER_DEADLOCK:
            return "restart_workers"
        else:
            return "terminate_oldest_task"
    
    async def _execute_resolution_strategy(self, deadlock: DeadlockInfo, strategy: str) -> bool:
        """해결 전략 실행"""
        
        try:
            if strategy == "break_dependency_cycle":
                return await self._break_dependency_cycle(deadlock)
            elif strategy == "force_resource_release":
                return await self._force_resource_release(deadlock)
            elif strategy == "restart_workers":
                return await self._restart_workers(deadlock)
            elif strategy == "terminate_oldest_task":
                return await self._terminate_oldest_task(deadlock)
            else:
                logger.error(f"Unknown resolution strategy: {strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
            return False
    
    async def _break_dependency_cycle(self, deadlock: DeadlockInfo) -> bool:
        """의존성 순환 해결"""
        # 순환에서 가장 최근에 시작된 작업을 제거
        if not deadlock.cycle_path:
            return False
        
        # 가장 최근 작업 찾기
        latest_task = None
        latest_time = None
        
        for task_id in deadlock.cycle_path:
            if task_id in self.task_nodes:
                task_node = self.task_nodes[task_id]
                if latest_time is None or (task_node.start_time and task_node.start_time > latest_time):
                    latest_task = task_id
                    latest_time = task_node.start_time
        
        if latest_task:
            logger.warning(f"Breaking dependency cycle by removing task: {latest_task}")
            await self._terminate_task(latest_task, "deadlock_resolution")
            return True
        
        return False
    
    async def _force_resource_release(self, deadlock: DeadlockInfo) -> bool:
        """강제 리소스 해제"""
        # 순환에 관련된 리소스 중 하나를 강제 해제
        for resource_id in deadlock.involved_resources:
            if resource_id in self.resource_locks:
                task_id = self.resource_locks[resource_id]
                logger.warning(f"Force releasing resource {resource_id} from task {task_id}")
                self.remove_resource_lock(task_id, resource_id)
                return True
        
        return False
    
    async def _restart_workers(self, deadlock: DeadlockInfo) -> bool:
        """워커 재시작"""
        # 실제 워커 재시작은 상위 시스템에서 처리
        logger.critical("Worker pool deadlock detected - requesting worker restart")
        
        # 블록된 워커들을 리셋 상태로 표시
        for worker_id in deadlock.involved_workers:
            if worker_id in self.worker_status:
                self.worker_status[worker_id]['status'] = 'restart_requested'
        
        return True
    
    async def _terminate_oldest_task(self, deadlock: DeadlockInfo) -> bool:
        """가장 오래된 작업 종료"""
        oldest_task = None
        oldest_time = None
        
        for task_id in deadlock.involved_tasks:
            if task_id in self.task_nodes:
                task_node = self.task_nodes[task_id]
                if oldest_time is None or (task_node.start_time and task_node.start_time < oldest_time):
                    oldest_task = task_id
                    oldest_time = task_node.start_time
        
        if oldest_task:
            logger.warning(f"Terminating oldest task to resolve deadlock: {oldest_task}")
            await self._terminate_task(oldest_task, "deadlock_resolution")
            return True
        
        return False
    
    async def _terminate_task(self, task_id: str, reason: str):
        """작업 종료"""
        if task_id not in self.task_nodes:
            return
        
        task_node = self.task_nodes[task_id]
        
        # 상태 업데이트
        task_node.status = "terminated"
        task_node.last_activity = datetime.now()
        
        # 보유 리소스 해제
        for resource_id in list(task_node.resources_held):
            self.remove_resource_lock(task_id, resource_id)
        
        # 대기 상태 정리
        task_node.waiting_for.clear()
        
        # 그래프에서 제거
        if self.task_dependency_graph.has_node(task_id):
            self.task_dependency_graph.remove_node(task_id)
        
        # 워커-작업 연결 해제
        if task_node.worker_id and self.worker_task_graph.has_edge(task_node.worker_id, task_id):
            self.worker_task_graph.remove_edge(task_node.worker_id, task_id)
        
        logger.warning(f"Task {task_id} terminated due to: {reason}")
    
    async def _check_timeouts(self):
        """타임아웃 작업 확인"""
        current_time = datetime.now()
        
        for task_id, task_node in self.task_nodes.items():
            if task_node.status in ["running", "waiting"] and task_node.start_time:
                elapsed = (current_time - task_node.start_time).total_seconds()
                
                if elapsed > task_node.timeout:
                    logger.warning(f"Task {task_id} timeout after {elapsed:.1f}s")
                    await self._terminate_task(task_id, "timeout")
    
    def _cleanup_old_records(self):
        """오래된 기록 정리"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        # 완료된 작업 정리
        completed_tasks = [
            task_id for task_id, task_node in self.task_nodes.items()
            if task_node.status in ["completed", "failed", "terminated"] 
            and task_node.last_activity 
            and task_node.last_activity < cutoff_time
        ]
        
        for task_id in completed_tasks:
            del self.task_nodes[task_id]
            
            # 그래프에서도 제거
            if self.task_dependency_graph.has_node(task_id):
                self.task_dependency_graph.remove_node(task_id)
        
        # 오래된 데드락 기록 정리
        self.detected_deadlocks = [
            deadlock for deadlock in self.detected_deadlocks
            if deadlock.detected_at > cutoff_time
        ]
    
    def get_status_report(self) -> Dict[str, Any]:
        """상태 보고서 생성"""
        current_time = datetime.now()
        
        # 작업 상태 통계
        status_counts = defaultdict(int)
        for task_node in self.task_nodes.values():
            status_counts[task_node.status] += 1
        
        # 최근 데드락 정보
        recent_deadlocks = [
            {
                'id': deadlock.deadlock_id,
                'type': deadlock.deadlock_type.value,
                'severity': deadlock.severity.value,
                'detected_at': deadlock.detected_at.isoformat(),
                'resolved': deadlock.resolution_successful,
                'resolution_time': (
                    deadlock.resolved_at - deadlock.detected_at
                ).total_seconds() if deadlock.resolved_at else None
            }
            for deadlock in self.detected_deadlocks[-5:]  # 최근 5개
        ]
        
        return {
            'monitoring_status': {
                'active': self.monitoring_active,
                'detection_interval': self.detection_interval,
                'timeout_threshold': self.timeout_threshold
            },
            'task_status': dict(status_counts),
            'resource_locks': len(self.resource_locks),
            'worker_count': len(self.worker_status),
            'graph_sizes': {
                'dependency_nodes': self.task_dependency_graph.number_of_nodes(),
                'dependency_edges': self.task_dependency_graph.number_of_edges(),
                'resource_nodes': self.resource_allocation_graph.number_of_nodes(),
                'resource_edges': self.resource_allocation_graph.number_of_edges()
            },
            'statistics': self.stats,
            'recent_deadlocks': recent_deadlocks
        }


# 전역 데드락 감지기 인스턴스
global_deadlock_detector = DeadlockDetector()