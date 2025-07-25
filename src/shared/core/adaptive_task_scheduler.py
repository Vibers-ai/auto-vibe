"""Adaptive Task Scheduler for VIBE

This module provides intelligent task scheduling with real-time progress tracking,
bottleneck detection, and dynamic priority adjustment.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
import statistics

from shared.core.schema import Task, TasksPlan
from shared.core.enhanced_logger import get_logger, LogCategory

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Task scheduling strategies."""
    FIFO = "fifo"                    # First In First Out
    PRIORITY = "priority"            # Priority-based
    SHORTEST_FIRST = "shortest_first" # Shortest job first
    CRITICAL_PATH = "critical_path"   # Critical path method
    ADAPTIVE = "adaptive"            # Adaptive based on performance


class TaskState(Enum):
    """Detailed task states for tracking."""
    PENDING = "pending"
    READY = "ready"              # Dependencies satisfied
    SCHEDULED = "scheduled"      # Assigned to worker
    EXECUTING = "executing"      # Currently running
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"          # Waiting for dependencies
    STALLED = "stalled"          # No progress detected


@dataclass
class TaskMetrics:
    """Performance metrics for a task."""
    task_id: str
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Time metrics
    queue_time: float = 0.0          # Time waiting in queue
    execution_time: float = 0.0      # Actual execution time
    total_time: float = 0.0          # Total time from schedule to complete
    
    # Progress metrics
    progress_percentage: float = 0.0
    last_progress_update: Optional[datetime] = None
    progress_rate: float = 0.0       # Progress per second
    estimated_remaining_time: float = 0.0
    
    # Performance metrics
    retry_count: int = 0
    error_count: int = 0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Dependencies
    blocking_tasks: Set[str] = field(default_factory=set)
    blocked_by_tasks: Set[str] = field(default_factory=set)


@dataclass
class SchedulingDecision:
    """Represents a scheduling decision."""
    task_id: str
    priority_score: float
    scheduling_reason: str
    estimated_duration: float
    prerequisites_met: bool
    resource_requirements: Dict[str, float] = field(default_factory=dict)


class AdaptiveTaskScheduler:
    """Intelligent task scheduler with adaptive strategies."""
    
    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.ADAPTIVE):
        self.strategy = strategy
        
        # Task tracking
        self.tasks: Dict[str, Task] = {}
        self.task_states: Dict[str, TaskState] = {}
        self.task_metrics: Dict[str, TaskMetrics] = {}
        
        # Dependency graph
        self.dependency_graph = nx.DiGraph()
        self.reverse_dependency_graph = nx.DiGraph()
        
        # Scheduling queues
        self.ready_queue: deque = deque()
        self.blocked_queue: Set[str] = set()
        self.executing_tasks: Set[str] = set()
        
        # Performance tracking
        self.historical_metrics: List[TaskMetrics] = []
        self.bottlenecks: List[Dict[str, Any]] = []
        self.performance_stats = {
            'average_queue_time': 0.0,
            'average_execution_time': 0.0,
            'throughput': 0.0,
            'success_rate': 0.0,
            'bottleneck_count': 0,
            'rescheduled_count': 0
        }
        
        # Adaptive learning
        self.task_type_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'avg_duration': 0.0,
            'success_rate': 1.0,
            'samples': 0
        })
        
        # Monitoring
        self.update_interval = 1.0  # seconds
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        self.bottleneck_callbacks: List[Callable] = []
        
        # Enhanced logger
        self.logger = get_logger(
            component="adaptive_scheduler",
            session_id=f"scheduler_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def load_tasks_plan(self, tasks_plan: TasksPlan):
        """Load tasks and build dependency graph."""
        # Clear existing data
        self.tasks.clear()
        self.dependency_graph.clear()
        self.reverse_dependency_graph.clear()
        
        # Load tasks
        for task in tasks_plan.tasks:
            self.tasks[task.id] = task
            self.task_states[task.id] = TaskState.PENDING
            self.task_metrics[task.id] = TaskMetrics(task_id=task.id)
            
            # Add to dependency graph
            self.dependency_graph.add_node(task.id, task=task)
            
            # Add edges for dependencies
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    self.dependency_graph.add_edge(dep_id, task.id)
                    self.reverse_dependency_graph.add_edge(task.id, dep_id)
        
        # Analyze critical path
        self._analyze_critical_path()
        
        # Initialize ready queue
        self._update_ready_queue()
        
        logger.info(f"Loaded {len(self.tasks)} tasks with dependency graph")
    
    async def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started adaptive task monitoring")
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped adaptive task monitoring")
    
    def schedule_next_task(self) -> Optional[SchedulingDecision]:
        """Get next task to execute based on scheduling strategy."""
        if not self.ready_queue:
            self._update_ready_queue()
            if not self.ready_queue:
                return None
        
        # Apply scheduling strategy
        if self.strategy == SchedulingStrategy.FIFO:
            task_id = self.ready_queue.popleft()
        elif self.strategy == SchedulingStrategy.PRIORITY:
            task_id = self._schedule_by_priority()
        elif self.strategy == SchedulingStrategy.SHORTEST_FIRST:
            task_id = self._schedule_shortest_first()
        elif self.strategy == SchedulingStrategy.CRITICAL_PATH:
            task_id = self._schedule_critical_path()
        else:  # ADAPTIVE
            task_id = self._schedule_adaptive()
        
        if not task_id:
            return None
        
        # Create scheduling decision
        task = self.tasks[task_id]
        decision = SchedulingDecision(
            task_id=task_id,
            priority_score=self._calculate_priority_score(task_id),
            scheduling_reason=f"Strategy: {self.strategy.value}",
            estimated_duration=self._estimate_task_duration(task_id),
            prerequisites_met=True
        )
        
        # Update state
        self.task_states[task_id] = TaskState.SCHEDULED
        self.task_metrics[task_id].scheduled_at = datetime.now()
        
        return decision
    
    def mark_task_started(self, task_id: str):
        """Mark task as started execution."""
        if task_id not in self.tasks:
            return
        
        self.task_states[task_id] = TaskState.EXECUTING
        self.executing_tasks.add(task_id)
        
        metrics = self.task_metrics[task_id]
        metrics.started_at = datetime.now()
        
        if metrics.scheduled_at:
            metrics.queue_time = (metrics.started_at - metrics.scheduled_at).total_seconds()
        
        logger.info(f"Task {task_id} started execution")
    
    def update_task_progress(self, task_id: str, progress: float, metadata: Optional[Dict[str, Any]] = None):
        """Update task execution progress."""
        if task_id not in self.task_metrics:
            return
        
        metrics = self.task_metrics[task_id]
        old_progress = metrics.progress_percentage
        metrics.progress_percentage = min(100.0, max(0.0, progress))
        metrics.last_progress_update = datetime.now()
        
        # Calculate progress rate
        if metrics.started_at and old_progress < progress:
            elapsed = (datetime.now() - metrics.started_at).total_seconds()
            if elapsed > 0:
                metrics.progress_rate = progress / elapsed
                
                # Estimate remaining time
                if metrics.progress_rate > 0 and progress < 100:
                    metrics.estimated_remaining_time = (100 - progress) / metrics.progress_rate
        
        # Update resource usage if provided
        if metadata and 'resource_usage' in metadata:
            metrics.resource_usage.update(metadata['resource_usage'])
        
        # Trigger callbacks
        for callback in self.progress_callbacks:
            asyncio.create_task(callback(task_id, progress, metrics))
    
    def mark_task_completed(self, task_id: str, success: bool = True):
        """Mark task as completed."""
        if task_id not in self.tasks:
            return
        
        self.task_states[task_id] = TaskState.COMPLETED if success else TaskState.FAILED
        self.executing_tasks.discard(task_id)
        
        metrics = self.task_metrics[task_id]
        metrics.completed_at = datetime.now()
        
        if metrics.started_at:
            metrics.execution_time = (metrics.completed_at - metrics.started_at).total_seconds()
        
        if metrics.scheduled_at:
            metrics.total_time = (metrics.completed_at - metrics.scheduled_at).total_seconds()
        
        # Update historical metrics
        self.historical_metrics.append(metrics)
        
        # Update task type performance
        task = self.tasks[task_id]
        self._update_task_type_performance(task.type, metrics, success)
        
        # Update ready queue for dependent tasks
        self._update_ready_queue()
        
        logger.info(f"Task {task_id} completed ({'success' if success else 'failed'}) in {metrics.execution_time:.2f}s")
    
    def reschedule_task(self, task_id: str, reason: str = "Manual reschedule"):
        """Reschedule a task."""
        if task_id not in self.tasks:
            return
        
        # Reset task state
        self.task_states[task_id] = TaskState.PENDING
        self.executing_tasks.discard(task_id)
        
        # Reset metrics but keep history
        old_metrics = self.task_metrics[task_id]
        old_metrics.retry_count += 1
        
        self.task_metrics[task_id] = TaskMetrics(
            task_id=task_id,
            retry_count=old_metrics.retry_count
        )
        
        self.performance_stats['rescheduled_count'] += 1
        
        # Re-evaluate ready queue
        self._update_ready_queue()
        
        logger.info(f"Rescheduled task {task_id}: {reason}")
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect bottlenecks in task execution."""
        bottlenecks = []
        current_time = datetime.now()
        
        # Check for stalled tasks
        for task_id in self.executing_tasks:
            metrics = self.task_metrics[task_id]
            
            # No progress for too long
            if metrics.last_progress_update:
                time_since_update = (current_time - metrics.last_progress_update).total_seconds()
                
                if time_since_update > 300:  # 5 minutes without progress
                    bottlenecks.append({
                        'type': 'stalled_task',
                        'task_id': task_id,
                        'duration': time_since_update,
                        'progress': metrics.progress_percentage,
                        'severity': 'high'
                    })
                    self.task_states[task_id] = TaskState.STALLED
            
            # Task taking much longer than estimated
            if metrics.started_at:
                elapsed = (current_time - metrics.started_at).total_seconds()
                estimated = self._estimate_task_duration(task_id)
                
                if elapsed > estimated * 2:  # Taking twice as long
                    bottlenecks.append({
                        'type': 'slow_execution',
                        'task_id': task_id,
                        'elapsed': elapsed,
                        'estimated': estimated,
                        'severity': 'medium'
                    })
        
        # Check for dependency bottlenecks
        blocked_counts = defaultdict(int)
        for task_id in self.blocked_queue:
            metrics = self.task_metrics[task_id]
            for blocking_task in metrics.blocking_tasks:
                blocked_counts[blocking_task] += 1
        
        for blocking_task, count in blocked_counts.items():
            if count >= 3:  # Blocking 3 or more tasks
                bottlenecks.append({
                    'type': 'dependency_bottleneck',
                    'task_id': blocking_task,
                    'blocked_count': count,
                    'severity': 'high' if count >= 5 else 'medium'
                })
        
        # Check queue congestion
        if len(self.ready_queue) > len(self.executing_tasks) * 3:
            bottlenecks.append({
                'type': 'queue_congestion',
                'queue_size': len(self.ready_queue),
                'executing': len(self.executing_tasks),
                'severity': 'medium'
            })
        
        self.bottlenecks = bottlenecks
        self.performance_stats['bottleneck_count'] = len(bottlenecks)
        
        # Trigger callbacks
        for callback in self.bottleneck_callbacks:
            asyncio.create_task(callback(bottlenecks))
        
        return bottlenecks
    
    def adjust_priorities(self, adjustments: Dict[str, float]):
        """Manually adjust task priorities."""
        for task_id, adjustment in adjustments.items():
            if task_id in self.tasks:
                # Store priority adjustment (would need to add this field to Task)
                logger.info(f"Adjusted priority for task {task_id} by {adjustment}")
        
        # Re-sort ready queue
        self._sort_ready_queue()
    
    def get_execution_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of task execution."""
        timeline = []
        
        for task_id, metrics in self.task_metrics.items():
            state = self.task_states.get(task_id, TaskState.PENDING)
            
            event = {
                'task_id': task_id,
                'state': state.value,
                'scheduled_at': metrics.scheduled_at.isoformat() if metrics.scheduled_at else None,
                'started_at': metrics.started_at.isoformat() if metrics.started_at else None,
                'completed_at': metrics.completed_at.isoformat() if metrics.completed_at else None,
                'progress': metrics.progress_percentage,
                'execution_time': metrics.execution_time,
                'queue_time': metrics.queue_time
            }
            
            timeline.append(event)
        
        # Sort by scheduled time
        timeline.sort(key=lambda x: x['scheduled_at'] or '9999')
        
        return timeline
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if self.historical_metrics:
            # Calculate statistics
            queue_times = [m.queue_time for m in self.historical_metrics if m.queue_time > 0]
            exec_times = [m.execution_time for m in self.historical_metrics if m.execution_time > 0]
            
            self.performance_stats['average_queue_time'] = statistics.mean(queue_times) if queue_times else 0
            self.performance_stats['average_execution_time'] = statistics.mean(exec_times) if exec_times else 0
            
            completed = sum(1 for t in self.task_states.values() if t == TaskState.COMPLETED)
            failed = sum(1 for t in self.task_states.values() if t == TaskState.FAILED)
            total = completed + failed
            
            self.performance_stats['success_rate'] = completed / total if total > 0 else 0
            self.performance_stats['throughput'] = total / (time.time() / 3600) if total > 0 else 0  # tasks per hour
        
        return {
            'summary': self.performance_stats,
            'task_states': dict(self.task_states),
            'bottlenecks': self.bottlenecks,
            'task_type_performance': dict(self.task_type_performance),
            'queue_status': {
                'ready': len(self.ready_queue),
                'blocked': len(self.blocked_queue),
                'executing': len(self.executing_tasks)
            }
        }
    
    # Private methods
    
    def _analyze_critical_path(self):
        """Analyze and mark critical path tasks."""
        if not self.dependency_graph:
            return
        
        # Find tasks with no successors (end tasks)
        end_tasks = [n for n in self.dependency_graph.nodes() if self.dependency_graph.out_degree(n) == 0]
        
        # Calculate longest paths
        critical_tasks = set()
        
        for end_task in end_tasks:
            # Find all paths to this end task
            start_tasks = [n for n in self.dependency_graph.nodes() if self.dependency_graph.in_degree(n) == 0]
            
            for start_task in start_tasks:
                try:
                    paths = list(nx.all_simple_paths(self.dependency_graph, start_task, end_task))
                    
                    # Find longest path by estimated duration
                    for path in paths:
                        path_duration = sum(self._estimate_task_duration(t) for t in path)
                        # Mark tasks in longest paths
                        if path_duration > 0:  # Threshold for critical
                            critical_tasks.update(path)
                except nx.NetworkXNoPath:
                    continue
        
        # Store critical path info
        for task_id in critical_tasks:
            if task_id in self.tasks:
                # Mark as critical (would need to add this field)
                logger.debug(f"Task {task_id} is on critical path")
    
    def _update_ready_queue(self):
        """Update queue of tasks ready to execute."""
        self.ready_queue.clear()
        self.blocked_queue.clear()
        
        for task_id, state in self.task_states.items():
            if state in [TaskState.PENDING, TaskState.READY]:
                # Check if dependencies are satisfied
                dependencies = self.tasks[task_id].dependencies
                
                if all(self.task_states.get(dep_id) == TaskState.COMPLETED for dep_id in dependencies):
                    self.ready_queue.append(task_id)
                    self.task_states[task_id] = TaskState.READY
                    
                    # Clear blocking info
                    self.task_metrics[task_id].blocking_tasks.clear()
                else:
                    self.blocked_queue.add(task_id)
                    self.task_states[task_id] = TaskState.BLOCKED
                    
                    # Track what's blocking
                    blocking = [dep_id for dep_id in dependencies 
                               if self.task_states.get(dep_id) != TaskState.COMPLETED]
                    self.task_metrics[task_id].blocking_tasks = set(blocking)
        
        # Sort ready queue based on strategy
        self._sort_ready_queue()
    
    def _sort_ready_queue(self):
        """Sort ready queue based on scheduling strategy."""
        if self.strategy == SchedulingStrategy.PRIORITY:
            self.ready_queue = deque(sorted(
                self.ready_queue,
                key=lambda t: self._calculate_priority_score(t),
                reverse=True
            ))
        elif self.strategy == SchedulingStrategy.SHORTEST_FIRST:
            self.ready_queue = deque(sorted(
                self.ready_queue,
                key=lambda t: self._estimate_task_duration(t)
            ))
        elif self.strategy == SchedulingStrategy.CRITICAL_PATH:
            # Prioritize critical path tasks
            critical = []
            non_critical = []
            
            for task_id in self.ready_queue:
                # Check if on critical path (would need to track this)
                if self._is_on_critical_path(task_id):
                    critical.append(task_id)
                else:
                    non_critical.append(task_id)
            
            self.ready_queue = deque(critical + non_critical)
    
    def _calculate_priority_score(self, task_id: str) -> float:
        """Calculate priority score for a task."""
        task = self.tasks[task_id]
        score = 0.0
        
        # Base priority from task type
        type_priorities = {
            'setup': 100,
            'critical': 90,
            'backend': 70,
            'frontend': 60,
            'testing': 40,
            'documentation': 20
        }
        score += type_priorities.get(task.type, 50)
        
        # Boost for tasks blocking others
        blocked_count = sum(1 for t in self.blocked_queue 
                           if task_id in self.task_metrics[t].blocking_tasks)
        score += blocked_count * 10
        
        # Penalty for retries
        score -= self.task_metrics[task_id].retry_count * 5
        
        # Boost for critical path
        if self._is_on_critical_path(task_id):
            score += 50
        
        return score
    
    def _estimate_task_duration(self, task_id: str) -> float:
        """Estimate task execution duration."""
        task = self.tasks[task_id]
        
        # Use historical data if available
        type_perf = self.task_type_performance.get(task.type, {})
        if type_perf.get('samples', 0) > 0:
            return type_perf['avg_duration']
        
        # Use task's estimated hours if available
        if hasattr(task, 'estimated_hours') and task.estimated_hours:
            return task.estimated_hours * 3600  # Convert to seconds
        
        # Default estimates by type
        default_durations = {
            'setup': 300,      # 5 minutes
            'backend': 600,    # 10 minutes
            'frontend': 600,   # 10 minutes
            'database': 450,   # 7.5 minutes
            'testing': 300,    # 5 minutes
            'deployment': 900, # 15 minutes
            'general': 450     # 7.5 minutes
        }
        
        return default_durations.get(task.type, 450)
    
    def _is_on_critical_path(self, task_id: str) -> bool:
        """Check if task is on critical path."""
        # Simplified check - would need proper critical path analysis
        # For now, check if task has many dependents
        dependents = list(self.dependency_graph.successors(task_id))
        return len(dependents) >= 2
    
    def _update_task_type_performance(self, task_type: str, metrics: TaskMetrics, success: bool):
        """Update performance statistics for task type."""
        perf = self.task_type_performance[task_type]
        
        # Update average duration
        if metrics.execution_time > 0:
            if perf['samples'] == 0:
                perf['avg_duration'] = metrics.execution_time
            else:
                # Moving average
                perf['avg_duration'] = (
                    perf['avg_duration'] * perf['samples'] + metrics.execution_time
                ) / (perf['samples'] + 1)
        
        # Update success rate
        if perf['samples'] == 0:
            perf['success_rate'] = 1.0 if success else 0.0
        else:
            perf['success_rate'] = (
                perf['success_rate'] * perf['samples'] + (1.0 if success else 0.0)
            ) / (perf['samples'] + 1)
        
        perf['samples'] += 1
    
    def _schedule_by_priority(self) -> Optional[str]:
        """Schedule by priority score."""
        if not self.ready_queue:
            return None
        
        # Already sorted by priority
        return self.ready_queue.popleft()
    
    def _schedule_shortest_first(self) -> Optional[str]:
        """Schedule shortest estimated task first."""
        if not self.ready_queue:
            return None
        
        # Already sorted by duration
        return self.ready_queue.popleft()
    
    def _schedule_critical_path(self) -> Optional[str]:
        """Schedule critical path tasks first."""
        if not self.ready_queue:
            return None
        
        # Already sorted with critical path first
        return self.ready_queue.popleft()
    
    def _schedule_adaptive(self) -> Optional[str]:
        """Adaptive scheduling based on current state."""
        if not self.ready_queue:
            return None
        
        # Analyze current situation
        executing_count = len(self.executing_tasks)
        blocked_count = len(self.blocked_queue)
        
        # If many tasks are blocked, prioritize unblocking tasks
        if blocked_count > executing_count * 2:
            # Find task that unblocks most others
            best_task = None
            max_unblocks = 0
            
            for task_id in list(self.ready_queue):
                unblocks = sum(1 for t in self.blocked_queue 
                             if task_id in self.task_metrics[t].blocking_tasks)
                if unblocks > max_unblocks:
                    max_unblocks = unblocks
                    best_task = task_id
            
            if best_task:
                self.ready_queue.remove(best_task)
                return best_task
        
        # Otherwise use priority-based scheduling
        return self._schedule_by_priority()
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Detect bottlenecks
                self.detect_bottlenecks()
                
                # Log current state
                self.logger.log(
                    level=logging.INFO,
                    category=LogCategory.PERFORMANCE,
                    message="Scheduler state update",
                    metadata={
                        'ready': len(self.ready_queue),
                        'blocked': len(self.blocked_queue),
                        'executing': len(self.executing_tasks),
                        'bottlenecks': len(self.bottlenecks)
                    }
                )
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)


# Singleton instance
_adaptive_scheduler: Optional[AdaptiveTaskScheduler] = None


def get_adaptive_scheduler() -> AdaptiveTaskScheduler:
    """Get or create singleton scheduler instance."""
    global _adaptive_scheduler
    if _adaptive_scheduler is None:
        _adaptive_scheduler = AdaptiveTaskScheduler()
    return _adaptive_scheduler