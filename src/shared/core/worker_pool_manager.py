"""Worker Pool Manager for parallel CLI session management

This module provides efficient parallel execution of Code Claude CLI sessions
with dynamic worker management and load balancing.
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import multiprocessing
from collections import deque
import psutil

from shared.utils.config import Config
from shared.core.schema import Task
from shared.core.enhanced_logger import get_logger, LogCategory

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker states."""
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    TERMINATED = "terminated"


class TaskPriority(Enum):
    """Task execution priority."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class WorkerStats:
    """Statistics for a worker."""
    worker_id: str
    state: WorkerState
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_task_time: float = 0.0
    last_task_time: Optional[datetime] = None
    current_task: Optional[str] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


@dataclass 
class QueuedTask:
    """Task in the execution queue."""
    task: Task
    priority: TaskPriority
    workspace_path: str
    master_context: str
    task_specific_context: str = ""
    dependencies: Set[str] = field(default_factory=set)
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    queued_at: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other):
        """Compare tasks by priority for queue ordering."""
        return self.priority.value < other.priority.value


class WorkerPoolManager:
    """Manages a pool of Code Claude CLI workers for parallel execution."""
    
    def __init__(self, config: Config, min_workers: int = 2, max_workers: int = None):
        self.config = config
        self.min_workers = min_workers
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        
        # Worker management
        self.workers: Dict[str, Any] = {}  # worker_id -> ClaudeCliExecutor
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.worker_semaphore = asyncio.Semaphore(self.max_workers)
        
        # Task queue and tracking
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.executing_tasks: Dict[str, QueuedTask] = {}  # task_id -> QueuedTask
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Dict[str, str] = {}  # task_id -> error_message
        
        # Performance tracking
        self.performance_stats = {
            'total_tasks_queued': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'average_queue_time': 0.0,
            'average_execution_time': 0.0,
            'peak_workers': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Worker pool state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.worker_tasks: List[asyncio.Task] = []
        
        # Dynamic scaling parameters
        self.scale_check_interval = 30  # seconds
        self.scale_up_threshold = 0.8   # Queue utilization threshold
        self.scale_down_threshold = 0.3  # Queue utilization threshold
        self.last_scale_check = datetime.now()
        
        # Enhanced logger
        self.logger = get_logger(
            component="worker_pool_manager",
            session_id=f"pool_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    async def start(self):
        """Start the worker pool."""
        if self.is_running:
            logger.warning("Worker pool already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start minimum workers
        for i in range(self.min_workers):
            worker_id = f"worker_{i}"
            await self._spawn_worker(worker_id)
        
        # Start task dispatcher
        dispatcher_task = asyncio.create_task(self._task_dispatcher())
        self.worker_tasks.append(dispatcher_task)
        
        # Start performance monitor
        monitor_task = asyncio.create_task(self._performance_monitor())
        self.worker_tasks.append(monitor_task)
        
        # Start dynamic scaler
        scaler_task = asyncio.create_task(self._dynamic_scaler())
        self.worker_tasks.append(scaler_task)
        
        logger.info(f"Worker pool started with {self.min_workers} workers")
    
    async def stop(self):
        """Stop the worker pool gracefully."""
        if not self.is_running:
            return
        
        logger.info("Stopping worker pool...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Clean up workers
        for worker_id in list(self.workers.keys()):
            await self._terminate_worker(worker_id)
        
        logger.info("Worker pool stopped")
    
    async def submit_task(self, task: Task, workspace_path: str, 
                         master_context: str, task_specific_context: str = "",
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         dependencies: Set[str] = None,
                         callback: Callable = None) -> str:
        """Submit a task to the worker pool."""
        if not self.is_running:
            raise RuntimeError("Worker pool is not running")
        
        # Check if task is already completed or executing
        if task.id in self.completed_tasks:
            logger.info(f"Task {task.id} already completed")
            return task.id
        
        if task.id in self.executing_tasks:
            logger.info(f"Task {task.id} already executing")
            return task.id
        
        # Create queued task
        queued_task = QueuedTask(
            task=task,
            priority=priority,
            workspace_path=workspace_path,
            master_context=master_context,
            task_specific_context=task_specific_context,
            dependencies=dependencies or set(),
            callback=callback
        )
        
        # Add to queue
        await self.task_queue.put((priority.value, queued_task))
        self.performance_stats['total_tasks_queued'] += 1
        
        logger.info(f"Task {task.id} queued with priority {priority.name}")
        return task.id
    
    async def submit_batch(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Submit multiple tasks as a batch."""
        task_ids = []
        
        # Sort tasks by dependencies
        sorted_tasks = self._topological_sort_tasks(tasks)
        
        for task_info in sorted_tasks:
            task_id = await self.submit_task(
                task=task_info['task'],
                workspace_path=task_info['workspace_path'],
                master_context=task_info['master_context'],
                task_specific_context=task_info.get('task_specific_context', ''),
                priority=task_info.get('priority', TaskPriority.MEDIUM),
                dependencies=set(task_info.get('dependencies', [])),
                callback=task_info.get('callback')
            )
            task_ids.append(task_id)
        
        return task_ids
    
    async def wait_for_task(self, task_id: str, timeout: float = None) -> Dict[str, Any]:
        """Wait for a specific task to complete."""
        start_time = time.time()
        
        while True:
            if task_id in self.completed_tasks:
                return {'success': True, 'task_id': task_id}
            
            if task_id in self.failed_tasks:
                return {
                    'success': False,
                    'task_id': task_id,
                    'error': self.failed_tasks[task_id]
                }
            
            if timeout and (time.time() - start_time) > timeout:
                return {
                    'success': False,
                    'task_id': task_id,
                    'error': 'Timeout waiting for task completion'
                }
            
            await asyncio.sleep(0.5)
    
    async def wait_for_all(self, task_ids: List[str], timeout: float = None) -> Dict[str, Any]:
        """Wait for multiple tasks to complete."""
        results = await asyncio.gather(
            *[self.wait_for_task(task_id, timeout) for task_id in task_ids],
            return_exceptions=True
        )
        
        return {
            'completed': [r['task_id'] for r in results if isinstance(r, dict) and r.get('success')],
            'failed': [r['task_id'] for r in results if isinstance(r, dict) and not r.get('success')],
            'errors': [r for r in results if isinstance(r, Exception)]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the worker pool."""
        active_workers = sum(1 for w in self.worker_stats.values() if w.state == WorkerState.BUSY)
        idle_workers = sum(1 for w in self.worker_stats.values() if w.state == WorkerState.IDLE)
        
        return {
            'is_running': self.is_running,
            'total_workers': len(self.workers),
            'active_workers': active_workers,
            'idle_workers': idle_workers,
            'queued_tasks': self.task_queue.qsize(),
            'executing_tasks': len(self.executing_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'performance': self.performance_stats,
            'worker_stats': {
                worker_id: {
                    'state': stats.state.value,
                    'tasks_completed': stats.tasks_completed,
                    'tasks_failed': stats.tasks_failed,
                    'average_task_time': stats.average_task_time,
                    'current_task': stats.current_task
                }
                for worker_id, stats in self.worker_stats.items()
            }
        }
    
    # Private methods
    
    async def _spawn_worker(self, worker_id: str):
        """Spawn a new worker."""
        try:
            # Import here to avoid circular imports
            from cli.agents.claude_cli_executor import ClaudeCliExecutor
            
            # Create worker
            worker = ClaudeCliExecutor(self.config, worker_id)
            self.workers[worker_id] = worker
            self.worker_stats[worker_id] = WorkerStats(
                worker_id=worker_id,
                state=WorkerState.IDLE
            )
            
            # Start worker task
            worker_task = asyncio.create_task(self._worker_loop(worker_id))
            self.worker_tasks.append(worker_task)
            
            logger.info(f"Spawned worker {worker_id}")
            
        except Exception as e:
            logger.error(f"Failed to spawn worker {worker_id}: {e}")
    
    async def _terminate_worker(self, worker_id: str):
        """Terminate a worker."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            
        if worker_id in self.worker_stats:
            self.worker_stats[worker_id].state = WorkerState.TERMINATED
        
        logger.info(f"Terminated worker {worker_id}")
    
    async def _worker_loop(self, worker_id: str):
        """Main loop for a worker."""
        worker = self.workers[worker_id]
        stats = self.worker_stats[worker_id]
        
        while self.is_running:
            try:
                # Get task from queue (with timeout to check shutdown)
                try:
                    priority, queued_task = await asyncio.wait_for(
                        self.task_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check dependencies
                if not await self._check_dependencies(queued_task):
                    # Re-queue if dependencies not met
                    await self.task_queue.put((priority, queued_task))
                    await asyncio.sleep(0.5)
                    continue
                
                # Update stats
                stats.state = WorkerState.BUSY
                stats.current_task = queued_task.task.id
                queue_time = (datetime.now() - queued_task.queued_at).total_seconds()
                
                # Track executing task
                self.executing_tasks[queued_task.task.id] = queued_task
                
                # Execute task
                start_time = time.time()
                try:
                    result = await worker.execute_task_with_curated_context(
                        task=queued_task.task,
                        workspace_path=queued_task.workspace_path,
                        master_context=queued_task.master_context,
                        task_specific_context=queued_task.task_specific_context
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Update stats
                    stats.tasks_completed += 1
                    stats.total_execution_time += execution_time
                    stats.average_task_time = stats.total_execution_time / stats.tasks_completed
                    stats.last_task_time = datetime.now()
                    
                    self.performance_stats['total_tasks_completed'] += 1
                    self._update_average_times(queue_time, execution_time)
                    
                    # Mark as completed
                    self.completed_tasks.add(queued_task.task.id)
                    
                    # Call callback if provided
                    if queued_task.callback:
                        await queued_task.callback(result)
                    
                    logger.info(f"Worker {worker_id} completed task {queued_task.task.id} in {execution_time:.2f}s")
                    
                except Exception as e:
                    # Handle task failure
                    stats.tasks_failed += 1
                    self.performance_stats['total_tasks_failed'] += 1
                    
                    error_msg = str(e)
                    logger.error(f"Worker {worker_id} failed task {queued_task.task.id}: {error_msg}")
                    
                    # Retry logic
                    if queued_task.retry_count < queued_task.max_retries:
                        queued_task.retry_count += 1
                        await self.task_queue.put((priority, queued_task))
                        logger.info(f"Re-queuing task {queued_task.task.id} (retry {queued_task.retry_count})")
                    else:
                        self.failed_tasks[queued_task.task.id] = error_msg
                
                finally:
                    # Clean up
                    if queued_task.task.id in self.executing_tasks:
                        del self.executing_tasks[queued_task.task.id]
                    
                    stats.state = WorkerState.IDLE
                    stats.current_task = None
                
            except Exception as e:
                logger.error(f"Worker {worker_id} loop error: {e}")
                stats.state = WorkerState.FAILED
                await asyncio.sleep(1)
    
    async def _task_dispatcher(self):
        """Dispatch tasks to workers based on availability and priority."""
        while self.is_running:
            try:
                # Check for idle workers and pending tasks
                idle_workers = [
                    w_id for w_id, stats in self.worker_stats.items()
                    if stats.state == WorkerState.IDLE
                ]
                
                if idle_workers and not self.task_queue.empty():
                    # Task assignment is handled by worker_loop
                    pass
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Task dispatcher error: {e}")
                await asyncio.sleep(1)
    
    async def _performance_monitor(self):
        """Monitor worker performance and system resources."""
        while self.is_running:
            try:
                # Update worker resource usage
                for worker_id, stats in self.worker_stats.items():
                    if stats.state == WorkerState.BUSY:
                        # Get process resource usage
                        try:
                            process = psutil.Process(os.getpid())
                            stats.cpu_usage = process.cpu_percent()
                            stats.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                        except:
                            pass
                
                # Update peak workers
                active_workers = sum(1 for s in self.worker_stats.values() if s.state == WorkerState.BUSY)
                self.performance_stats['peak_workers'] = max(
                    self.performance_stats['peak_workers'],
                    active_workers
                )
                
                # Log performance metrics
                if active_workers > 0:
                    self.logger.log(
                        level=logging.INFO,
                        category=LogCategory.PERFORMANCE,
                        message=f"Worker pool: {active_workers} active, {self.task_queue.qsize()} queued",
                        metadata={
                            'active_workers': active_workers,
                            'queued_tasks': self.task_queue.qsize(),
                            'completed_tasks': len(self.completed_tasks),
                            'average_execution_time': self.performance_stats['average_execution_time']
                        }
                    )
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _dynamic_scaler(self):
        """Dynamically scale workers based on workload."""
        while self.is_running:
            try:
                now = datetime.now()
                if (now - self.last_scale_check).seconds < self.scale_check_interval:
                    await asyncio.sleep(1)
                    continue
                
                self.last_scale_check = now
                
                # Calculate utilization
                queue_size = self.task_queue.qsize()
                active_workers = sum(1 for s in self.worker_stats.values() if s.state == WorkerState.BUSY)
                total_workers = len(self.workers)
                
                if total_workers == 0:
                    continue
                
                utilization = (active_workers + min(queue_size, total_workers)) / total_workers
                
                # Scale up if needed
                if utilization > self.scale_up_threshold and total_workers < self.max_workers:
                    new_worker_id = f"worker_{total_workers}"
                    await self._spawn_worker(new_worker_id)
                    logger.info(f"Scaled up to {total_workers + 1} workers (utilization: {utilization:.2f})")
                
                # Scale down if needed
                elif utilization < self.scale_down_threshold and total_workers > self.min_workers:
                    # Find idle worker to terminate
                    for worker_id, stats in self.worker_stats.items():
                        if stats.state == WorkerState.IDLE:
                            await self._terminate_worker(worker_id)
                            logger.info(f"Scaled down to {total_workers - 1} workers (utilization: {utilization:.2f})")
                            break
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Dynamic scaler error: {e}")
                await asyncio.sleep(self.scale_check_interval)
    
    async def _check_dependencies(self, queued_task: QueuedTask) -> bool:
        """Check if all dependencies for a task are satisfied."""
        if not queued_task.dependencies:
            return True
        
        for dep_id in queued_task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    def _topological_sort_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort tasks based on dependencies."""
        # Build dependency graph
        graph = {}
        in_degree = {}
        task_map = {}
        
        for task_info in tasks:
            task_id = task_info['task'].id
            task_map[task_id] = task_info
            graph[task_id] = set(task_info.get('dependencies', []))
            in_degree[task_id] = 0
        
        # Calculate in-degrees
        for task_id, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        sorted_tasks = []
        
        while queue:
            task_id = queue.popleft()
            sorted_tasks.append(task_map[task_id])
            
            for dep in graph[task_id]:
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
        
        # Add any remaining tasks (cycles)
        for task_info in tasks:
            if task_info not in sorted_tasks:
                sorted_tasks.append(task_info)
        
        return sorted_tasks
    
    def _update_average_times(self, queue_time: float, execution_time: float):
        """Update average queue and execution times."""
        total_completed = self.performance_stats['total_tasks_completed']
        
        if total_completed == 1:
            self.performance_stats['average_queue_time'] = queue_time
            self.performance_stats['average_execution_time'] = execution_time
        else:
            # Moving average
            alpha = 0.1  # Smoothing factor
            self.performance_stats['average_queue_time'] = (
                alpha * queue_time + 
                (1 - alpha) * self.performance_stats['average_queue_time']
            )
            self.performance_stats['average_execution_time'] = (
                alpha * execution_time + 
                (1 - alpha) * self.performance_stats['average_execution_time']
            )


# Singleton instance
_worker_pool_manager: Optional[WorkerPoolManager] = None


def get_worker_pool_manager(config: Config) -> WorkerPoolManager:
    """Get or create the singleton worker pool manager."""
    global _worker_pool_manager
    if _worker_pool_manager is None:
        _worker_pool_manager = WorkerPoolManager(config)
    return _worker_pool_manager