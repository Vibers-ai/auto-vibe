"""Parallel Task Execution Engine for VIBE."""

import asyncio
import logging
from typing import Dict, Any, List, Set, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx

from shared.core.schema import Task, TasksPlan
from shared.utils.config import Config
from shared.core.feedback_loop import FeedbackLoop

logger = logging.getLogger(__name__)


class TaskGroup(Enum):
    """Task grouping for parallel execution."""
    SETUP = "setup"
    DATABASE = "database" 
    BACKEND = "backend"
    FRONTEND = "frontend"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    GENERAL = "general"


@dataclass
class ParallelExecutionResult:
    """Result from parallel task execution."""
    task_id: str
    success: bool
    result: Dict[str, Any]
    execution_time: float
    group: TaskGroup
    thread_id: str
    error: Optional[str] = None


@dataclass
class ExecutionStats:
    """Statistics for parallel execution."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    parallel_tasks_executed: int = 0
    total_execution_time: float = 0.0
    max_parallel_degree: int = 0
    groups_executed_parallel: Set[TaskGroup] = field(default_factory=set)


class DependencyGraphManager:
    """Manages task dependency graph for parallel execution."""
    
    def __init__(self, tasks: List[Task]):
        self.tasks = {task.id: task for task in tasks}
        self.graph = self._build_dependency_graph(tasks)
        self.execution_groups = self._analyze_execution_groups()
    
    def _build_dependency_graph(self, tasks: List[Task]) -> nx.DiGraph:
        """Build NetworkX directed graph from task dependencies."""
        graph = nx.DiGraph()
        
        # Add all tasks as nodes
        for task in tasks:
            graph.add_node(task.id, task=task, group=TaskGroup(task.type))
        
        # Add dependency edges
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    graph.add_edge(dep_id, task.id)
        
        # Validate no cycles
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            raise ValueError(f"Dependency cycles detected: {cycles}")
        
        return graph
    
    def _analyze_execution_groups(self) -> Dict[str, Set[str]]:
        """Analyze which tasks can be grouped for parallel execution."""
        groups = {}
        
        # Group by task type and dependency level
        for task_id in self.graph.nodes():
            task = self.tasks[task_id]
            group_key = f"{task.type}_level_{self._get_dependency_level(task_id)}"
            
            if group_key not in groups:
                groups[group_key] = set()
            groups[group_key].add(task_id)
        
        return groups
    
    def _get_dependency_level(self, task_id: str) -> int:
        """Get the dependency level (depth) of a task."""
        if not list(self.graph.predecessors(task_id)):
            return 0
        
        max_pred_level = 0
        for pred in self.graph.predecessors(task_id):
            max_pred_level = max(max_pred_level, self._get_dependency_level(pred))
        
        return max_pred_level + 1
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready_tasks = []
        
        for task_id in self.graph.nodes():
            if task_id in completed_tasks:
                continue
            
            # Check if all dependencies are completed
            dependencies = set(self.graph.predecessors(task_id))
            if dependencies.issubset(completed_tasks):
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def get_parallel_batches(self) -> List[List[str]]:
        """Get batches of tasks that can be executed in parallel."""
        completed = set()
        batches = []
        
        while len(completed) < len(self.tasks):
            ready_tasks = self.get_ready_tasks(completed)
            
            if not ready_tasks:
                # This shouldn't happen with a valid DAG
                remaining = set(self.tasks.keys()) - completed
                raise RuntimeError(f"Deadlock detected. Remaining tasks: {remaining}")
            
            # Group ready tasks by type for better parallelization
            batch_groups = self._group_tasks_for_batch(ready_tasks)
            
            for batch in batch_groups:
                if batch:
                    batches.append(batch)
                    completed.update(batch)
        
        return batches
    
    def _group_tasks_for_batch(self, ready_tasks: List[str]) -> List[List[str]]:
        """Group ready tasks into optimal parallel batches."""
        task_groups = {}
        
        # Group by task type
        for task_id in ready_tasks:
            task = self.tasks[task_id]
            task_type = task.type
            
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(task_id)
        
        # Create batches - each type can run in parallel
        batches = []
        
        # Setup tasks must run first
        if 'setup' in task_groups:
            batches.append(task_groups.pop('setup'))
        
        # Other tasks can run in parallel by type
        for task_type, task_list in task_groups.items():
            batches.append(task_list)
        
        return batches


class CodeClaudePool:
    """Pool of Code Claude instances for parallel execution."""
    
    def __init__(self, config: Config, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers
        self.active_workers = {}
        self.available_workers = asyncio.Queue(maxsize=max_workers)
        self.worker_stats = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the worker pool."""
        if self._initialized:
            return
        
        logger.info(f"Initializing Code Claude pool with {self.max_workers} workers")
        
        # Import here to avoid circular imports
        from cli.agents.claude_cli_executor import ClaudeCliExecutor
        
        for i in range(self.max_workers):
            worker_id = f"claude_worker_{i}"
            worker = ClaudeCliExecutor(
                config=self.config,
                worker_id=worker_id
            )
            
            await self.available_workers.put({
                'id': worker_id,
                'executor': worker,
                'created_at': datetime.now(),
                'task_count': 0
            })
            
            self.worker_stats[worker_id] = {
                'tasks_completed': 0,
                'total_execution_time': 0.0,
                'errors': 0
            }
        
        self._initialized = True
        logger.info("Code Claude pool initialization completed")
    
    async def acquire_worker(self):
        """Acquire a worker from the pool."""
        if not self._initialized:
            await self.initialize()
        
        worker_info = await self.available_workers.get()
        worker_id = worker_info['id']
        
        self.active_workers[worker_id] = worker_info
        logger.debug(f"Worker {worker_id} acquired")
        
        return worker_info
    
    async def release_worker(self, worker_info: Dict[str, Any]):
        """Release a worker back to the pool."""
        worker_id = worker_info['id']
        
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
        
        await self.available_workers.put(worker_info)
        logger.debug(f"Worker {worker_id} released")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'max_workers': self.max_workers,
            'active_workers': len(self.active_workers),
            'available_workers': self.available_workers.qsize(),
            'worker_stats': self.worker_stats
        }


class ParallelTaskExecutor:
    """Enhanced task executor with parallel processing capabilities."""
    
    def __init__(self, config: Config, max_parallel_tasks: int = 4):
        self.config = config
        self.max_parallel_tasks = max_parallel_tasks
        self.claude_pool = CodeClaudePool(config, max_parallel_tasks)
        self.stats = ExecutionStats()
        self.feedback_loop = None  # Will be initialized with workspace path
        
        # Import Master Claude supervisor
        from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
        self.master_supervisor = MasterClaudeCliSupervisor(config)
    
    async def execute_tasks_parallel(
        self, 
        tasks_plan: TasksPlan, 
        workspace_path: str
    ) -> Tuple[bool, ExecutionStats]:
        """Execute tasks with intelligent parallel processing."""
        
        logger.info(f"Starting parallel execution of {len(tasks_plan.tasks)} tasks")
        
        # Initialize feedback loop
        self.feedback_loop = FeedbackLoop(workspace_path)
        
        # Initialize statistics
        self.stats.total_tasks = len(tasks_plan.tasks)
        start_time = datetime.now()
        
        # Build dependency graph
        dep_manager = DependencyGraphManager(tasks_plan.tasks)
        parallel_batches = dep_manager.get_parallel_batches()
        
        logger.info(f"Organized into {len(parallel_batches)} parallel batches")
        
        # Initialize Claude pool
        await self.claude_pool.initialize()
        
        completed_tasks = set()
        overall_success = True
        
        # Execute each batch
        for batch_idx, batch in enumerate(parallel_batches):
            logger.info(f"Executing batch {batch_idx + 1}/{len(parallel_batches)} with {len(batch)} tasks")
            
            # Update max parallel degree
            self.stats.max_parallel_degree = max(self.stats.max_parallel_degree, len(batch))
            
            # Execute batch in parallel
            batch_results = await self._execute_batch_parallel(
                batch, tasks_plan, workspace_path, dep_manager
            )
            
            # Process results
            batch_success = True
            for result in batch_results:
                if result.success:
                    completed_tasks.add(result.task_id)
                    self.stats.completed_tasks += 1
                    self.stats.groups_executed_parallel.add(result.group)
                    
                    if len(batch) > 1:  # Only count if actually parallel
                        self.stats.parallel_tasks_executed += 1
                else:
                    self.stats.failed_tasks += 1
                    batch_success = False
                    overall_success = False
                    logger.error(f"Task {result.task_id} failed: {result.error}")
            
            # Stop if batch failed and we have critical failures
            if not batch_success and self._has_critical_failures(batch_results):
                logger.error("Critical failures detected, stopping execution")
                break
        
        # Calculate final statistics
        end_time = datetime.now()
        self.stats.total_execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Parallel execution completed. Success: {overall_success}")
        logger.info(f"Stats: {self.stats.completed_tasks}/{self.stats.total_tasks} completed, "
                   f"{self.stats.parallel_tasks_executed} executed in parallel")
        
        # Log feedback summary if available
        if self.feedback_loop:
            summary = self.feedback_loop.get_execution_summary()
            logger.info(f"Feedback Summary:")
            logger.info(f"  - Success rate: {summary['success_rate']:.1%}")
            for pattern in summary['common_patterns']:
                logger.info(f"  - {pattern}")
            for recommendation in summary['recommendations']:
                logger.info(f"  - Recommendation: {recommendation}")
        
        return overall_success, self.stats
    
    async def _execute_batch_parallel(
        self,
        batch: List[str],
        tasks_plan: TasksPlan,
        workspace_path: str,
        dep_manager: DependencyGraphManager
    ) -> List[ParallelExecutionResult]:
        """Execute a batch of tasks in parallel."""
        
        if len(batch) == 1:
            # Single task - execute directly
            task_id = batch[0]
            task = dep_manager.tasks[task_id]
            result = await self._execute_single_task(task, tasks_plan, workspace_path)
            return [result]
        
        # Multiple tasks - execute in parallel
        semaphore = asyncio.Semaphore(min(len(batch), self.max_parallel_tasks))
        
        async def execute_with_semaphore(task_id: str):
            async with semaphore:
                task = dep_manager.tasks[task_id]
                return await self._execute_single_task(task, tasks_plan, workspace_path)
        
        # Execute all tasks in batch concurrently
        tasks_coroutines = [execute_with_semaphore(task_id) for task_id in batch]
        results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_id = batch[i]
                task = dep_manager.tasks[task_id]
                error_result = ParallelExecutionResult(
                    task_id=task_id,
                    success=False,
                    result={},
                    execution_time=0.0,
                    group=TaskGroup(task.type),
                    thread_id="error",
                    error=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_single_task(
        self,
        task: Task,
        tasks_plan: TasksPlan,
        workspace_path: str
    ) -> ParallelExecutionResult:
        """Execute a single task with worker pool."""
        
        start_time = datetime.now()
        worker_info = None
        
        try:
            # Acquire worker from pool
            worker_info = await self.claude_pool.acquire_worker()
            worker_id = worker_info['id']
            
            logger.info(f"Executing task {task.id} with worker {worker_id}")
            
            # Apply feedback corrections to task files if available
            if self.feedback_loop:
                original_files = task.files_to_create_or_modify.copy()
                corrected_files = []
                for file_path in original_files:
                    suggestions = self.feedback_loop.suggest_file_correction(file_path)
                    # Add all suggestions to increase chances of finding the right file
                    corrected_files.extend(suggestions)
                
                # Update task with unique corrected files
                # Note: This modifies the task object, but it's intentional as we want the 
                # corrected files to be used for verification later
                task.files_to_create_or_modify = list(dict.fromkeys(corrected_files))
                if len(original_files) != len(task.files_to_create_or_modify):
                    logger.info(f"Applied feedback corrections for task {task.id}: {len(original_files)} -> {len(task.files_to_create_or_modify)} files")
            
            # Execute task with Master Claude supervision
            result = await self.master_supervisor.execute_task_with_cli_supervision(
                task, workspace_path, tasks_plan
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update worker stats
            self.claude_pool.worker_stats[worker_id]['tasks_completed'] += 1
            self.claude_pool.worker_stats[worker_id]['total_execution_time'] += execution_time
            
            # Record feedback if available
            success = result.get('overall_success', result.get('success', False))
            if self.feedback_loop and 'file_verification' in result:
                actual_files = []
                file_verification = result.get('file_verification', {})
                file_details = file_verification.get('file_details', {}) if isinstance(file_verification, dict) else {}
                
                for file_path, details in file_details.items():
                    if isinstance(details, dict) and details.get('exists', False):
                        actual_files.append(details.get('full_path', file_path))
                
                # Get tech stack info from result or master assessment
                tech_stack = None
                if 'master_assessment' in result:
                    tech_stack = result['master_assessment'].get('tech_stack')
                elif 'technology_stack' in result:
                    tech_stack = result['technology_stack']
                
                self.feedback_loop.record_task_execution(
                    task_id=task.id,
                    expected_files=task.files_to_create_or_modify,
                    actual_files=actual_files,
                    success=success,
                    tech_stack=tech_stack
                )
            
            return ParallelExecutionResult(
                task_id=task.id,
                success=success,
                result=result,
                execution_time=execution_time,
                group=TaskGroup(task.type),
                thread_id=worker_id
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if worker_info:
                worker_id = worker_info['id']
                self.claude_pool.worker_stats[worker_id]['errors'] += 1
            
            logger.error(f"Task {task.id} execution failed: {e}")
            
            return ParallelExecutionResult(
                task_id=task.id,
                success=False,
                result={},
                execution_time=execution_time,
                group=TaskGroup(task.type),
                thread_id=worker_info['id'] if worker_info else "unknown",
                error=str(e)
            )
        
        finally:
            # Always release worker back to pool
            if worker_info:
                await self.claude_pool.release_worker(worker_info)
    
    def _has_critical_failures(self, batch_results: List[ParallelExecutionResult]) -> bool:
        """Check if batch has critical failures that should stop execution."""
        
        critical_task_types = {'setup', 'database'}
        
        for result in batch_results:
            if not result.success and result.group.value in critical_task_types:
                return True
        
        return False
    
    def get_execution_report(self) -> Dict[str, Any]:
        """Generate detailed execution report."""
        
        pool_stats = self.claude_pool.get_pool_stats()
        
        efficiency_score = 0
        if self.stats.total_tasks > 0:
            efficiency_score = (
                (self.stats.parallel_tasks_executed / self.stats.total_tasks) * 100
            )
        
        return {
            'execution_summary': {
                'total_tasks': self.stats.total_tasks,
                'completed_tasks': self.stats.completed_tasks,
                'failed_tasks': self.stats.failed_tasks,
                'success_rate': (self.stats.completed_tasks / self.stats.total_tasks * 100) if self.stats.total_tasks > 0 else 0,
                'total_execution_time': self.stats.total_execution_time
            },
            'parallelization_metrics': {
                'parallel_tasks_executed': self.stats.parallel_tasks_executed,
                'max_parallel_degree': self.stats.max_parallel_degree,
                'parallelization_efficiency': efficiency_score,
                'groups_executed_parallel': list(self.stats.groups_executed_parallel)
            },
            'worker_pool_stats': pool_stats,
            'performance_insights': self._generate_performance_insights()
        }
    
    def _generate_performance_insights(self) -> List[str]:
        """Generate performance insights and recommendations."""
        insights = []
        
        if self.stats.parallel_tasks_executed > 0:
            insights.append(f"Successfully executed {self.stats.parallel_tasks_executed} tasks in parallel")
        
        if self.stats.max_parallel_degree > 1:
            insights.append(f"Achieved maximum parallelization of {self.stats.max_parallel_degree} concurrent tasks")
        
        if len(self.stats.groups_executed_parallel) > 1:
            groups = ', '.join([g.value for g in self.stats.groups_executed_parallel])
            insights.append(f"Parallelized task groups: {groups}")
        
        # Pool efficiency
        pool_stats = self.claude_pool.get_pool_stats()
        if pool_stats['max_workers'] > 1:
            avg_tasks_per_worker = sum(
                stats['tasks_completed'] 
                for stats in pool_stats['worker_stats'].values()
            ) / pool_stats['max_workers']
            insights.append(f"Average tasks per worker: {avg_tasks_per_worker:.1f}")
        
        return insights