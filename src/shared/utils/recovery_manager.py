"""Partial Execution Recovery System for VIBE."""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import pickle
import hashlib

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RECOVERED = "recovered"


@dataclass
class ExecutionCheckpoint:
    """Checkpoint for execution state."""
    checkpoint_id: str
    timestamp: datetime
    project_id: str
    task_id: str
    status: TaskStatus
    context: Dict[str, Any]
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    dependencies_completed: Set[str] = field(default_factory=set)
    error_info: Optional[Dict[str, Any]] = None
    recovery_attempts: int = 0


class RecoveryManager:
    """Manages partial execution recovery and state persistence."""
    
    def __init__(self, project_id: str, recovery_dir: str = ".vibe_recovery"):
        self.project_id = project_id
        self.recovery_dir = Path(recovery_dir)
        self.recovery_dir.mkdir(exist_ok=True)
        
        self.checkpoints: Dict[str, ExecutionCheckpoint] = {}
        self.task_dependencies: Dict[str, Set[str]] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        self._load_recovery_state()
    
    def create_checkpoint(self, task_id: str, status: TaskStatus, context: Dict[str, Any] = None) -> str:
        """Create a checkpoint for current execution state."""
        checkpoint_id = self._generate_checkpoint_id(task_id)
        
        checkpoint = ExecutionCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            project_id=self.project_id,
            task_id=task_id,
            status=status,
            context=context or {},
            files_created=self._get_newly_created_files(),
            files_modified=self._get_recently_modified_files(),
            dependencies_completed=self._get_completed_dependencies(task_id)
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        self._save_checkpoint(checkpoint)
        
        logger.info(f"Created checkpoint {checkpoint_id} for task {task_id} with status {status.value}")
        return checkpoint_id
    
    def update_checkpoint(self, checkpoint_id: str, status: TaskStatus = None, 
                         context: Dict[str, Any] = None, error_info: Dict[str, Any] = None):
        """Update an existing checkpoint."""
        if checkpoint_id not in self.checkpoints:
            logger.warning(f"Checkpoint {checkpoint_id} not found")
            return
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        if status:
            checkpoint.status = status
        if context:
            checkpoint.context.update(context)
        if error_info:
            checkpoint.error_info = error_info
        
        checkpoint.timestamp = datetime.now()
        self._save_checkpoint(checkpoint)
        
        logger.info(f"Updated checkpoint {checkpoint_id} with status {status.value if status else 'unchanged'}")
    
    def can_recover_from_failure(self, failed_task_id: str) -> bool:
        """Check if we can recover from a task failure."""
        # Find the most recent checkpoint for this task
        task_checkpoints = [
            cp for cp in self.checkpoints.values() 
            if cp.task_id == failed_task_id
        ]
        
        if not task_checkpoints:
            return False
        
        latest_checkpoint = max(task_checkpoints, key=lambda x: x.timestamp)
        
        # Check if recovery has been attempted too many times
        if latest_checkpoint.recovery_attempts >= 3:
            logger.warning(f"Too many recovery attempts for task {failed_task_id}")
            return False
        
        # Check if dependencies are still satisfied
        return self._verify_dependencies(failed_task_id)
    
    async def recover_from_failure(self, failed_task_id: str) -> Optional[ExecutionCheckpoint]:
        """Attempt to recover from a task failure."""
        if not self.can_recover_from_failure(failed_task_id):
            return None
        
        # Find the best recovery point
        recovery_checkpoint = self._find_best_recovery_point(failed_task_id)
        if not recovery_checkpoint:
            return None
        
        logger.info(f"Attempting recovery for task {failed_task_id} from checkpoint {recovery_checkpoint.checkpoint_id}")
        
        # Clean up partial work
        await self._cleanup_partial_work(recovery_checkpoint)
        
        # Reset task state
        recovery_checkpoint.status = TaskStatus.PENDING
        recovery_checkpoint.recovery_attempts += 1
        recovery_checkpoint.error_info = None
        
        self._save_checkpoint(recovery_checkpoint)
        
        logger.info(f"Recovery prepared for task {failed_task_id}")
        return recovery_checkpoint
    
    def get_execution_resume_point(self) -> Optional[str]:
        """Get the best point to resume execution from."""
        # Find all pending tasks whose dependencies are completed
        resumable_tasks = []
        
        for checkpoint in self.checkpoints.values():
            if checkpoint.status == TaskStatus.PENDING:
                if self._are_dependencies_completed(checkpoint.task_id):
                    resumable_tasks.append(checkpoint)
        
        if not resumable_tasks:
            # Check for failed tasks that can be recovered
            failed_tasks = [
                cp for cp in self.checkpoints.values() 
                if cp.status == TaskStatus.FAILED and self.can_recover_from_failure(cp.task_id)
            ]
            
            if failed_tasks:
                # Return the earliest failed task for recovery
                earliest_failed = min(failed_tasks, key=lambda x: x.timestamp)
                return earliest_failed.task_id
            
            return None
        
        # Return the earliest pending task
        earliest_pending = min(resumable_tasks, key=lambda x: x.timestamp)
        return earliest_pending.task_id
    
    def mark_task_completed(self, task_id: str, output_files: List[str] = None):
        """Mark a task as completed and update dependencies."""
        # Find the most recent checkpoint for this task
        task_checkpoints = [
            cp for cp in self.checkpoints.values() 
            if cp.task_id == task_id
        ]
        
        if task_checkpoints:
            latest_checkpoint = max(task_checkpoints, key=lambda x: x.timestamp)
            latest_checkpoint.status = TaskStatus.COMPLETED
            if output_files:
                latest_checkpoint.files_created.extend(output_files)
            self._save_checkpoint(latest_checkpoint)
        
        # Update execution log
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "action": "completed",
            "output_files": output_files or []
        })
        
        self._save_execution_log()
        logger.info(f"Task {task_id} marked as completed")
    
    def get_project_progress(self) -> Dict[str, Any]:
        """Get overall project progress summary."""
        total_tasks = len(set(cp.task_id for cp in self.checkpoints.values()))
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = 0
        
        latest_checkpoints = self._get_latest_checkpoints_per_task()
        
        for checkpoint in latest_checkpoints.values():
            status_counts[checkpoint.status.value] += 1
        
        completed_count = status_counts[TaskStatus.COMPLETED.value]
        failed_count = status_counts[TaskStatus.FAILED.value]
        
        return {
            "project_id": self.project_id,
            "total_tasks": total_tasks,
            "completed": completed_count,
            "failed": failed_count,
            "progress_percentage": (completed_count / total_tasks * 100) if total_tasks > 0 else 0,
            "status_breakdown": status_counts,
            "can_resume": self.get_execution_resume_point() is not None,
            "next_resumable_task": self.get_execution_resume_point()
        }
    
    def _generate_checkpoint_id(self, task_id: str) -> str:
        """Generate a unique checkpoint ID."""
        timestamp = datetime.now().isoformat()
        content = f"{self.project_id}_{task_id}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_newly_created_files(self) -> List[str]:
        """Get list of files created in current execution."""
        # This is a simplified implementation
        # In a real scenario, you'd track file system changes
        return []
    
    def _get_recently_modified_files(self) -> List[str]:
        """Get list of files modified in current execution.""" 
        # This is a simplified implementation
        # In a real scenario, you'd track file system changes
        return []
    
    def _get_completed_dependencies(self, task_id: str) -> Set[str]:
        """Get set of completed dependencies for a task."""
        dependencies = self.task_dependencies.get(task_id, set())
        completed_deps = set()
        
        latest_checkpoints = self._get_latest_checkpoints_per_task()
        
        for dep_id in dependencies:
            if (dep_id in latest_checkpoints and 
                latest_checkpoints[dep_id].status == TaskStatus.COMPLETED):
                completed_deps.add(dep_id)
        
        return completed_deps
    
    def _get_latest_checkpoints_per_task(self) -> Dict[str, ExecutionCheckpoint]:
        """Get the latest checkpoint for each task."""
        latest_checkpoints = {}
        
        for checkpoint in self.checkpoints.values():
            task_id = checkpoint.task_id
            if (task_id not in latest_checkpoints or 
                checkpoint.timestamp > latest_checkpoints[task_id].timestamp):
                latest_checkpoints[task_id] = checkpoint
        
        return latest_checkpoints
    
    def _verify_dependencies(self, task_id: str) -> bool:
        """Verify that all dependencies for a task are still satisfied."""
        dependencies = self.task_dependencies.get(task_id, set())
        latest_checkpoints = self._get_latest_checkpoints_per_task()
        
        for dep_id in dependencies:
            if (dep_id not in latest_checkpoints or 
                latest_checkpoints[dep_id].status != TaskStatus.COMPLETED):
                return False
        
        return True
    
    def _are_dependencies_completed(self, task_id: str) -> bool:
        """Check if all dependencies for a task are completed."""
        return self._verify_dependencies(task_id)
    
    def _find_best_recovery_point(self, task_id: str) -> Optional[ExecutionCheckpoint]:
        """Find the best checkpoint to recover from."""
        task_checkpoints = [
            cp for cp in self.checkpoints.values() 
            if cp.task_id == task_id and cp.status != TaskStatus.FAILED
        ]
        
        if not task_checkpoints:
            return None
        
        # Return the latest non-failed checkpoint
        return max(task_checkpoints, key=lambda x: x.timestamp)
    
    async def _cleanup_partial_work(self, checkpoint: ExecutionCheckpoint):
        """Clean up any partial work from failed execution."""
        # Remove any partially created files
        for file_path in checkpoint.files_created:
            try:
                path = Path(file_path)
                if path.exists() and path.stat().st_size == 0:  # Empty file
                    path.unlink()
                    logger.info(f"Removed empty file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove file {file_path}: {e}")
    
    def _save_checkpoint(self, checkpoint: ExecutionCheckpoint):
        """Save checkpoint to disk."""
        checkpoint_file = self.recovery_dir / f"checkpoint_{checkpoint.checkpoint_id}.json"
        
        checkpoint_data = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "timestamp": checkpoint.timestamp.isoformat(),
            "project_id": checkpoint.project_id,
            "task_id": checkpoint.task_id,
            "status": checkpoint.status.value,
            "context": checkpoint.context,
            "files_created": checkpoint.files_created,
            "files_modified": checkpoint.files_modified,
            "dependencies_completed": list(checkpoint.dependencies_completed),
            "error_info": checkpoint.error_info,
            "recovery_attempts": checkpoint.recovery_attempts
        }
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")
    
    def _load_recovery_state(self):
        """Load existing recovery state from disk."""
        try:
            checkpoint_files = list(self.recovery_dir.glob("checkpoint_*.json"))
            
            for checkpoint_file in checkpoint_files:
                try:
                    with open(checkpoint_file, 'r') as f:
                        data = json.load(f)
                    
                    checkpoint = ExecutionCheckpoint(
                        checkpoint_id=data["checkpoint_id"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        project_id=data["project_id"],
                        task_id=data["task_id"],
                        status=TaskStatus(data["status"]),
                        context=data["context"],
                        files_created=data.get("files_created", []),
                        files_modified=data.get("files_modified", []),
                        dependencies_completed=set(data.get("dependencies_completed", [])),
                        error_info=data.get("error_info"),
                        recovery_attempts=data.get("recovery_attempts", 0)
                    )
                    
                    self.checkpoints[checkpoint.checkpoint_id] = checkpoint
                    
                except Exception as e:
                    logger.warning(f"Could not load checkpoint from {checkpoint_file}: {e}")
            
            logger.info(f"Loaded {len(self.checkpoints)} checkpoints for project {self.project_id}")
            
        except Exception as e:
            logger.warning(f"Could not load recovery state: {e}")
    
    def _save_execution_log(self):
        """Save execution log to disk."""
        log_file = self.recovery_dir / f"execution_log_{self.project_id}.json"
        
        try:
            with open(log_file, 'w') as f:
                json.dump(self.execution_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save execution log: {e}")


# Global recovery manager instance
_recovery_managers: Dict[str, RecoveryManager] = {}


def get_recovery_manager(project_id: str) -> RecoveryManager:
    """Get or create a recovery manager for a project."""
    if project_id not in _recovery_managers:
        _recovery_managers[project_id] = RecoveryManager(project_id)
    return _recovery_managers[project_id]