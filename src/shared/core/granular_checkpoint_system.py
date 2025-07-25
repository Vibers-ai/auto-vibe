"""Granular Checkpoint System for VIBE

This module provides fine-grained checkpoint management for task execution,
enabling precise recovery from failures at the file, function, or even line level.
"""

import json
import logging
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pickle
import shutil
import difflib

logger = logging.getLogger(__name__)


class CheckpointLevel(Enum):
    """Granularity levels for checkpoints."""
    TASK = "task"           # Entire task
    FILE = "file"           # Individual file
    FUNCTION = "function"   # Function/class level
    BLOCK = "block"         # Code block level


class CheckpointState(Enum):
    """State of a checkpoint."""
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class FileSnapshot:
    """Snapshot of a file's state."""
    file_path: str
    content_hash: str
    content: str
    timestamp: datetime
    size: int
    exists: bool
    
    @classmethod
    def from_file(cls, file_path: str) -> 'FileSnapshot':
        """Create snapshot from file."""
        path = Path(file_path)
        
        if not path.exists():
            return cls(
                file_path=file_path,
                content_hash="",
                content="",
                timestamp=datetime.now(),
                size=0,
                exists=False
            )
        
        try:
            content = path.read_text(encoding='utf-8')
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            return cls(
                file_path=file_path,
                content_hash=content_hash,
                content=content,
                timestamp=datetime.now(),
                size=len(content),
                exists=True
            )
        except Exception as e:
            logger.error(f"Error creating snapshot for {file_path}: {e}")
            raise


@dataclass
class CodeBlockCheckpoint:
    """Checkpoint for a specific code block."""
    block_id: str
    block_type: str  # function, class, method, etc.
    block_name: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    dependencies: List[str] = field(default_factory=list)
    state: CheckpointState = CheckpointState.CREATED
    error_info: Optional[Dict[str, Any]] = None


@dataclass
class GranularCheckpoint:
    """Fine-grained checkpoint with multiple levels."""
    checkpoint_id: str
    task_id: str
    level: CheckpointLevel
    timestamp: datetime
    state: CheckpointState
    
    # File-level tracking
    file_snapshots: Dict[str, FileSnapshot] = field(default_factory=dict)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    
    # Block-level tracking
    code_blocks: Dict[str, CodeBlockCheckpoint] = field(default_factory=dict)
    completed_blocks: Set[str] = field(default_factory=set)
    
    # Execution context
    context: Dict[str, Any] = field(default_factory=dict)
    parent_checkpoint_id: Optional[str] = None
    child_checkpoint_ids: List[str] = field(default_factory=list)
    
    # Recovery information
    can_resume: bool = True
    resume_point: Optional[Dict[str, Any]] = None
    rollback_actions: List[Dict[str, Any]] = field(default_factory=list)


class GranularCheckpointSystem:
    """Manages fine-grained checkpoints for precise recovery."""
    
    def __init__(self, checkpoint_dir: str = ".vibe_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # In-memory checkpoint tracking
        self.active_checkpoints: Dict[str, GranularCheckpoint] = {}
        self.checkpoint_hierarchy: Dict[str, List[str]] = {}  # parent_id -> child_ids
        
        # File tracking for quick lookup
        self.file_to_checkpoints: Dict[str, Set[str]] = {}  # file_path -> checkpoint_ids
        
        # Recovery strategies
        self.recovery_strategies = {
            CheckpointLevel.TASK: self._recover_task_level,
            CheckpointLevel.FILE: self._recover_file_level,
            CheckpointLevel.FUNCTION: self._recover_function_level,
            CheckpointLevel.BLOCK: self._recover_block_level
        }
        
        # Performance metrics
        self.metrics = {
            'checkpoints_created': 0,
            'checkpoints_restored': 0,
            'files_tracked': 0,
            'blocks_tracked': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0
        }
    
    async def create_checkpoint(self, task_id: str, level: CheckpointLevel = CheckpointLevel.TASK,
                               parent_id: Optional[str] = None) -> str:
        """Create a new checkpoint."""
        checkpoint_id = self._generate_checkpoint_id(task_id, level)
        
        checkpoint = GranularCheckpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            level=level,
            timestamp=datetime.now(),
            state=CheckpointState.CREATED,
            parent_checkpoint_id=parent_id
        )
        
        # Link to parent if exists
        if parent_id and parent_id in self.active_checkpoints:
            parent = self.active_checkpoints[parent_id]
            parent.child_checkpoint_ids.append(checkpoint_id)
            self.checkpoint_hierarchy.setdefault(parent_id, []).append(checkpoint_id)
        
        self.active_checkpoints[checkpoint_id] = checkpoint
        self.metrics['checkpoints_created'] += 1
        
        # Persist checkpoint
        await self._save_checkpoint(checkpoint)
        
        logger.info(f"Created {level.value} checkpoint {checkpoint_id} for task {task_id}")
        return checkpoint_id
    
    async def snapshot_file(self, checkpoint_id: str, file_path: str) -> bool:
        """Take a snapshot of a file."""
        if checkpoint_id not in self.active_checkpoints:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        
        checkpoint = self.active_checkpoints[checkpoint_id]
        
        try:
            snapshot = FileSnapshot.from_file(file_path)
            checkpoint.file_snapshots[file_path] = snapshot
            
            # Track file in index
            self.file_to_checkpoints.setdefault(file_path, set()).add(checkpoint_id)
            self.metrics['files_tracked'] += 1
            
            # Determine if file is new or modified
            if not snapshot.exists:
                checkpoint.files_deleted.append(file_path)
            elif self._is_new_file(file_path, checkpoint_id):
                checkpoint.files_created.append(file_path)
            else:
                checkpoint.files_modified.append(file_path)
            
            await self._save_checkpoint(checkpoint)
            return True
            
        except Exception as e:
            logger.error(f"Failed to snapshot file {file_path}: {e}")
            return False
    
    async def snapshot_code_block(self, checkpoint_id: str, file_path: str,
                                 block_type: str, block_name: str,
                                 start_line: int, end_line: int,
                                 content: str, dependencies: List[str] = None) -> str:
        """Create a checkpoint for a code block."""
        if checkpoint_id not in self.active_checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint = self.active_checkpoints[checkpoint_id]
        block_id = f"{Path(file_path).stem}_{block_name}_{start_line}"
        
        block_checkpoint = CodeBlockCheckpoint(
            block_id=block_id,
            block_type=block_type,
            block_name=block_name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            dependencies=dependencies or []
        )
        
        checkpoint.code_blocks[block_id] = block_checkpoint
        self.metrics['blocks_tracked'] += 1
        
        await self._save_checkpoint(checkpoint)
        logger.debug(f"Snapshotted {block_type} '{block_name}' in {file_path}")
        
        return block_id
    
    async def mark_block_completed(self, checkpoint_id: str, block_id: str) -> bool:
        """Mark a code block as successfully completed."""
        if checkpoint_id not in self.active_checkpoints:
            return False
        
        checkpoint = self.active_checkpoints[checkpoint_id]
        
        if block_id in checkpoint.code_blocks:
            checkpoint.code_blocks[block_id].state = CheckpointState.COMPLETED
            checkpoint.completed_blocks.add(block_id)
            await self._save_checkpoint(checkpoint)
            return True
        
        return False
    
    async def update_checkpoint(self, checkpoint_id: str, 
                               state: Optional[CheckpointState] = None,
                               context: Optional[Dict[str, Any]] = None,
                               resume_point: Optional[Dict[str, Any]] = None) -> bool:
        """Update checkpoint state and context."""
        if checkpoint_id not in self.active_checkpoints:
            return False
        
        checkpoint = self.active_checkpoints[checkpoint_id]
        
        if state:
            checkpoint.state = state
        
        if context:
            checkpoint.context.update(context)
        
        if resume_point:
            checkpoint.resume_point = resume_point
        
        await self._save_checkpoint(checkpoint)
        return True
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Tuple[bool, Optional[GranularCheckpoint]]:
        """Restore from a checkpoint."""
        try:
            # Load checkpoint if not in memory
            if checkpoint_id not in self.active_checkpoints:
                checkpoint = await self._load_checkpoint(checkpoint_id)
                if not checkpoint:
                    return False, None
                self.active_checkpoints[checkpoint_id] = checkpoint
            else:
                checkpoint = self.active_checkpoints[checkpoint_id]
            
            # Execute recovery strategy based on level
            recovery_strategy = self.recovery_strategies.get(checkpoint.level)
            if recovery_strategy:
                success = await recovery_strategy(checkpoint)
                
                if success:
                    self.metrics['successful_recoveries'] += 1
                    self.metrics['checkpoints_restored'] += 1
                else:
                    self.metrics['failed_recoveries'] += 1
                
                return success, checkpoint
            
            return False, checkpoint
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            self.metrics['failed_recoveries'] += 1
            return False, None
    
    async def find_best_checkpoint(self, task_id: str, target_file: Optional[str] = None) -> Optional[str]:
        """Find the best checkpoint to restore from."""
        candidates = []
        
        for checkpoint_id, checkpoint in self.active_checkpoints.items():
            if checkpoint.task_id != task_id:
                continue
            
            # Score based on completeness and specificity
            score = 0
            
            # Prefer completed checkpoints
            if checkpoint.state == CheckpointState.COMPLETED:
                score += 100
            elif checkpoint.state == CheckpointState.IN_PROGRESS:
                score += 50
            
            # Prefer more granular checkpoints
            level_scores = {
                CheckpointLevel.BLOCK: 40,
                CheckpointLevel.FUNCTION: 30,
                CheckpointLevel.FILE: 20,
                CheckpointLevel.TASK: 10
            }
            score += level_scores.get(checkpoint.level, 0)
            
            # If targeting specific file, prefer checkpoints with that file
            if target_file and target_file in checkpoint.file_snapshots:
                score += 50
            
            # Prefer recent checkpoints
            age_minutes = (datetime.now() - checkpoint.timestamp).total_seconds() / 60
            score -= min(age_minutes, 60)  # Penalty for age, capped at 60
            
            candidates.append((score, checkpoint_id, checkpoint))
        
        if not candidates:
            return None
        
        # Sort by score and return best
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    async def create_rollback_plan(self, checkpoint_id: str) -> List[Dict[str, Any]]:
        """Create a plan to rollback to a checkpoint."""
        if checkpoint_id not in self.active_checkpoints:
            checkpoint = await self._load_checkpoint(checkpoint_id)
            if not checkpoint:
                return []
        else:
            checkpoint = self.active_checkpoints[checkpoint_id]
        
        rollback_plan = []
        
        # Plan file restorations
        for file_path, snapshot in checkpoint.file_snapshots.items():
            current_snapshot = FileSnapshot.from_file(file_path)
            
            if not snapshot.exists and current_snapshot.exists:
                # File should not exist
                rollback_plan.append({
                    'action': 'delete',
                    'file': file_path,
                    'reason': 'File created after checkpoint'
                })
            elif snapshot.exists:
                if not current_snapshot.exists:
                    # File should exist
                    rollback_plan.append({
                        'action': 'create',
                        'file': file_path,
                        'content': snapshot.content,
                        'reason': 'File missing'
                    })
                elif snapshot.content_hash != current_snapshot.content_hash:
                    # File content differs
                    rollback_plan.append({
                        'action': 'restore',
                        'file': file_path,
                        'content': snapshot.content,
                        'diff': self._generate_diff(snapshot.content, current_snapshot.content),
                        'reason': 'File modified after checkpoint'
                    })
        
        checkpoint.rollback_actions = rollback_plan
        return rollback_plan
    
    def get_checkpoint_tree(self, root_checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """Get hierarchical view of checkpoints."""
        def build_tree(checkpoint_id: str) -> Dict[str, Any]:
            if checkpoint_id not in self.active_checkpoints:
                return {'id': checkpoint_id, 'status': 'not_loaded'}
            
            checkpoint = self.active_checkpoints[checkpoint_id]
            node = {
                'id': checkpoint_id,
                'task_id': checkpoint.task_id,
                'level': checkpoint.level.value,
                'state': checkpoint.state.value,
                'timestamp': checkpoint.timestamp.isoformat(),
                'files': len(checkpoint.file_snapshots),
                'blocks': len(checkpoint.code_blocks),
                'completed_blocks': len(checkpoint.completed_blocks),
                'children': []
            }
            
            for child_id in checkpoint.child_checkpoint_ids:
                node['children'].append(build_tree(child_id))
            
            return node
        
        if root_checkpoint_id:
            return build_tree(root_checkpoint_id)
        
        # Find root checkpoints (no parent)
        roots = []
        for checkpoint_id, checkpoint in self.active_checkpoints.items():
            if not checkpoint.parent_checkpoint_id:
                roots.append(build_tree(checkpoint_id))
        
        return {'roots': roots, 'total_checkpoints': len(self.active_checkpoints)}
    
    # Private methods
    
    def _generate_checkpoint_id(self, task_id: str, level: CheckpointLevel) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"{task_id}_{level.value}_{timestamp}"
    
    def _is_new_file(self, file_path: str, current_checkpoint_id: str) -> bool:
        """Check if file is new in this checkpoint."""
        # Check if file exists in any parent checkpoint
        current = self.active_checkpoints.get(current_checkpoint_id)
        
        while current and current.parent_checkpoint_id:
            parent = self.active_checkpoints.get(current.parent_checkpoint_id)
            if parent and file_path in parent.file_snapshots:
                return False
            current = parent
        
        return True
    
    async def _save_checkpoint(self, checkpoint: GranularCheckpoint):
        """Persist checkpoint to disk."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.checkpoint"
        
        try:
            # Save as JSON for readability
            data = asdict(checkpoint)
            # Convert datetime objects
            data['timestamp'] = checkpoint.timestamp.isoformat()
            for file_data in data['file_snapshots'].values():
                if isinstance(file_data.get('timestamp'), datetime):
                    file_data['timestamp'] = file_data['timestamp'].isoformat()
            
            checkpoint_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
            
            # Also save binary backup
            backup_file = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.pkl"
            with open(backup_file, 'wb') as f:
                pickle.dump(checkpoint, f)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")
    
    async def _load_checkpoint(self, checkpoint_id: str) -> Optional[GranularCheckpoint]:
        """Load checkpoint from disk."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.checkpoint"
        backup_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        # Try pickle first (preserves types)
        if backup_file.exists():
            try:
                with open(backup_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load pickle checkpoint: {e}")
        
        # Fallback to JSON
        if checkpoint_file.exists():
            try:
                data = json.loads(checkpoint_file.read_text(encoding='utf-8'))
                # Reconstruct objects
                # This would need proper deserialization logic
                logger.warning("JSON checkpoint loading not fully implemented")
                return None
            except Exception as e:
                logger.error(f"Failed to load JSON checkpoint: {e}")
        
        return None
    
    async def _recover_task_level(self, checkpoint: GranularCheckpoint) -> bool:
        """Recover at task level - restore all files."""
        try:
            for file_path, snapshot in checkpoint.file_snapshots.items():
                if snapshot.exists:
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(file_path).write_text(snapshot.content, encoding='utf-8')
                elif Path(file_path).exists():
                    Path(file_path).unlink()
            
            logger.info(f"Recovered task-level checkpoint {checkpoint.checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Task-level recovery failed: {e}")
            return False
    
    async def _recover_file_level(self, checkpoint: GranularCheckpoint) -> bool:
        """Recover specific files from checkpoint."""
        try:
            recovered_files = 0
            
            for file_path in checkpoint.files_modified + checkpoint.files_created:
                if file_path in checkpoint.file_snapshots:
                    snapshot = checkpoint.file_snapshots[file_path]
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(file_path).write_text(snapshot.content, encoding='utf-8')
                    recovered_files += 1
            
            logger.info(f"Recovered {recovered_files} files from checkpoint {checkpoint.checkpoint_id}")
            return recovered_files > 0
            
        except Exception as e:
            logger.error(f"File-level recovery failed: {e}")
            return False
    
    async def _recover_function_level(self, checkpoint: GranularCheckpoint) -> bool:
        """Recover at function/class level."""
        try:
            # Group blocks by file
            file_blocks = {}
            for block_id, block in checkpoint.code_blocks.items():
                if block.state == CheckpointState.COMPLETED:
                    file_blocks.setdefault(block.file_path, []).append(block)
            
            # Reconstruct files with completed blocks
            for file_path, blocks in file_blocks.items():
                # This would need sophisticated code merging logic
                logger.info(f"Function-level recovery for {file_path} with {len(blocks)} blocks")
                # Placeholder - actual implementation would merge code blocks
            
            return True
            
        except Exception as e:
            logger.error(f"Function-level recovery failed: {e}")
            return False
    
    async def _recover_block_level(self, checkpoint: GranularCheckpoint) -> bool:
        """Recover at block level - most granular."""
        try:
            # Similar to function level but more granular
            logger.info(f"Block-level recovery for checkpoint {checkpoint.checkpoint_id}")
            # Placeholder - would need AST manipulation
            return True
            
        except Exception as e:
            logger.error(f"Block-level recovery failed: {e}")
            return False
    
    def _generate_diff(self, old_content: str, new_content: str) -> List[str]:
        """Generate diff between two versions of content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile='checkpoint',
            tofile='current',
            n=3
        ))
        
        return diff


# Singleton instance
_granular_checkpoint_system: Optional[GranularCheckpointSystem] = None


def get_checkpoint_system() -> GranularCheckpointSystem:
    """Get or create the singleton checkpoint system."""
    global _granular_checkpoint_system
    if _granular_checkpoint_system is None:
        _granular_checkpoint_system = GranularCheckpointSystem()
    return _granular_checkpoint_system