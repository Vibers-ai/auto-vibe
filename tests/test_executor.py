"""Tests for task executor."""

import pytest
import asyncio
import tempfile
import sqlite3
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.core.executor import TaskExecutor, TaskStatus, ExecutorState
from src.core.schema import Task, TasksPlan, AcceptanceCriteria, create_sample_tasks
from src.utils.config import Config


class TestExecutorState:
    """Test executor state management."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
            db_path = temp_file.name
        
        yield db_path
        
        # Cleanup
        try:
            import os
            os.unlink(db_path)
        except OSError:
            pass
    
    def test_database_initialization(self, temp_db):
        """Test database initialization."""
        state = ExecutorState(temp_db)
        
        # Check that tables were created
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            
            # Check execution_sessions table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='execution_sessions'")
            assert cursor.fetchone() is not None
            
            # Check task_executions table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_executions'")
            assert cursor.fetchone() is not None
    
    def test_session_creation(self, temp_db):
        """Test session creation."""
        state = ExecutorState(temp_db)
        
        session_id = state.create_session(
            project_id="test-project",
            tasks_file_path="test-tasks.json",
            total_tasks=5
        )
        
        assert session_id.startswith("session-")
        
        # Verify session was stored
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT project_id, total_tasks FROM execution_sessions WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == "test-project"
            assert result[1] == 5
    
    def test_task_status_updates(self, temp_db):
        """Test task status updates."""
        state = ExecutorState(temp_db)
        
        session_id = state.create_session("test-project", "test.json", 1)
        
        # Update task status
        state.update_task_status(session_id, "task-1", "in_progress")
        
        # Verify status was updated
        status = state.get_task_status(session_id, "task-1")
        assert status == "in_progress"
        
        # Complete the task
        state.update_task_status(session_id, "task-1", "completed", output="Task completed successfully")
        
        status = state.get_task_status(session_id, "task-1")
        assert status == "completed"
    
    def test_session_progress_tracking(self, temp_db):
        """Test session progress tracking."""
        state = ExecutorState(temp_db)
        
        session_id = state.create_session("test-project", "test.json", 3)
        
        # Add some task executions
        state.update_task_status(session_id, "task-1", "completed")
        state.update_task_status(session_id, "task-2", "failed", error="Task failed")
        state.update_task_status(session_id, "task-3", "in_progress")
        
        # Get progress
        progress = state.get_session_progress(session_id)
        
        assert progress['project_id'] == "test-project"
        assert progress['total_tasks'] == 3
        assert len(progress['tasks']) == 3
        assert progress['tasks']['task-1']['status'] == "completed"
        assert progress['tasks']['task-2']['status'] == "failed"
        assert progress['tasks']['task-2']['last_error'] == "Task failed"


class TestTaskExecutor:
    """Test task executor functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock(spec=Config)
        config.parallel_tasks = 2
        return config
    
    @pytest.fixture
    def sample_tasks_file(self):
        """Create a sample tasks.json file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            sample_plan = create_sample_tasks()
            temp_file.write(sample_plan.json(indent=2))
            temp_file.flush()
            
            yield temp_file.name
        
        # Cleanup
        try:
            import os
            os.unlink(temp_file.name)
        except OSError:
            pass
    
    @pytest.fixture
    def simple_tasks_plan(self):
        """Create a simple tasks plan for testing."""
        task1 = Task(
            id="task-1",
            description="First task",
            type="setup",
            dependencies=[],
            project_area="shared",
            files_to_create_or_modify=["file1.py"],
            acceptance_criteria=AcceptanceCriteria()
        )
        
        task2 = Task(
            id="task-2",
            description="Second task",
            type="backend",
            dependencies=["task-1"],
            project_area="backend",
            files_to_create_or_modify=["file2.py"],
            acceptance_criteria=AcceptanceCriteria()
        )
        
        return TasksPlan(
            project_id="test-project",
            created_at=datetime.now().isoformat(),
            total_tasks=2,
            tasks=[task1, task2]
        )
    
    def test_task_graph_building(self, mock_config, simple_tasks_plan):
        """Test building task dependency graph."""
        executor = TaskExecutor(mock_config)
        executor.tasks_plan = simple_tasks_plan
        
        executor._build_task_graph()
        
        assert executor.task_graph is not None
        assert len(executor.task_graph.nodes) == 2
        assert len(executor.task_graph.edges) == 1
        
        # Check that the dependency edge exists
        assert executor.task_graph.has_edge("task-1", "task-2")
        
        # Verify it's a DAG
        import networkx as nx
        assert nx.is_directed_acyclic_graph(executor.task_graph)
    
    def test_task_status_initialization(self, mock_config, simple_tasks_plan):
        """Test task status initialization."""
        executor = TaskExecutor(mock_config)
        executor.tasks_plan = simple_tasks_plan
        executor.session_id = "test-session"
        
        # Mock the state
        executor.state = Mock()
        
        executor._initialize_task_status()
        
        # task-1 has no dependencies, should be READY
        assert executor.task_status_map["task-1"] == TaskStatus.READY
        
        # task-2 has dependencies, should be PENDING
        assert executor.task_status_map["task-2"] == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_task_dependency_resolution(self, mock_config, simple_tasks_plan):
        """Test that completing a task makes dependent tasks ready."""
        executor = TaskExecutor(mock_config)
        executor.tasks_plan = simple_tasks_plan
        executor.session_id = "test-session"
        executor.state = Mock()
        
        # Build graph and initialize status
        executor._build_task_graph()
        executor._initialize_task_status()
        
        # Initially task-2 should be pending
        assert executor.task_status_map["task-2"] == TaskStatus.PENDING
        
        # Complete task-1
        executor.task_status_map["task-1"] = TaskStatus.COMPLETED
        
        # Check if task-2 becomes ready
        await executor._check_and_queue_ready_tasks("task-1")
        
        assert executor.task_status_map["task-2"] == TaskStatus.READY
    
    @pytest.mark.asyncio
    async def test_failed_task_blocks_dependencies(self, mock_config, simple_tasks_plan):
        """Test that failing a task blocks dependent tasks."""
        executor = TaskExecutor(mock_config)
        executor.tasks_plan = simple_tasks_plan
        executor.session_id = "test-session"
        executor.state = Mock()
        
        # Build graph and initialize status
        executor._build_task_graph()
        executor._initialize_task_status()
        
        # Mark task-1 as failed
        executor.task_status_map["task-1"] = TaskStatus.FAILED
        
        # Check that dependent tasks get blocked
        await executor._mark_dependent_tasks_blocked("task-1")
        
        assert executor.task_status_map["task-2"] == TaskStatus.BLOCKED
    
    @pytest.mark.asyncio
    async def test_single_task_execution_placeholder(self, mock_config):
        """Test single task execution (placeholder implementation)."""
        executor = TaskExecutor(mock_config)
        
        task = Task(
            id="test-task",
            description="Test task",
            type="backend",
            dependencies=[],
            project_area="backend",
            files_to_create_or_modify=["test.py"],
            acceptance_criteria=AcceptanceCriteria()
        )
        
        # The current implementation is a placeholder that always succeeds
        result = await executor._execute_single_task(task)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_load_tasks_with_valid_file(self, mock_config, sample_tasks_file):
        """Test loading tasks from a valid file."""
        executor = TaskExecutor(mock_config)
        
        result = await executor._load_tasks(sample_tasks_file)
        
        assert result is True
        assert executor.tasks_plan is not None
        assert len(executor.tasks_plan.tasks) > 0
    
    @pytest.mark.asyncio
    async def test_load_tasks_with_invalid_file(self, mock_config):
        """Test loading tasks from an invalid file."""
        executor = TaskExecutor(mock_config)
        
        result = await executor._load_tasks("nonexistent-file.json")
        
        assert result is False
        assert executor.tasks_plan is None
    
    def test_execution_summary_without_session(self, mock_config):
        """Test getting execution summary without an active session."""
        executor = TaskExecutor(mock_config)
        
        summary = executor.get_execution_summary()
        
        assert "error" in summary
        assert summary["error"] == "No active execution session"


class TestTaskExecutorIntegration:
    """Integration tests for task executor."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock(spec=Config)
        config.parallel_tasks = 1  # Use single thread for predictable testing
        return config
    
    @pytest.fixture
    def circular_dependency_tasks(self):
        """Create tasks with circular dependencies for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            # Create tasks with circular dependency
            task1 = Task(
                id="task-1",
                description="First task",
                type="setup",
                dependencies=["task-2"],  # Depends on task-2
                project_area="shared",
                files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            )
            
            task2 = Task(
                id="task-2", 
                description="Second task",
                type="backend",
                dependencies=["task-1"],  # Depends on task-1 (circular!)
                project_area="backend",
                files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            )
            
            plan = TasksPlan(
                project_id="circular-test",
                created_at=datetime.now().isoformat(),
                total_tasks=2,
                tasks=[task1, task2]
            )
            
            temp_file.write(plan.json(indent=2))
            temp_file.flush()
            
            yield temp_file.name
        
        # Cleanup
        try:
            import os
            os.unlink(temp_file.name)
        except OSError:
            pass
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, mock_config, circular_dependency_tasks):
        """Test that circular dependencies are detected and cause failure."""
        executor = TaskExecutor(mock_config)
        
        # This should fail during task loading due to circular dependencies
        result = await executor._load_tasks(circular_dependency_tasks)
        
        # The validation should catch the circular dependency
        assert result is False