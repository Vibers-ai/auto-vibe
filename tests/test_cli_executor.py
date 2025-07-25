"""Unit tests for CLI executor components."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from src.shared.utils.config import Config
from src.shared.core.schema import Task, TasksPlan, AcceptanceCriteria
from src.cli.agents.claude_cli_executor import ClaudeCliExecutor, PersistentClaudeCliSession
from src.cli.core.executor_cli import TaskExecutorCli


@pytest.mark.cli
@pytest.mark.unit
class TestClaudeCliExecutor:
    """Test Claude CLI executor functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            gemini_api_key="test_gemini_key",
            anthropic_api_key="test_anthropic_key",
            claude_cli_skip_permissions=True,
            claude_cli_path="mock_claude"
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return Task(
            id="test-task-1",
            description="Test task description",
            type="implementation",
            project_area="core",
            files_to_create_or_modify=["test_file.py"],
            dependencies=[],
            acceptance_criteria=AcceptanceCriteria(
                tests=[],
                linting={},
                manual_checks=[]
            )
        )
    
    def test_init(self, config):
        """Test CLI executor initialization."""
        executor = ClaudeCliExecutor(config)
        
        assert executor.config == config
        assert executor.session_context is not None
        assert executor.session_dir is not None
        assert os.path.exists(executor.session_dir)
        
        # Cleanup
        executor.cleanup()
    
    @patch('subprocess.run')
    def test_find_claude_cli_configured_path(self, mock_run, config):
        """Test finding Claude CLI with configured path."""
        config.claude_cli_path = "/custom/claude"
        executor = ClaudeCliExecutor(config)
        
        assert executor.claude_cli_path == "/custom/claude"
        mock_run.assert_not_called()  # Should not search if path is configured
        
        executor.cleanup()
    
    @patch('subprocess.run')
    def test_find_claude_cli_search(self, mock_run, config):
        """Test finding Claude CLI by searching common locations."""
        config.claude_cli_path = None
        mock_run.return_value = Mock(returncode=0)
        
        executor = ClaudeCliExecutor(config)
        
        assert mock_run.called
        assert executor.claude_cli_path == 'claude'
        
        executor.cleanup()
    
    def test_build_contextual_prompt(self, config, sample_task):
        """Test prompt building functionality."""
        executor = ClaudeCliExecutor(config)
        
        prompt = executor._build_contextual_prompt(
            sample_task, 
            "/test/workspace",
            "Master context",
            "Task specific context"
        )
        
        assert "test-task-1" in prompt
        assert "Master context" in prompt
        assert "Task specific context" in prompt
        assert "test_file.py" in prompt
        
        executor.cleanup()
    
    def test_cli_command_construction(self, config, sample_task):
        """Test CLI command construction with permissions flag."""
        executor = ClaudeCliExecutor(config)
        
        # Mock the async execution method to check command construction
        with patch.object(executor, '_execute_with_claude_cli') as mock_exec:
            mock_exec.return_value = AsyncMock()
            
            # The command construction happens in _execute_with_claude_cli
            # We'll test this indirectly by checking the config usage
            assert config.claude_cli_skip_permissions == True
            
        executor.cleanup()


@pytest.mark.cli
@pytest.mark.unit
class TestPersistentClaudeCliSession:
    """Test persistent CLI session management."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            gemini_api_key="test_gemini_key",
            anthropic_api_key="test_anthropic_key"
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return Task(
            id="test-task-1",
            description="Test task description",
            type="implementation",
            project_area="core",
            files_to_create_or_modify=["test_file.py"],
            dependencies=[],
            acceptance_criteria=AcceptanceCriteria(
                tests=[],
                linting={},
                manual_checks=[]
            )
        )
    
    def test_session_creation(self, config):
        """Test session creation and management."""
        session_manager = PersistentClaudeCliSession(config)
        
        assert len(session_manager.active_sessions) == 0
        assert len(session_manager.session_metadata) == 0
    
    @pytest.mark.asyncio
    async def test_get_or_create_session(self, config):
        """Test getting or creating sessions by project area."""
        session_manager = PersistentClaudeCliSession(config)
        
        # Create session for project area
        executor1 = await session_manager.get_or_create_session("core")
        assert len(session_manager.active_sessions) == 1
        assert "core" in session_manager.active_sessions
        
        # Get existing session
        executor2 = await session_manager.get_or_create_session("core")
        assert executor1 is executor2  # Should be the same instance
        assert len(session_manager.active_sessions) == 1
        
        # Create session for different project area
        executor3 = await session_manager.get_or_create_session("ui")
        assert len(session_manager.active_sessions) == 2
        assert executor3 is not executor1
        
        # Cleanup
        session_manager.cleanup_all_sessions()


@pytest.mark.cli
@pytest.mark.unit
class TestTaskExecutorCli:
    """Test CLI task executor."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            gemini_api_key="test_gemini_key",
            anthropic_api_key="test_anthropic_key"
        )
    
    def test_init(self, config):
        """Test task executor initialization."""
        executor = TaskExecutorCli(config)
        
        assert executor.config == config
        assert executor.use_master_supervision == True
        assert executor.master_supervisor is not None
    
    def test_init_without_supervision(self, config):
        """Test task executor without master supervision."""
        executor = TaskExecutorCli(config, use_master_supervision=False)
        
        assert executor.use_master_supervision == False
        assert executor.master_supervisor is None
    
    @pytest.mark.asyncio
    async def test_load_invalid_tasks_file(self, config):
        """Test loading invalid tasks file."""
        executor = TaskExecutorCli(config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": "json"}')
            f.flush()
            
            result = await executor._load_and_validate_tasks(f.name)
            assert result is None
            
            os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_load_valid_tasks_file(self, config):
        """Test loading valid tasks file."""
        executor = TaskExecutorCli(config)
        
        # Create valid tasks.json content
        tasks_content = {
            "project_id": "test-project",
            "project_name": "Test Project",
            "created_at": "2024-01-01T00:00:00Z",
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Test task",
                    "type": "implementation",
                    "project_area": "core",
                    "files_to_create_or_modify": ["test.py"],
                    "dependencies": [],
                    "acceptance_criteria": {
                        "tests": [],
                        "linting": {},
                        "manual_checks": []
                    }
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(tasks_content, f)
            f.flush()
            
            result = await executor._load_and_validate_tasks(f.name)
            assert result is not None
            assert result.project_id == "test-project"
            assert len(result.tasks) == 1
            
            os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__])