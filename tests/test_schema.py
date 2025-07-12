"""Tests for task schema validation."""

import pytest
import json
from datetime import datetime

from src.core.schema import (
    Task, TasksPlan, AcceptanceCriteria, TestCriteria,
    TaskSchemaValidator, create_sample_tasks, has_circular_dependencies
)


class TestTaskSchema:
    """Test task schema validation."""
    
    def test_valid_task_creation(self):
        """Test creating a valid task."""
        task = Task(
            id="test-task",
            description="This is a test task for validation",
            type="backend",
            dependencies=[],
            project_area="backend",
            files_to_create_or_modify=["test.py"],
            acceptance_criteria=AcceptanceCriteria(
                tests=[TestCriteria(type="unit", file="test_file.py")],
                linting={},
                manual_checks=[]
            )
        )
        
        assert task.id == "test-task"
        assert task.type == "backend"
        assert task.status == "pending"  # default value
    
    def test_invalid_task_type(self):
        """Test task creation with invalid type."""
        with pytest.raises(ValueError):
            Task(
                id="test-task",
                description="This is a test task",
                type="invalid-type",  # Invalid type
                dependencies=[],
                project_area="backend",
                files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            )
    
    def test_self_dependency_validation(self):
        """Test that tasks cannot depend on themselves."""
        with pytest.raises(ValueError):
            Task(
                id="test-task",
                description="This is a test task",
                type="backend",
                dependencies=["test-task"],  # Self-dependency
                project_area="backend",
                files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            )
    
    def test_valid_tasks_plan(self):
        """Test creating a valid tasks plan."""
        task1 = Task(
            id="task-1",
            description="First task",
            type="setup",
            dependencies=[],
            project_area="shared",
            files_to_create_or_modify=[],
            acceptance_criteria=AcceptanceCriteria()
        )
        
        task2 = Task(
            id="task-2",
            description="Second task",
            type="backend",
            dependencies=["task-1"],
            project_area="backend",
            files_to_create_or_modify=[],
            acceptance_criteria=AcceptanceCriteria()
        )
        
        plan = TasksPlan(
            project_id="test-project",
            created_at=datetime.now().isoformat(),
            total_tasks=2,
            tasks=[task1, task2]
        )
        
        assert plan.project_id == "test-project"
        assert len(plan.tasks) == 2
        assert plan.total_tasks == 2
    
    def test_invalid_dependency_reference(self):
        """Test tasks plan with invalid dependency reference."""
        task1 = Task(
            id="task-1",
            description="First task",
            type="setup",
            dependencies=["nonexistent-task"],  # Invalid dependency
            project_area="shared",
            files_to_create_or_modify=[],
            acceptance_criteria=AcceptanceCriteria()
        )
        
        with pytest.raises(ValueError):
            TasksPlan(
                project_id="test-project",
                created_at=datetime.now().isoformat(),
                total_tasks=1,
                tasks=[task1]
            )
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        task1 = Task(
            id="task-1",
            description="First task",
            type="setup",
            dependencies=["task-2"],
            project_area="shared",
            files_to_create_or_modify=[],
            acceptance_criteria=AcceptanceCriteria()
        )
        
        task2 = Task(
            id="task-2",
            description="Second task",
            type="backend",
            dependencies=["task-1"],
            project_area="backend",
            files_to_create_or_modify=[],
            acceptance_criteria=AcceptanceCriteria()
        )
        
        with pytest.raises(ValueError):
            TasksPlan(
                project_id="test-project",
                created_at=datetime.now().isoformat(),
                total_tasks=2,
                tasks=[task1, task2]
            )
    
    def test_sample_tasks_creation(self):
        """Test creating sample tasks."""
        sample_plan = create_sample_tasks()
        
        assert isinstance(sample_plan, TasksPlan)
        assert len(sample_plan.tasks) > 0
        assert sample_plan.total_tasks == len(sample_plan.tasks)
    
    def test_schema_validator_with_valid_data(self):
        """Test schema validator with valid data."""
        sample_plan = create_sample_tasks()
        data = json.loads(sample_plan.json())
        
        # Test JSON schema validation
        assert TaskSchemaValidator.validate_json_schema(data)
        
        # Test Pydantic validation
        validated_plan = TaskSchemaValidator.validate_pydantic_model(data)
        assert validated_plan is not None
        assert isinstance(validated_plan, TasksPlan)
    
    def test_schema_validator_with_invalid_data(self):
        """Test schema validator with invalid data."""
        invalid_data = {
            "project_id": "",  # Empty project ID
            "created_at": "invalid-date",
            "total_tasks": -1,  # Negative number
            "tasks": []
        }
        
        # Should fail JSON schema validation
        assert not TaskSchemaValidator.validate_json_schema(invalid_data)
        
        # Should fail Pydantic validation
        validated_plan = TaskSchemaValidator.validate_pydantic_model(invalid_data)
        assert validated_plan is None
    
    def test_schema_validator_fixes(self):
        """Test schema validator's ability to fix common issues."""
        incomplete_data = {
            "tasks": [
                {
                    "description": "A task without required fields",
                    "type": "backend"
                }
            ]
        }
        
        # Should be able to fix and validate
        fixed_plan = TaskSchemaValidator.validate_and_fix(incomplete_data)
        assert fixed_plan is not None
        assert isinstance(fixed_plan, TasksPlan)
        assert len(fixed_plan.tasks) == 1
        
        # Check that missing fields were added
        task = fixed_plan.tasks[0]
        assert task.id is not None
        assert task.project_area is not None
        assert task.dependencies == []
        assert task.acceptance_criteria is not None


class TestCircularDependencies:
    """Test circular dependency detection."""
    
    def test_no_circular_dependencies(self):
        """Test with valid DAG."""
        tasks = [
            Task(
                id="a", description="Task A", type="setup", dependencies=[],
                project_area="shared", files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            ),
            Task(
                id="b", description="Task B", type="backend", dependencies=["a"],
                project_area="backend", files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            ),
            Task(
                id="c", description="Task C", type="frontend", dependencies=["a"],
                project_area="frontend", files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            )
        ]
        
        assert not has_circular_dependencies(tasks)
    
    def test_simple_circular_dependency(self):
        """Test simple circular dependency A -> B -> A."""
        tasks = [
            Task(
                id="a", description="Task A", type="setup", dependencies=["b"],
                project_area="shared", files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            ),
            Task(
                id="b", description="Task B", type="backend", dependencies=["a"],
                project_area="backend", files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            )
        ]
        
        assert has_circular_dependencies(tasks)
    
    def test_complex_circular_dependency(self):
        """Test complex circular dependency A -> B -> C -> A."""
        tasks = [
            Task(
                id="a", description="Task A", type="setup", dependencies=["c"],
                project_area="shared", files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            ),
            Task(
                id="b", description="Task B", type="backend", dependencies=["a"],
                project_area="backend", files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            ),
            Task(
                id="c", description="Task C", type="frontend", dependencies=["b"],
                project_area="frontend", files_to_create_or_modify=[],
                acceptance_criteria=AcceptanceCriteria()
            )
        ]
        
        assert has_circular_dependencies(tasks)