"""Task JSON schema definition and validation logic for VIBE."""

import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, validator, ValidationError
from jsonschema import validate, ValidationError as JsonSchemaError
import logging

logger = logging.getLogger(__name__)


# JSON Schema for tasks.json validation
TASKS_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "project_id": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]+$",
            "description": "Unique identifier for the project"
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp of when the tasks were created"
        },
        "total_tasks": {
            "type": "integer",
            "minimum": 0,
            "description": "Total number of tasks in the plan"
        },
        "tasks": {
            "type": "array",
            "items": {
                "$ref": "#/definitions/task"
            },
            "description": "Array of task objects"
        }
    },
    "required": ["project_id", "created_at", "total_tasks", "tasks"],
    "definitions": {
        "task": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "description": "Unique identifier for the task"
                },
                "description": {
                    "type": "string",
                    "minLength": 10,
                    "description": "Clear description of what the task accomplishes"
                },
                "type": {
                    "type": "string",
                    "enum": ["setup", "backend", "frontend", "database", "testing", "deployment", "general"],
                    "description": "Category of the task"
                },
                "dependencies": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Array of task IDs that must complete before this task"
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "failed"],
                    "default": "pending",
                    "description": "Current status of the task"
                },
                "project_area": {
                    "type": "string",
<<<<<<< HEAD
                    "enum": ["backend", "frontend", "shared", "database", "infrastructure", "testing", "deployment", "documentation"],
=======
                    "enum": ["backend", "frontend", "shared"],
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
                    "description": "Which area of the project this task belongs to"
                },
                "files_to_create_or_modify": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of file paths that will be created or modified"
                },
                "acceptance_criteria": {
                    "$ref": "#/definitions/acceptance_criteria"
                },
                "estimated_hours": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 8.0,
                    "description": "Estimated time to complete the task in hours"
                },
                "technical_details": {
                    "$ref": "#/definitions/technical_details"
                }
            },
            "required": ["id", "description", "type", "dependencies", "project_area", "files_to_create_or_modify", "acceptance_criteria"]
        },
        "acceptance_criteria": {
            "type": "object",
            "properties": {
                "tests": {
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/test_criteria"
                    },
                    "description": "Automated tests that must pass"
                },
                "linting": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "files": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "description": "Linting checks that must pass"
                },
                "manual_checks": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Manual verification steps"
                }
            },
            "required": ["tests", "linting", "manual_checks"]
        },
        "test_criteria": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["unit", "integration", "e2e"]
                },
                "file": {
                    "type": "string",
                    "description": "Path to the test file"
                },
                "function": {
                    "type": "string",
                    "description": "Name of the test function"
                }
            },
            "required": ["type", "file"]
        },
        "technical_details": {
            "type": "object",
            "properties": {
                "framework_specific": {
                    "type": "string",
                    "description": "Framework-specific implementation notes"
                },
                "dependencies_to_install": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Package dependencies that need to be installed"
                },
                "environment_variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Environment variables required for this task"
                }
            }
        }
    }
}


# Pydantic models for type-safe task handling
class TestCriteria(BaseModel):
    type: str = Field(..., pattern=r"^(unit|integration|e2e)$")
    file: str
    function: Optional[str] = None


class AcceptanceCriteria(BaseModel):
    tests: List[TestCriteria] = Field(default_factory=list)
    linting: Dict[str, Any] = Field(default_factory=dict)
    manual_checks: List[str] = Field(default_factory=list)


class TechnicalDetails(BaseModel):
    framework_specific: Optional[str] = None
    dependencies_to_install: List[str] = Field(default_factory=list)
    environment_variables: List[str] = Field(default_factory=list)


class Task(BaseModel):
    id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = Field(..., min_length=10)
    type: str = Field(..., pattern=r"^(setup|backend|frontend|database|testing|deployment|general)$")
    dependencies: List[str] = Field(default_factory=list)
    status: str = Field(default="pending", pattern=r"^(pending|in_progress|completed|failed)$")
<<<<<<< HEAD
    project_area: str = Field(..., pattern=r"^(backend|frontend|shared|database|infrastructure|testing|deployment|documentation)$")
=======
    project_area: str = Field(..., pattern=r"^(backend|frontend|shared)$")
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
    files_to_create_or_modify: List[str] = Field(default_factory=list)
    acceptance_criteria: AcceptanceCriteria
    estimated_hours: Optional[float] = Field(None, ge=0.5, le=8.0)
    technical_details: Optional[TechnicalDetails] = None
    
    @validator('dependencies')
    def validate_dependencies(cls, v, values):
        # Ensure task doesn't depend on itself
        if 'id' in values and values['id'] in v:
            raise ValueError("Task cannot depend on itself")
        return v


class TasksPlan(BaseModel):
    project_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    created_at: str
    total_tasks: int = Field(..., ge=0)
    tasks: List[Task]
    
    @validator('created_at')
    def validate_created_at(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("created_at must be a valid ISO 8601 timestamp")
        return v
    
    @validator('total_tasks')
    def validate_total_tasks(cls, v, values):
        if 'tasks' in values and len(values['tasks']) != v:
            raise ValueError("total_tasks must match the actual number of tasks")
        return v
    
    @validator('tasks')
    def validate_task_dependencies(cls, v):
        task_ids = {task.id for task in v}
        
        for task in v:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep_id}")
        
        # Check for circular dependencies
        if has_circular_dependencies(v):
            raise ValueError("Circular dependencies detected in task plan")
        
        return v


def has_circular_dependencies(tasks: List[Task]) -> bool:
    """Check if the task list has circular dependencies using DFS."""
    task_map = {task.id: task for task in tasks}
    visited = set()
    rec_stack = set()
    
    def dfs(task_id: str) -> bool:
        if task_id in rec_stack:
            return True  # Found a cycle
        if task_id in visited:
            return False
        
        visited.add(task_id)
        rec_stack.add(task_id)
        
        task = task_map.get(task_id)
        if task:
            for dep_id in task.dependencies:
                if dfs(dep_id):
                    return True
        
        rec_stack.remove(task_id)
        return False
    
    for task in tasks:
        if task.id not in visited:
            if dfs(task.id):
                return True
    
    return False


class TaskSchemaValidator:
    """Validator for tasks.json files."""
    
    @staticmethod
    def validate_json_schema(data: Dict[str, Any]) -> bool:
        """Validate data against the JSON schema."""
        try:
            validate(instance=data, schema=TASKS_JSON_SCHEMA)
            return True
        except JsonSchemaError as e:
            logger.error(f"JSON schema validation failed: {e}")
            return False
    
    @staticmethod
    def validate_pydantic_model(data: Dict[str, Any]) -> Optional[TasksPlan]:
        """Validate data using Pydantic models."""
        try:
            return TasksPlan(**data)
        except ValidationError as e:
            logger.error(f"Pydantic validation failed: {e}")
            return None
    
    @classmethod
    def validate_file(cls, file_path: str) -> Optional[TasksPlan]:
        """Validate a tasks.json file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading tasks file {file_path}: {e}")
            return None
        
        # First validate against JSON schema
        if not cls.validate_json_schema(data):
            return None
        
        # Then validate with Pydantic for type safety
        return cls.validate_pydantic_model(data)
    
    @classmethod
    def validate_and_fix(cls, data: Dict[str, Any]) -> TasksPlan:
        """Validate data and apply fixes for common issues."""
        # Try to fix common issues
        fixed_data = cls._apply_fixes(data.copy())
        
        # Validate the fixed data
        validated = cls.validate_pydantic_model(fixed_data)
        if validated is None:
            raise ValueError("Unable to validate tasks data even after applying fixes")
        
        return validated
    
    @staticmethod
    def _apply_fixes(data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply common fixes to task data."""
        # Ensure required top-level fields
        if 'project_id' not in data:
            data['project_id'] = f"vibe-project-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if 'created_at' not in data:
            data['created_at'] = datetime.now().isoformat()
        
        # Fix tasks
        tasks = data.get('tasks', [])
        fixed_tasks = []
        
        for i, task in enumerate(tasks):
            fixed_task = task.copy()
            
            # Ensure task ID
            if 'id' not in fixed_task or not fixed_task['id']:
                fixed_task['id'] = f"task-{i+1:04d}"
            
            # Ensure required fields with defaults
            if 'dependencies' not in fixed_task:
                fixed_task['dependencies'] = []
            
            if 'status' not in fixed_task:
                fixed_task['status'] = 'pending'
            
            if 'files_to_create_or_modify' not in fixed_task:
                fixed_task['files_to_create_or_modify'] = []
            
            if 'acceptance_criteria' not in fixed_task:
                fixed_task['acceptance_criteria'] = {
                    'tests': [],
                    'linting': {},
                    'manual_checks': []
                }
            
            # Infer type if missing
            if 'type' not in fixed_task:
                fixed_task['type'] = 'general'
            
<<<<<<< HEAD
            # Infer project_area if missing or invalid
            if 'project_area' not in fixed_task:
                # Infer based on task type
                task_type = fixed_task.get('type', 'general')
                if task_type == 'database':
                    fixed_task['project_area'] = 'database'
                elif task_type == 'backend':
                    fixed_task['project_area'] = 'backend'
                elif task_type == 'frontend':
                    fixed_task['project_area'] = 'frontend'
                elif task_type == 'testing':
                    fixed_task['project_area'] = 'testing'
                elif task_type == 'deployment':
                    fixed_task['project_area'] = 'deployment'
                else:
                    fixed_task['project_area'] = 'shared'
            elif fixed_task['project_area'] not in ['backend', 'frontend', 'shared', 'database', 'infrastructure', 'testing', 'deployment', 'documentation']:
                # Fix invalid project_area values
                task_type = fixed_task.get('type', 'general')
                if task_type == 'database':
                    fixed_task['project_area'] = 'database'
                elif task_type == 'backend':
                    fixed_task['project_area'] = 'backend'
                elif task_type == 'frontend':
                    fixed_task['project_area'] = 'frontend'
                elif task_type == 'testing':
                    fixed_task['project_area'] = 'testing'
                elif task_type == 'deployment':
                    fixed_task['project_area'] = 'deployment'
                else:
                    fixed_task['project_area'] = 'shared'
=======
            # Infer project_area if missing
            if 'project_area' not in fixed_task:
                fixed_task['project_area'] = 'shared'
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            
            fixed_tasks.append(fixed_task)
        
        data['tasks'] = fixed_tasks
        data['total_tasks'] = len(fixed_tasks)
        
        return data


def create_sample_tasks() -> TasksPlan:
    """Create a sample tasks plan for testing."""
    return TasksPlan(
        project_id="sample-project",
        created_at=datetime.now().isoformat(),
        total_tasks=3,
        tasks=[
            Task(
                id="setup-project",
                description="Initialize project structure and dependencies",
                type="setup",
                dependencies=[],
                project_area="shared",
                files_to_create_or_modify=["package.json", "requirements.txt"],
                acceptance_criteria=AcceptanceCriteria(
                    tests=[],
                    linting={},
                    manual_checks=["Project directory structure is created"]
                ),
                estimated_hours=1.0
            ),
            Task(
                id="create-backend",
                description="Create basic backend API structure",
                type="backend",
                dependencies=["setup-project"],
                project_area="backend",
                files_to_create_or_modify=["backend/app.py", "backend/models.py"],
                acceptance_criteria=AcceptanceCriteria(
                    tests=[
                        TestCriteria(type="unit", file="tests/test_backend.py", function="test_api_health")
                    ],
                    linting={"command": "ruff check backend/"},
                    manual_checks=["API responds to health check"]
                ),
                estimated_hours=2.0
            ),
            Task(
                id="create-frontend",
                description="Create basic frontend structure",
                type="frontend",
                dependencies=["setup-project"],
                project_area="frontend",
                files_to_create_or_modify=["frontend/src/App.js", "frontend/src/components/"],
                acceptance_criteria=AcceptanceCriteria(
                    tests=[
                        TestCriteria(type="unit", file="frontend/src/App.test.js", function="test_app_renders")
                    ],
                    linting={"command": "npm run lint"},
                    manual_checks=["Frontend loads without errors"]
                ),
                estimated_hours=2.0
            )
        ]
    )