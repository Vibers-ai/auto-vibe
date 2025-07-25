"""Core shared components."""

from .schema import (
    Task,
    TasksPlan,
    AcceptanceCriteria,
    TechnicalDetails,
    TestCriteria,
    TaskSchemaValidator,
    TASKS_JSON_SCHEMA,
    create_sample_tasks,
    has_circular_dependencies
)

from .state_manager import ExecutorState
from .feedback_loop import FeedbackLoop

__all__ = [
    'Task',
    'TasksPlan',
    'AcceptanceCriteria',
    'TechnicalDetails',
    'TestCriteria',
    'TaskSchemaValidator',
    'TASKS_JSON_SCHEMA',
    'create_sample_tasks',
    'has_circular_dependencies',
    'ExecutorState',
    'FeedbackLoop'
]