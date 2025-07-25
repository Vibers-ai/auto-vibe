"""VIBE - Autonomous Coding Agent

VIBE transforms project documentation into complete, working software applications
using AI agents specialized in planning and code execution.
"""

__version__ = "0.1.0"
__author__ = "VIBE AI"

from .shared.agents.document_ingestion import DocumentIngestionAgent
from .shared.agents.master_planner import MasterPlannerAgent
from .shared.core.schema import Task, TasksPlan
from .shared.utils.config import Config

# CLI executor는 필요시 별도 import
# from .cli.core.executor_cli import TaskExecutorCli

__all__ = [
    'DocumentIngestionAgent',
    'MasterPlannerAgent', 
    'Task',
    'TasksPlan',
    'Config'
]