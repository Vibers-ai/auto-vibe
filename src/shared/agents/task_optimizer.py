"""Task Optimization Module for VIBE

This module provides advanced task generation and dependency optimization
capabilities for the Master Planner.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1  # Must be done first (e.g., setup, core infrastructure)
    HIGH = 2      # Important for core functionality
    MEDIUM = 3    # Standard features
    LOW = 4       # Nice-to-have features


@dataclass
class OptimizedTask:
    """Enhanced task representation with optimization metadata."""
    id: str
    description: str
    type: str
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_hours: float = 2.0
    parallelizable: bool = True
    critical_path: bool = False
    project_area: str = "shared"
    files_to_create_or_modify: List[str] = field(default_factory=list)
    
    # Optimization metadata
    earliest_start: int = 0  # Earliest this task can start
    latest_start: int = 0    # Latest this task can start without delaying project
    slack: int = 0           # Amount of time this task can be delayed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type,
            "dependencies": self.dependencies,
            "status": "pending",
            "project_area": self.project_area,
            "files_to_create_or_modify": self.files_to_create_or_modify,
            "acceptance_criteria": {
                "tests": [],
                "linting": {},
                "manual_checks": []
            },
            "estimated_hours": self.estimated_hours,
            "technical_details": {
                "priority": self.priority.name,
                "parallelizable": self.parallelizable,
                "critical_path": self.critical_path
            }
        }


class TaskOptimizer:
    """Optimizes task generation and dependencies."""
    
    def __init__(self):
        self.task_templates = self._initialize_task_templates()
        self.dependency_patterns = self._initialize_dependency_patterns()
    
    def _initialize_task_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize common task templates."""
        return {
            "setup": [
                {
                    "id": "setup-project-structure",
                    "description": "Initialize project directory structure and configuration files",
                    "type": "setup",
                    "priority": TaskPriority.CRITICAL,
                    "estimated_hours": 1.0,
                    "files": ["package.json", "tsconfig.json", ".env.example", "README.md"]
                },
                {
                    "id": "setup-git-workflow",
                    "description": "Configure Git hooks, branch protection, and CI/CD workflows",
                    "type": "setup",
                    "priority": TaskPriority.CRITICAL,
                    "estimated_hours": 1.0,
                    "files": [".gitignore", ".github/workflows/ci.yml", ".husky/"]
                }
            ],
            "backend_core": [
                {
                    "id": "setup-backend-server",
                    "description": "Create Express/Fastify server with middleware configuration",
                    "type": "backend",
                    "priority": TaskPriority.CRITICAL,
                    "estimated_hours": 2.0,
                    "dependencies": ["setup-project-structure"],
                    "files": ["backend/src/server.ts", "backend/src/middleware/"]
                },
                {
                    "id": "setup-database-connection",
                    "description": "Configure database connection and ORM setup",
                    "type": "backend",
                    "priority": TaskPriority.CRITICAL,
                    "estimated_hours": 2.0,
                    "dependencies": ["setup-backend-server"],
                    "files": ["backend/src/database/", "backend/prisma/schema.prisma"]
                },
                {
                    "id": "implement-auth-system",
                    "description": "Implement JWT-based authentication and authorization",
                    "type": "backend",
                    "priority": TaskPriority.HIGH,
                    "estimated_hours": 4.0,
                    "dependencies": ["setup-database-connection"],
                    "files": ["backend/src/auth/", "backend/src/middleware/auth.ts"]
                }
            ],
            "frontend_core": [
                {
                    "id": "setup-frontend-framework",
                    "description": "Initialize Next.js/React application with routing",
                    "type": "frontend",
                    "priority": TaskPriority.CRITICAL,
                    "estimated_hours": 2.0,
                    "dependencies": ["setup-project-structure"],
                    "files": ["frontend/src/app/layout.tsx", "frontend/src/app/page.tsx"]
                },
                {
                    "id": "setup-ui-components",
                    "description": "Create base UI components and design system",
                    "type": "frontend",
                    "priority": TaskPriority.HIGH,
                    "estimated_hours": 3.0,
                    "dependencies": ["setup-frontend-framework"],
                    "files": ["frontend/src/components/ui/", "frontend/src/styles/"]
                },
                {
                    "id": "implement-state-management",
                    "description": "Set up global state management (Zustand/Redux)",
                    "type": "frontend",
                    "priority": TaskPriority.HIGH,
                    "estimated_hours": 2.0,
                    "dependencies": ["setup-frontend-framework"],
                    "files": ["frontend/src/store/", "frontend/src/hooks/"]
                }
            ],
            "api": [
                {
                    "id": "create-api-routes",
                    "description": "Implement RESTful API endpoints",
                    "type": "backend",
                    "priority": TaskPriority.HIGH,
                    "estimated_hours": 3.0,
                    "dependencies": ["setup-backend-server", "setup-database-connection"],
                    "files": ["backend/src/routes/", "backend/src/controllers/"]
                }
            ],
            "testing": [
                {
                    "id": "setup-testing-framework",
                    "description": "Configure Jest, React Testing Library, and Cypress",
                    "type": "testing",
                    "priority": TaskPriority.MEDIUM,
                    "estimated_hours": 2.0,
                    "dependencies": ["setup-project-structure"],
                    "files": ["jest.config.js", "cypress.config.ts", "tests/"]
                },
                {
                    "id": "write-unit-tests",
                    "description": "Write unit tests for core business logic",
                    "type": "testing",
                    "priority": TaskPriority.MEDIUM,
                    "estimated_hours": 4.0,
                    "dependencies": ["create-api-routes", "implement-auth-system"],
                    "files": ["backend/src/__tests__/", "frontend/src/__tests__/"]
                }
            ]
        }
    
    def _initialize_dependency_patterns(self) -> Dict[str, List[str]]:
        """Initialize common dependency patterns."""
        return {
            "database_dependent": ["setup-database-connection"],
            "auth_dependent": ["implement-auth-system"],
            "api_dependent": ["create-api-routes"],
            "frontend_data": ["create-api-routes", "setup-frontend-framework"],
            "testing": ["setup-testing-framework"]
        }
    
    def optimize_task_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize task dependencies using graph analysis."""
        # Convert to OptimizedTask objects
        optimized_tasks = []
        task_map = {}
        
        for task in tasks:
            opt_task = OptimizedTask(
                id=task["id"],
                description=task["description"],
                type=task.get("type", "general"),
                dependencies=task.get("dependencies", []),
                project_area=task.get("project_area", "shared"),
                files_to_create_or_modify=task.get("files_to_create_or_modify", [])
            )
            
            # Infer priority
            opt_task.priority = self._infer_priority(opt_task)
            
            optimized_tasks.append(opt_task)
            task_map[opt_task.id] = opt_task
        
        # Build dependency graph
        graph = self._build_dependency_graph(optimized_tasks)
        
        # Find critical path
        critical_path_tasks = self._find_critical_path(graph, task_map)
        for task_id in critical_path_tasks:
            task_map[task_id].critical_path = True
        
        # Optimize dependencies
        self._optimize_dependencies(graph, task_map)
        
        # Calculate scheduling metrics
        self._calculate_scheduling_metrics(graph, task_map)
        
        # Convert back to dictionaries
        return [task.to_dict() for task in optimized_tasks]
    
    def _infer_priority(self, task: OptimizedTask) -> TaskPriority:
        """Infer task priority based on type and description."""
        desc_lower = task.description.lower()
        
        # Critical tasks
        if task.type == "setup" or "initialize" in desc_lower or "configure" in desc_lower:
            return TaskPriority.CRITICAL
        
        # High priority tasks
        if any(keyword in desc_lower for keyword in ["auth", "security", "database", "api", "core"]):
            return TaskPriority.HIGH
        
        # Low priority tasks
        if any(keyword in desc_lower for keyword in ["optional", "enhancement", "polish", "styling"]):
            return TaskPriority.LOW
        
        return TaskPriority.MEDIUM
    
    def _build_dependency_graph(self, tasks: List[OptimizedTask]) -> nx.DiGraph:
        """Build a directed graph of task dependencies."""
        graph = nx.DiGraph()
        
        # Add nodes
        for task in tasks:
            graph.add_node(task.id, task=task)
        
        # Add edges (dependencies)
        for task in tasks:
            for dep in task.dependencies:
                if dep in graph.nodes:
                    graph.add_edge(dep, task.id)
        
        return graph
    
    def _find_critical_path(self, graph: nx.DiGraph, task_map: Dict[str, OptimizedTask]) -> Set[str]:
        """Find the critical path in the project."""
        if not graph.nodes:
            return set()
        
        # Find start and end nodes
        start_nodes = [n for n in graph.nodes if graph.in_degree(n) == 0]
        end_nodes = [n for n in graph.nodes if graph.out_degree(n) == 0]
        
        if not start_nodes or not end_nodes:
            return set()
        
        # Calculate longest path (critical path)
        critical_path_nodes = set()
        
        for start in start_nodes:
            for end in end_nodes:
                try:
                    paths = list(nx.all_simple_paths(graph, start, end))
                    if paths:
                        # Find longest path by total estimated hours
                        longest_path = max(paths, key=lambda p: sum(
                            task_map[node].estimated_hours for node in p
                        ))
                        critical_path_nodes.update(longest_path)
                except nx.NetworkXNoPath:
                    continue
        
        return critical_path_nodes
    
    def _optimize_dependencies(self, graph: nx.DiGraph, task_map: Dict[str, OptimizedTask]):
        """Optimize dependencies to enable parallelization."""
        # Identify tasks that can be parallelized
        for node in graph.nodes:
            task = task_map[node]
            
            # Check if task has siblings (same dependencies)
            siblings = []
            for other_node in graph.nodes:
                if other_node != node:
                    other_task = task_map[other_node]
                    if set(task.dependencies) == set(other_task.dependencies):
                        siblings.append(other_node)
            
            # Mark as parallelizable if has siblings and not on critical path
            if siblings and not task.critical_path:
                task.parallelizable = True
                for sibling in siblings:
                    task_map[sibling].parallelizable = True
    
    def _calculate_scheduling_metrics(self, graph: nx.DiGraph, task_map: Dict[str, OptimizedTask]):
        """Calculate scheduling metrics for each task."""
        # Topological sort for scheduling
        try:
            topo_order = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            logger.warning("Circular dependencies detected, skipping scheduling metrics")
            return
        
        # Forward pass - calculate earliest start times
        for task_id in topo_order:
            task = task_map[task_id]
            if not task.dependencies:
                task.earliest_start = 0
            else:
                # Earliest start is the maximum of all dependencies' completion times
                max_completion = 0
                for dep_id in task.dependencies:
                    if dep_id in task_map:
                        dep_task = task_map[dep_id]
                        completion_time = dep_task.earliest_start + int(dep_task.estimated_hours)
                        max_completion = max(max_completion, completion_time)
                task.earliest_start = max_completion
        
        # Backward pass - calculate latest start times
        # First, find the project completion time
        project_completion = max(
            task.earliest_start + int(task.estimated_hours) 
            for task in task_map.values()
        )
        
        # Process in reverse topological order
        for task_id in reversed(topo_order):
            task = task_map[task_id]
            
            # Find successors
            successors = list(graph.successors(task_id))
            
            if not successors:
                # No successors, can start as late as possible
                task.latest_start = project_completion - int(task.estimated_hours)
            else:
                # Latest start is constrained by successors
                min_successor_start = min(
                    task_map[succ].latest_start 
                    for succ in successors 
                    if succ in task_map
                )
                task.latest_start = min_successor_start - int(task.estimated_hours)
            
            # Calculate slack
            task.slack = task.latest_start - task.earliest_start
    
    def generate_optimized_tasks(self, project_type: str, tech_stack: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate optimized tasks based on project type and tech stack."""
        tasks = []
        
        # Always include setup tasks
        for template in self.task_templates["setup"]:
            task = OptimizedTask(
                id=template["id"],
                description=template["description"],
                type=template["type"],
                priority=template["priority"],
                estimated_hours=template["estimated_hours"],
                files_to_create_or_modify=template["files"]
            )
            tasks.append(task)
        
        # Add backend core tasks if backend exists
        if tech_stack.get("backend"):
            for template in self.task_templates["backend_core"]:
                task = OptimizedTask(
                    id=template["id"],
                    description=template["description"],
                    type=template["type"],
                    priority=template["priority"],
                    estimated_hours=template["estimated_hours"],
                    dependencies=template.get("dependencies", []),
                    files_to_create_or_modify=template["files"],
                    project_area="backend"
                )
                tasks.append(task)
        
        # Add frontend core tasks if frontend exists
        if tech_stack.get("frontend"):
            for template in self.task_templates["frontend_core"]:
                task = OptimizedTask(
                    id=template["id"],
                    description=template["description"],
                    type=template["type"],
                    priority=template["priority"],
                    estimated_hours=template["estimated_hours"],
                    dependencies=template.get("dependencies", []),
                    files_to_create_or_modify=template["files"],
                    project_area="frontend"
                )
                tasks.append(task)
        
        # Add API tasks if needed
        if tech_stack.get("backend") and project_type in ["web_app", "e-commerce", "saas"]:
            for template in self.task_templates["api"]:
                task = OptimizedTask(
                    id=template["id"],
                    description=template["description"],
                    type=template["type"],
                    priority=template["priority"],
                    estimated_hours=template["estimated_hours"],
                    dependencies=template.get("dependencies", []),
                    files_to_create_or_modify=template["files"],
                    project_area="backend"
                )
                tasks.append(task)
        
        # Add testing tasks
        for template in self.task_templates["testing"]:
            task = OptimizedTask(
                id=template["id"],
                description=template["description"],
                type=template["type"],
                priority=template["priority"],
                estimated_hours=template["estimated_hours"],
                dependencies=template.get("dependencies", []),
                files_to_create_or_modify=template["files"],
                project_area="testing"
            )
            tasks.append(task)
        
        # Convert to dictionaries and optimize
        task_dicts = [task.to_dict() for task in tasks]
        return self.optimize_task_dependencies(task_dicts)


# Singleton instance
task_optimizer = TaskOptimizer()