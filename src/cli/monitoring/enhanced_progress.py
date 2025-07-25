"""Enhanced Progress Monitoring for CLI Execution."""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.progress import (
    Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn,
    SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn
)
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich.tree import Tree

console = Console()


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    WAITING = "waiting"  # Waiting for dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Individual task progress information."""
    task_id: str
    description: str
    status: TaskStatus
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get task duration."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def is_active(self) -> bool:
        """Check if task is currently active."""
        return self.status in [TaskStatus.RUNNING, TaskStatus.WAITING]
    
    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]


@dataclass
class ParallelWorker:
    """Parallel worker information."""
    worker_id: str
    status: str = "idle"  # idle, busy, error
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_activity: Optional[datetime] = None


class EnhancedProgressMonitor:
    """Enhanced progress monitoring with parallel task visualization."""
    
    def __init__(self, max_workers: int = 4, update_interval: float = 0.5):
        self.max_workers = max_workers
        self.update_interval = update_interval
        
        # Task tracking
        self.tasks: Dict[str, TaskProgress] = {}
        self.workers: Dict[str, ParallelWorker] = {}
        self.task_queue: List[str] = []
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Progress bars
        self.main_progress = None
        self.task_progress = None
        self.main_task_id = None
        
        # UI components
        self.layout = None
        self.live = None
        self.is_running = False
        
        # Statistics
        self.start_time = None
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Callbacks
        self.status_callback: Optional[Callable] = None
        
        # Initialize workers
        for i in range(max_workers):
            worker_id = f"worker-{i+1}"
            self.workers[worker_id] = ParallelWorker(worker_id=worker_id)
    
    def add_task(self, task_id: str, description: str, dependencies: List[str] = None) -> None:
        """Add a task to monitor."""
        self.tasks[task_id] = TaskProgress(
            task_id=task_id,
            description=description,
            status=TaskStatus.PENDING,
            dependencies=dependencies or []
        )
        self.task_queue.append(task_id)
        self.total_tasks += 1
        
        # Update dependency graph
        self.dependency_graph[task_id] = dependencies or []
    
    def start_monitoring(self, title: str = "Task Execution Progress") -> None:
        """Start the enhanced progress monitoring."""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_time = datetime.now()
        
        # Create progress bars
        self.main_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )
        
        self.task_progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.fields[status]}"),
        )
        
        # Add main task
        self.main_task_id = self.main_progress.add_task(
            title, total=self.total_tasks
        )
        
        # Create layout
        self.layout = self._create_layout()
        
        # Start live display
        self.live = Live(self.layout, refresh_per_second=2, console=console)
        self.live.start()
        
        # Start background update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          progress: float = None, error: str = None,
                          worker_id: str = None) -> None:
        """Update task status."""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        old_status = task.status
        task.status = status
        
        if progress is not None:
            task.progress = min(100.0, max(0.0, progress))
        
        if error:
            task.error = error
            
        if worker_id:
            task.worker_id = worker_id
            
        # Update timestamps
        if status == TaskStatus.RUNNING and old_status != TaskStatus.RUNNING:
            task.start_time = datetime.now()
            if worker_id and worker_id in self.workers:
                self.workers[worker_id].status = "busy"
                self.workers[worker_id].current_task = task_id
                self.workers[worker_id].last_activity = datetime.now()
                
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.end_time = datetime.now()
            if task.progress < 100 and status == TaskStatus.COMPLETED:
                task.progress = 100.0
                
            # Update worker status
            if worker_id and worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.status = "idle"
                worker.current_task = None
                worker.last_activity = datetime.now()
                
                if status == TaskStatus.COMPLETED:
                    worker.tasks_completed += 1
                elif status == TaskStatus.FAILED:
                    worker.tasks_failed += 1
            
            # Update global counters
            if old_status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                if status == TaskStatus.COMPLETED:
                    self.completed_tasks += 1
                elif status == TaskStatus.FAILED:
                    self.failed_tasks += 1
                    
        # Call status callback
        if self.status_callback:
            self.status_callback(task_id, task)
    
    def update_task_progress(self, task_id: str, progress: float, 
                           status_message: str = None) -> None:
        """Update task progress percentage."""
        if task_id in self.tasks:
            self.tasks[task_id].progress = min(100.0, max(0.0, progress))
            if status_message:
                # Could store status message in task if needed
                pass
    
    def get_parallel_status(self) -> Dict[str, Any]:
        """Get current parallel execution status."""
        active_tasks = [t for t in self.tasks.values() if t.is_active]
        busy_workers = [w for w in self.workers.values() if w.status == "busy"]
        
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'active_tasks': len(active_tasks),
            'busy_workers': len(busy_workers),
            'idle_workers': self.max_workers - len(busy_workers),
            'parallelization_efficiency': len(busy_workers) / self.max_workers if self.max_workers > 0 else 0,
            'active_task_details': [
                {
                    'task_id': t.task_id,
                    'description': t.description,
                    'progress': t.progress,
                    'worker_id': t.worker_id,
                    'duration': str(t.duration) if t.duration else None
                } for t in active_tasks
            ]
        }
    
    def stop_monitoring(self) -> None:
        """Stop the progress monitoring."""
        self.is_running = False
        
        if self.live:
            self.live.stop()
            self.live = None
        
        # Display final summary
        self._display_final_summary()
    
    def _create_layout(self) -> Layout:
        """Create the layout for live display."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        
        layout["main"].split_column(
            Layout(name="top", size=15),
            Layout(name="bottom")
        )
        
        layout["top"].split_row(
            Layout(name="progress"),
            Layout(name="dependencies")
        )
        
        layout["bottom"].split_row(
            Layout(name="tasks"),
            Layout(name="workers")
        )
        
        return layout
    
    def _update_loop(self) -> None:
        """Background update loop."""
        while self.is_running:
            try:
                self._update_display()
                time.sleep(self.update_interval)
            except Exception as e:
                # Silently handle display update errors
                pass
    
    def _update_display(self) -> None:
        """Update the live display."""
        if not self.layout or not self.is_running:
            return
            
        try:
            # Update header
            self.layout["header"].update(self._create_header())
            
            # Update main progress
            self.layout["progress"].update(Panel(
                self.main_progress,
                title="[bold blue]Overall Progress",
                border_style="blue"
            ))
            
            # Update dependency visualization
            self.layout["dependencies"].update(Panel(
                self._create_dependency_tree(),
                title="[bold cyan]Task Dependencies",
                border_style="cyan"
            ))
            
            # Update task details
            self.layout["tasks"].update(Panel(
                self._create_task_details(),
                title="[bold green]Active Tasks",
                border_style="green"
            ))
            
            # Update workers
            self.layout["workers"].update(Panel(
                self._create_workers_table(),
                title="[bold yellow]Workers",
                border_style="yellow"
            ))
            
            # Update main progress bar
            if self.main_task_id is not None:
                completed = self.completed_tasks + self.failed_tasks
                self.main_progress.update(
                    self.main_task_id,
                    completed=completed,
                    description=f"Executing tasks ({completed}/{self.total_tasks})"
                )
                
        except Exception:
            # Silently handle display errors
            pass
    
    def _create_header(self) -> Panel:
        """Create header panel with summary stats."""
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        stats_text = Text()
        stats_text.append("⏱️  ", style="blue")
        stats_text.append(f"Elapsed: {str(elapsed).split('.')[0]}", style="white")
        stats_text.append("  |  ", style="dim")
        stats_text.append("✅ ", style="green")
        stats_text.append(f"Completed: {self.completed_tasks}", style="green")
        stats_text.append("  |  ", style="dim")
        stats_text.append("❌ ", style="red")
        stats_text.append(f"Failed: {self.failed_tasks}", style="red")
        stats_text.append("  |  ", style="dim")
        stats_text.append("⚡ ", style="yellow")
        
        active_count = len([t for t in self.tasks.values() if t.is_active])
        stats_text.append(f"Active: {active_count}", style="yellow")
        
        return Panel(
            Align.center(stats_text),
            border_style="bright_blue"
        )
    
    def _create_task_details(self) -> Table:
        """Create task details table."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Task ID", style="cyan", width=15)
        table.add_column("Status", width=10)
        table.add_column("Progress", width=12)
        table.add_column("Worker", width=8)
        table.add_column("Duration", width=10)
        
        # Show active tasks first, then recent completed/failed
        active_tasks = [t for t in self.tasks.values() if t.is_active]
        recent_tasks = [
            t for t in self.tasks.values() 
            if t.is_complete and t.end_time and 
            (datetime.now() - t.end_time).seconds < 30
        ]
        
        display_tasks = active_tasks + recent_tasks[:5-len(active_tasks)]
        
        for task in display_tasks:
            # Status with icon
            if task.status == TaskStatus.RUNNING:
                status_text = "[yellow]🔄 Running[/yellow]"
            elif task.status == TaskStatus.WAITING:
                status_text = "[blue]⏳ Waiting[/blue]"
            elif task.status == TaskStatus.COMPLETED:
                status_text = "[green]✅ Done[/green]"
            elif task.status == TaskStatus.FAILED:
                status_text = "[red]❌ Failed[/red]"
            else:
                status_text = "[dim]⏸️  Pending[/dim]"
            
            # Progress bar
            progress_text = f"{task.progress:5.1f}%"
            if task.is_active:
                progress_text = f"[yellow]{progress_text}[/yellow]"
            elif task.status == TaskStatus.COMPLETED:
                progress_text = f"[green]{progress_text}[/green]"
            elif task.status == TaskStatus.FAILED:
                progress_text = f"[red]{progress_text}[/red]"
            
            # Worker info
            worker_text = task.worker_id or "-"
            
            # Duration
            duration_text = str(task.duration).split('.')[0] if task.duration else "-"
            
            table.add_row(
                task.task_id[:15],
                status_text,
                progress_text,
                worker_text,
                duration_text
            )
        
        return table
    
    def _create_dependency_tree(self) -> Tree:
        """Create dependency tree visualization."""
        tree = Tree("📋 Task Dependencies", style="bold cyan")
        
        # Find root tasks (no dependencies)
        root_tasks = [
            task_id for task_id, deps in self.dependency_graph.items() 
            if not deps
        ]
        
        # Build tree recursively
        added_to_tree = set()
        
        def add_task_to_tree(parent, task_id, depth=0):
            if task_id in added_to_tree or depth > 3:  # Prevent infinite loops and too deep
                return
                
            added_to_tree.add(task_id)
            task = self.tasks.get(task_id)
            if not task:
                return
                
            # Create task display text with status
            if task.status == TaskStatus.COMPLETED:
                icon = "✅"
                style = "green"
            elif task.status == TaskStatus.RUNNING:
                icon = "🔄"
                style = "yellow"
            elif task.status == TaskStatus.FAILED:
                icon = "❌"
                style = "red"
            elif task.status == TaskStatus.WAITING:
                icon = "⏳"
                style = "blue"
            else:
                icon = "⏸️"
                style = "dim"
            
            task_text = f"{icon} {task_id[:12]}{'...' if len(task_id) > 12 else ''}"
            task_node = parent.add(task_text, style=style)
            
            # Add dependent tasks
            dependent_tasks = [
                tid for tid, deps in self.dependency_graph.items() 
                if task_id in deps
            ]
            
            for dep_task in dependent_tasks[:3]:  # Limit to prevent overcrowding
                add_task_to_tree(task_node, dep_task, depth + 1)
        
        # Add root tasks and their dependencies
        for root_task in root_tasks[:5]:  # Limit root tasks shown
            add_task_to_tree(tree, root_task)
        
        # Show orphaned tasks (tasks with dependencies not in our graph)
        orphaned = [
            task_id for task_id in self.tasks.keys() 
            if task_id not in added_to_tree
        ]
        
        if orphaned:
            orphan_node = tree.add("🔗 Other Tasks", style="dim")
            for orphan in orphaned[:3]:  # Limit shown
                task = self.tasks[orphan]
                if task.status == TaskStatus.COMPLETED:
                    icon = "✅"
                elif task.status == TaskStatus.RUNNING:
                    icon = "🔄"
                elif task.status == TaskStatus.FAILED:
                    icon = "❌"
                else:
                    icon = "⏸️"
                orphan_node.add(f"{icon} {orphan[:12]}")
        
        return tree
    
    def _create_workers_table(self) -> Table:
        """Create simplified workers table for compact display."""
        table = Table(show_header=True, header_style="bold yellow", box=None)
        table.add_column("Worker", style="cyan", width=10)
        table.add_column("Status", width=8)
        table.add_column("Tasks", width=6)
        
        for worker in self.workers.values():
            # Status with icon
            if worker.status == "busy":
                status_text = "[yellow]🔄 Busy[/yellow]"
            elif worker.status == "idle":
                status_text = "[green]💤 Idle[/green]"
            else:
                status_text = "[red]❌ Error[/red]"
            
            # Task count
            task_count = f"{worker.tasks_completed}"
            if worker.tasks_failed > 0:
                task_count += f"/{worker.tasks_failed}F"
            
            table.add_row(
                worker.worker_id.replace("worker-", "W"),
                status_text,
                task_count
            )
        
        return table
    
    def _create_workers_panel(self) -> Panel:
        """Create workers status panel."""
        workers_table = Table(show_header=True, header_style="bold cyan")
        workers_table.add_column("Worker", style="cyan")
        workers_table.add_column("Status", width=10)
        workers_table.add_column("Current Task", width=20)
        workers_table.add_column("Completed", width=10)
        workers_table.add_column("Failed", width=8)
        workers_table.add_column("Last Activity", width=12)
        
        for worker in self.workers.values():
            # Status with color
            if worker.status == "busy":
                status_text = "[yellow]🔄 Busy[/yellow]"
            elif worker.status == "idle":
                status_text = "[green]💤 Idle[/green]"
            else:
                status_text = "[red]❌ Error[/red]"
            
            # Current task
            current_task = worker.current_task[:18] + "..." if worker.current_task and len(worker.current_task) > 20 else (worker.current_task or "-")
            
            # Last activity
            activity_text = "-"
            if worker.last_activity:
                delta = datetime.now() - worker.last_activity
                if delta.seconds < 60:
                    activity_text = f"{delta.seconds}s ago"
                else:
                    activity_text = f"{delta.seconds//60}m ago"
            
            workers_table.add_row(
                worker.worker_id,
                status_text,
                current_task,
                str(worker.tasks_completed),
                str(worker.tasks_failed),
                activity_text
            )
        
        return Panel(
            workers_table,
            title="[bold cyan]Parallel Workers Status",
            border_style="cyan"
        )
    
    def _display_final_summary(self) -> None:
        """Display final execution summary."""
        if not self.start_time:
            return
            
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        console.print("\n" + "="*80)
        console.print("[bold blue]📊 ENHANCED EXECUTION SUMMARY[/bold blue]")
        console.print("="*80)
        
        # Create summary table
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Metric", style="bold cyan", width=25)
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Total Duration", str(total_duration).split('.')[0])
        summary_table.add_row("Total Tasks", str(self.total_tasks))
        summary_table.add_row("Completed Tasks", f"[green]{self.completed_tasks}[/green]")
        summary_table.add_row("Failed Tasks", f"[red]{self.failed_tasks}[/red]")
        
        success_rate = (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
        success_color = "green" if success_rate >= 80 else "yellow" if success_rate >= 60 else "red"
        summary_table.add_row("Success Rate", f"[{success_color}]{success_rate:.1f}%[/{success_color}]")
        
        # Calculate average task duration
        completed_tasks_with_duration = [
            t for t in self.tasks.values() 
            if t.status == TaskStatus.COMPLETED and t.duration
        ]
        if completed_tasks_with_duration:
            avg_duration = sum([t.duration.total_seconds() for t in completed_tasks_with_duration]) / len(completed_tasks_with_duration)
            summary_table.add_row("Avg Task Duration", f"{avg_duration:.1f}s")
        
        # Worker efficiency
        total_worker_tasks = sum(w.tasks_completed + w.tasks_failed for w in self.workers.values())
        if total_worker_tasks > 0:
            worker_efficiency = (total_worker_tasks / (self.max_workers * total_duration.total_seconds() / 60)) * 100
            summary_table.add_row("Worker Efficiency", f"{min(100, worker_efficiency):.1f}%")
        
        console.print(summary_table)
        
        # Worker summary
        console.print("\n[bold cyan]Worker Performance:[/bold cyan]")
        for worker in self.workers.values():
            total_worker_tasks = worker.tasks_completed + worker.tasks_failed
            worker_success_rate = (worker.tasks_completed / total_worker_tasks * 100) if total_worker_tasks > 0 else 0
            console.print(f"  {worker.worker_id}: {worker.tasks_completed} completed, {worker.tasks_failed} failed ({worker_success_rate:.1f}% success)")
        
        console.print("="*80)
    
    def set_status_callback(self, callback: Callable[[str, TaskProgress], None]) -> None:
        """Set callback for task status changes."""
        self.status_callback = callback