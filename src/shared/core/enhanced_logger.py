"""Enhanced Logging System for VIBE with structured debugging support."""

import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import traceback

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel


class LogLevel(Enum):
    """Enhanced log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TASK = "TASK"  # Task-specific logging
    WORKSPACE = "WORKSPACE"  # Workspace operation logging
    CLAUDE = "CLAUDE"  # Claude CLI interaction logging


class LogCategory(Enum):
    """Log categories for better organization."""
    TASK_EXECUTION = "task_execution"
    WORKSPACE_ANALYSIS = "workspace_analysis"
    CLAUDE_INTERACTION = "claude_interaction"
    STRUCTURE_VALIDATION = "structure_validation"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    ERROR_RECOVERY = "error_recovery"
    PERFORMANCE = "performance"
    SYSTEM = "system"


@dataclass
class StructuredLogEntry:
    """Structured log entry with rich metadata."""
    timestamp: str
    level: str
    category: str
    component: str
    message: str
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    workspace_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    stack_trace: Optional[str] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ContextualLogger:
    """Logger with contextual information tracking."""
    
    def __init__(self, component: str, task_id: Optional[str] = None, 
                 session_id: Optional[str] = None, workspace_path: Optional[str] = None):
        self.component = component
        self.task_id = task_id
        self.session_id = session_id
        self.workspace_path = workspace_path
        self._start_time = None
    
    def start_operation(self) -> None:
        """Start timing an operation."""
        self._start_time = datetime.now()
    
    def end_operation(self) -> float:
        """End timing and return duration in milliseconds."""
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds() * 1000
            self._start_time = None
            return duration
        return 0.0
    
    def log(self, level: LogLevel, category: LogCategory, message: str, 
            metadata: Optional[Dict[str, Any]] = None, 
            include_stack: bool = False) -> None:
        """Log with full context."""
        
        entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            category=category.value,
            component=self.component,
            message=message,
            task_id=self.task_id,
            session_id=self.session_id,
            workspace_path=self.workspace_path,
            metadata=metadata or {},
            duration_ms=self.end_operation() if self._start_time else None
        )
        
        if include_stack:
            entry.stack_trace = traceback.format_exc()
        
        # Log to enhanced logger
        enhanced_logger.log_structured(entry)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
              metadata: Optional[Dict[str, Any]] = None) -> None:
        self.log(LogLevel.DEBUG, category, message, metadata)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM,
             metadata: Optional[Dict[str, Any]] = None) -> None:
        self.log(LogLevel.INFO, category, message, metadata)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                metadata: Optional[Dict[str, Any]] = None) -> None:
        self.log(LogLevel.WARNING, category, message, metadata)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM,
              metadata: Optional[Dict[str, Any]] = None, include_stack: bool = True) -> None:
        self.log(LogLevel.ERROR, category, message, metadata, include_stack)
    
    def task_start(self, task_description: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log task start."""
        self.start_operation()
        self.log(LogLevel.TASK, LogCategory.TASK_EXECUTION, 
                f"Starting task: {task_description}", metadata)
    
    def task_complete(self, task_description: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log task completion."""
        self.log(LogLevel.TASK, LogCategory.TASK_EXECUTION, 
                f"Completed task: {task_description}", metadata)
    
    def task_failed(self, task_description: str, error: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log task failure."""
        meta = metadata or {}
        meta['error'] = error
        self.log(LogLevel.ERROR, LogCategory.TASK_EXECUTION, 
                f"Task failed: {task_description}", meta, include_stack=True)
    
    def workspace_operation(self, operation: str, path: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log workspace operations."""
        meta = metadata or {}
        meta['operation'] = operation
        meta['path'] = path
        self.log(LogLevel.WORKSPACE, LogCategory.WORKSPACE_ANALYSIS, 
                f"Workspace {operation}: {path}", meta)
    
    def claude_interaction(self, interaction_type: str, prompt_length: int,
                         response_length: int = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log Claude CLI interactions."""
        meta = metadata or {}
        meta['interaction_type'] = interaction_type
        meta['prompt_length'] = prompt_length
        if response_length is not None:
            meta['response_length'] = response_length
        
        self.log(LogLevel.CLAUDE, LogCategory.CLAUDE_INTERACTION, 
                f"Claude {interaction_type}: prompt={prompt_length} chars", meta)


class EnhancedLogger:
    """Enhanced logging system with structured output and debugging support."""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Console for rich output
        self.console = Console()
        
        # Session-specific log file
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.logs_dir / f"session_{self.session_timestamp}.jsonl"
        self.debug_log_file = self.logs_dir / f"debug_{self.session_timestamp}.log"
        
        # In-memory log buffer for debugging
        self.log_buffer: List[StructuredLogEntry] = []
        self.max_buffer_size = 1000
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup Python logging integration
        self._setup_python_logging()
        
        # Track component loggers
        self.component_loggers: Dict[str, ContextualLogger] = {}
    
    def _setup_python_logging(self) -> None:
        """Setup integration with Python's logging module."""
        
        # Create custom formatter
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                # Create structured entry from Python log record
                entry = StructuredLogEntry(
                    timestamp=datetime.fromtimestamp(record.created).isoformat(),
                    level=record.levelname,
                    category=LogCategory.SYSTEM.value,
                    component=record.name,
                    message=record.getMessage(),
                    metadata={'module': record.module, 'function': record.funcName, 'line': record.lineno}
                )
                
                if record.exc_info:
                    entry.stack_trace = self.formatException(record.exc_info)
                
                enhanced_logger.log_structured(entry)
                return super().format(record)
        
        # Setup file handler for debug logs
        file_handler = logging.FileHandler(self.debug_log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        
        # Setup rich console handler
        console_handler = RichHandler(
            console=self.console, 
            show_level=True, 
            show_time=True,
            rich_tracebacks=True
        )
        console_handler.setLevel(logging.INFO)
        
        # Get root logger and configure
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add our handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def log_structured(self, entry: StructuredLogEntry) -> None:
        """Log a structured entry."""
        with self._lock:
            # Add to buffer
            self.log_buffer.append(entry)
            if len(self.log_buffer) > self.max_buffer_size:
                self.log_buffer.pop(0)
            
            # Write to JSON Lines file
            try:
                with open(self.session_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry.to_dict()) + '\n')
            except Exception as e:
                # Fallback to console if file writing fails
                self.console.print(f"[red]Failed to write log: {e}[/red]")
            
            # Display in console for important messages
            self._display_console_message(entry)
    
    def _display_console_message(self, entry: StructuredLogEntry) -> None:
        """Display message in console based on level and category."""
        
        # Skip debug messages unless explicitly enabled
        if entry.level == LogLevel.DEBUG.value and not os.getenv('VIBE_DEBUG'):
            return
        
        # Format message based on category
        if entry.category == LogCategory.TASK_EXECUTION.value:
            if entry.level == LogLevel.TASK.value:
                icon = "🎯" if "Starting" in entry.message else "✅"
                self.console.print(f"[blue]{icon} {entry.message}[/blue]")
            elif entry.level == LogLevel.ERROR.value:
                self.console.print(f"[red]❌ {entry.message}[/red]")
        
        elif entry.category == LogCategory.WORKSPACE_ANALYSIS.value:
            if entry.level == LogLevel.WORKSPACE.value:
                self.console.print(f"[cyan]📁 {entry.message}[/cyan]")
            elif entry.level == LogLevel.WARNING.value:
                self.console.print(f"[yellow]⚠️ {entry.message}[/yellow]")
        
        elif entry.category == LogCategory.CLAUDE_INTERACTION.value:
            if entry.level == LogLevel.CLAUDE.value:
                self.console.print(f"[magenta]🤖 {entry.message}[/magenta]")
        
        elif entry.category == LogCategory.STRUCTURE_VALIDATION.value:
            if entry.level == LogLevel.ERROR.value:
                self.console.print(f"[red]🏗️ Structure Issue: {entry.message}[/red]")
            elif entry.level == LogLevel.WARNING.value:
                self.console.print(f"[yellow]🏗️ Structure Warning: {entry.message}[/yellow]")
        
        # Always show errors and critical messages
        elif entry.level in [LogLevel.ERROR.value, LogLevel.CRITICAL.value]:
            style = "red" if entry.level == LogLevel.ERROR.value else "bold red"
            self.console.print(f"[{style}]❌ {entry.component}: {entry.message}[/{style}]")
    
    def get_component_logger(self, component: str, task_id: Optional[str] = None,
                           session_id: Optional[str] = None, workspace_path: Optional[str] = None) -> ContextualLogger:
        """Get or create a contextual logger for a component."""
        
        logger_key = f"{component}_{task_id}_{session_id}"
        
        if logger_key not in self.component_loggers:
            self.component_loggers[logger_key] = ContextualLogger(
                component=component,
                task_id=task_id,
                session_id=session_id,
                workspace_path=workspace_path
            )
        
        return self.component_loggers[logger_key]
    
    def get_logs_by_category(self, category: LogCategory, limit: int = 100) -> List[StructuredLogEntry]:
        """Get logs filtered by category."""
        with self._lock:
            filtered_logs = [
                entry for entry in self.log_buffer 
                if entry.category == category.value
            ]
            return filtered_logs[-limit:]
    
    def get_logs_by_task(self, task_id: str, limit: int = 100) -> List[StructuredLogEntry]:
        """Get logs for a specific task."""
        with self._lock:
            filtered_logs = [
                entry for entry in self.log_buffer 
                if entry.task_id == task_id
            ]
            return filtered_logs[-limit:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors in current session."""
        with self._lock:
            errors = [
                entry for entry in self.log_buffer 
                if entry.level in [LogLevel.ERROR.value, LogLevel.CRITICAL.value]
            ]
            
            error_by_component = {}
            for error in errors:
                if error.component not in error_by_component:
                    error_by_component[error.component] = []
                error_by_component[error.component].append({
                    'message': error.message,
                    'timestamp': error.timestamp,
                    'category': error.category,
                    'metadata': error.metadata
                })
            
            return {
                'total_errors': len(errors),
                'errors_by_component': error_by_component,
                'recent_errors': [entry.to_dict() for entry in errors[-5:]]
            }
    
    def display_session_summary(self) -> None:
        """Display session summary in console."""
        
        summary_table = Table(title=f"Session Summary - {self.session_timestamp}")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Count", style="white")
        summary_table.add_column("Last Entry", style="dim")
        
        categories = {}
        for entry in self.log_buffer:
            if entry.category not in categories:
                categories[entry.category] = {'count': 0, 'last_entry': None}
            categories[entry.category]['count'] += 1
            categories[entry.category]['last_entry'] = entry.timestamp
        
        for category, data in categories.items():
            summary_table.add_row(
                category.replace('_', ' ').title(),
                str(data['count']),
                data['last_entry'][:19] if data['last_entry'] else "-"
            )
        
        self.console.print(summary_table)
        
        # Display error summary if any
        error_summary = self.get_error_summary()
        if error_summary['total_errors'] > 0:
            error_panel = Panel(
                f"Total Errors: {error_summary['total_errors']}\n" +
                f"Components with errors: {', '.join(error_summary['errors_by_component'].keys())}",
                title="Error Summary",
                border_style="red"
            )
            self.console.print(error_panel)
        
        # Display log file locations
        info_panel = Panel(
            f"Structured logs: {self.session_log_file}\n" +
            f"Debug logs: {self.debug_log_file}",
            title="Log Files",
            border_style="blue"
        )
        self.console.print(info_panel)
    
    def export_logs(self, output_file: str, category: Optional[LogCategory] = None,
                   task_id: Optional[str] = None) -> None:
        """Export logs to file with optional filtering."""
        
        logs_to_export = self.log_buffer
        
        if category:
            logs_to_export = [log for log in logs_to_export if log.category == category.value]
        
        if task_id:
            logs_to_export = [log for log in logs_to_export if log.task_id == task_id]
        
        export_data = {
            'session_timestamp': self.session_timestamp,
            'export_timestamp': datetime.now().isoformat(),
            'total_logs': len(logs_to_export),
            'filters': {
                'category': category.value if category else None,
                'task_id': task_id
            },
            'logs': [log.to_dict() for log in logs_to_export]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        self.console.print(f"[green]Exported {len(logs_to_export)} logs to {output_file}[/green]")


# Global enhanced logger instance
enhanced_logger = EnhancedLogger()


def get_logger(component: str, task_id: Optional[str] = None, 
               session_id: Optional[str] = None, workspace_path: Optional[str] = None) -> ContextualLogger:
    """Get a contextual logger for a component."""
    return enhanced_logger.get_component_logger(component, task_id, session_id, workspace_path)