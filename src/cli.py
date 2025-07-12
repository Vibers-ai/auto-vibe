"""Command Line Interface for VIBE."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

import sys
from pathlib import Path

# src 디렉토리를 Python path에 추가
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from shared.utils.config import Config
from shared.agents.document_ingestion import DocumentIngestionAgent
from shared.agents.master_planner import MasterPlannerAgent

app = typer.Typer(name="vibe", help="VIBE - Autonomous Coding Agent")
console = Console()


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,  # Force DEBUG level to see all logs
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_level=True, show_time=True)]
    )


@app.command()
def generate(
    docs_path: str = typer.Option("docs", "--docs", "-d", help="Path to documentation folder"),
    output_path: str = typer.Option("output", "--output", "-o", help="Path to output folder"),
    env_file: Optional[str] = typer.Option(None, "--env", "-e", help="Path to environment file"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    skip_planning: bool = typer.Option(False, "--skip-planning", help="Skip planning phase and use existing tasks.json"),
):
    """Generate a complete project from documentation."""
    setup_logging(log_level)
    
    console.print(Panel.fit(
        Text("VIBE - Autonomous Coding Agent", style="bold blue"),
        subtitle="Transforming docs into code",
        border_style="blue"
    ))
    
    try:
        # Load configuration
        config = Config.from_env(env_file)
        config.validate()
        
        # Run the main generation pipeline
        asyncio.run(run_generation_pipeline(
            config, docs_path, output_path, skip_planning
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


async def run_generation_pipeline(
    config: Config, 
    docs_path: str, 
    output_path: str, 
    skip_planning: bool
):
    """Main generation pipeline."""
    
    # Phase 1: Document Ingestion (unless skipping planning)
    if not skip_planning:
        console.print("\n[blue]Phase 1: Document Ingestion & Synthesis[/blue]")
        doc_agent = DocumentIngestionAgent(config)
        
        try:
            project_brief_path = doc_agent.process_documents(docs_path)
            console.print(f"[green]✓ Project brief created: {project_brief_path}[/green]")
        except FileNotFoundError:
            console.print(f"[red]Error: Documentation folder not found: {docs_path}[/red]")
            console.print("[yellow]Tip: Create a 'docs' folder and add your project documentation (PDFs, Word docs, images, etc.)[/yellow]")
            return
        except Exception as e:
            console.print(f"[red]Error during document processing: {e}[/red]")
            return
        
        # Phase 2: Master Planning
        console.print("\n[blue]Phase 2: Master Planning & Architecture Design[/blue]")
        planner = MasterPlannerAgent(config)
        
        try:
            tasks_path = planner.create_master_plan(project_brief_path)
            console.print(f"[green]✓ Master plan created: {tasks_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error during planning: {e}[/red]")
            return
    else:
        console.print("[yellow]Skipping planning phase, using existing tasks.json[/yellow]")
        tasks_path = "tasks.json"
        
        # Validate existing tasks.json
        if not Path(tasks_path).exists():
            console.print(f"[red]Error: {tasks_path} not found[/red]")
            return
        
        from shared.core.schema import TaskSchemaValidator
        validated_plan = TaskSchemaValidator.validate_file(tasks_path)
        if validated_plan is None:
            console.print(f"[red]Error: Invalid tasks.json file[/red]")
            return
    
    # Phase 3: Task Execution
    console.print("\n[blue]Phase 3: Task Execution & Code Generation[/blue]")
    from cli.core.executor_cli import TaskExecutorCli
    executor = TaskExecutorCli(config)
    
    try:
        # Pass the correct output_path to executor
        success = await executor.execute_tasks(tasks_path, output_path)
        
        if success:
            console.print(f"\n[green]🎉 Project generation completed successfully![/green]")
            console.print(f"[green]Check the '{output_path}' folder for your generated project.[/green]")
        else:
            console.print(f"\n[red]❌ Project generation failed.[/red]")
            console.print("[yellow]Check the logs above for error details.[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error during execution: {e}[/red]")
        return


@app.command()
def validate(
    tasks_file: str = typer.Argument("tasks.json", help="Path to tasks.json file to validate"),
):
    """Validate a tasks.json file."""
    console.print(f"[blue]Validating {tasks_file}...[/blue]")
    
    from shared.core.schema import TaskSchemaValidator
    validated_plan = TaskSchemaValidator.validate_file(tasks_file)
    
    if validated_plan:
        console.print(f"[green]✓ {tasks_file} is valid![/green]")
        console.print(f"[green]  Project ID: {validated_plan.project_id}[/green]")
        console.print(f"[green]  Total tasks: {len(validated_plan.tasks)}[/green]")
        
        # Show task breakdown
        task_types = {}
        for task in validated_plan.tasks:
            task_types[task.type] = task_types.get(task.type, 0) + 1
        
        console.print("[green]  Task breakdown:[/green]")
        for task_type, count in task_types.items():
            console.print(f"[green]    {task_type}: {count}[/green]")
    else:
        console.print(f"[red]❌ {tasks_file} is invalid![/red]")
        raise typer.Exit(1)


@app.command()
def sample(
    output_file: str = typer.Option("sample-tasks.json", "--output", "-o", help="Output file for sample tasks"),
):
    """Generate a sample tasks.json file for reference."""
    console.print("[blue]Generating sample tasks.json...[/blue]")
    
    from shared.core.schema import create_sample_tasks
    sample_plan = create_sample_tasks()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(sample_plan.json(indent=2))
    
    console.print(f"[green]✓ Sample tasks.json created: {output_file}[/green]")
    console.print("[yellow]You can use this as a reference for creating your own tasks.json file.[/yellow]")


@app.command()
def status(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID to check"),
):
    """Check the status of a task execution session."""
    console.print("[blue]Checking execution status...[/blue]")
    
    from shared.core.state_manager import ExecutorState
    
    state = ExecutorState()
    
    if session_id:
        progress = state.get_session_progress(session_id)
        if not progress:
            console.print(f"[red]Session {session_id} not found[/red]")
            return
        
        _display_session_status(progress)
    else:
        console.print("[yellow]Session ID not provided. Use --session to specify a session.[/yellow]")
        console.print("[yellow]Check executor_state.db for available sessions.[/yellow]")


@app.command()
def context(
    action: str = typer.Argument(help="Action: stats, compress, export, import, preview"),
    project_id: Optional[str] = typer.Option(None, "--project", "-p", help="Project ID"),
    file_path: Optional[str] = typer.Option(None, "--file", "-f", help="File path for export/import"),
    task_id: Optional[str] = typer.Option(None, "--task", "-t", help="Task ID for preview"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="Compression strategy"),
):
    """Manage Master Claude's context and memory."""
    
    try:
        config = Config.from_env()
        
        if action == "stats":
            _show_context_stats(config, project_id)
        elif action == "compress":
            _force_context_compression(config, project_id, strategy)
        elif action == "export":
            if not file_path:
                console.print("[red]Error: --file required for export[/red]")
                return
            _export_context(config, project_id, file_path)
        elif action == "import":
            if not file_path:
                console.print("[red]Error: --file required for import[/red]")
                return
            _import_context(config, project_id, file_path)
        elif action == "preview":
            if not task_id:
                console.print("[red]Error: --task required for preview[/red]")
                return
            _preview_task_context(config, project_id, task_id)
        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("[yellow]Available actions: stats, compress, export, import, preview[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def monitor(
    mode: str = typer.Option("terminal", "--mode", "-m", help="Monitoring mode: terminal, web, both"),
    port: int = typer.Option(8000, "--port", "-p", help="Web dashboard port"),
    host: str = typer.Option("localhost", "--host", help="Web dashboard host"),
):
    """Start Master Claude monitoring dashboard."""
    
    try:
        config = Config.from_env()
        
        if mode == "terminal":
            _start_terminal_monitoring(config)
        elif mode == "web":
            _start_web_monitoring(config, host, port)
        elif mode == "both":
            _start_both_monitoring(config, host, port)
        else:
            console.print(f"[red]Unknown mode: {mode}[/red]")
            console.print("[yellow]Available modes: terminal, web, both[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def demo_monitoring():
    """Run the Master Claude monitoring demonstration."""
    console.print("[blue]🔍 Starting Master Claude monitoring demo...[/blue]")
    
    try:
        import asyncio
        from demo_monitoring import main
        asyncio.run(main())
    except ImportError:
        console.print("[red]Demo module not found. Please ensure demo_monitoring.py is available.[/red]")
    except Exception as e:
        console.print(f"[red]Demo error: {e}[/red]")


def _display_session_status(progress):
    """Display session status information."""
    from rich.table import Table
    
    # Session overview
    console.print(f"[blue]Session: {progress['session_id']}[/blue]")
    console.print(f"[blue]Project: {progress['project_id']}[/blue]")
    console.print(f"[blue]Status: {progress['status']}[/blue]")
    console.print(f"[blue]Started: {progress['started_at']}[/blue]")
    
    # Progress summary
    total = progress['total_tasks']
    completed = progress['completed_tasks']
    failed = progress['failed_tasks']
    pending = total - completed - failed
    
    console.print(f"[green]Progress: {completed}/{total} completed, {failed} failed, {pending} pending[/green]")
    
    # Task details
    if progress['tasks']:
        table = Table(title="Task Status")
        table.add_column("Task ID", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Attempts", style="yellow")
        table.add_column("Last Error", style="red")
        
        for task_id, task_info in progress['tasks'].items():
            error = task_info['last_error'] or ""
            if len(error) > 50:
                error = error[:47] + "..."
            
            table.add_row(
                task_id,
                task_info['status'],
                str(task_info['attempts']),
                error
            )
        
        console.print(table)


@app.command()
def init(
    project_name: str = typer.Argument(help="Name of the project to initialize"),
    include_sample_docs: bool = typer.Option(True, "--sample-docs/--no-sample-docs", help="Include sample documentation"),
):
    """Initialize a new VIBE project structure."""
    console.print(f"[blue]Initializing VIBE project: {project_name}[/blue]")
    
    project_path = Path(project_name)
    
    if project_path.exists():
        console.print(f"[red]Error: Directory {project_name} already exists![/red]")
        raise typer.Exit(1)
    
    # Create project structure
    project_path.mkdir()
    (project_path / "docs").mkdir()
    (project_path / "output").mkdir()
    (project_path / "output" / "logs").mkdir()
    
    # Create .env file
    env_content = """# VIBE Configuration
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key_here

# Optional: Model Configuration
GEMINI_MODEL=gemini-2.0-flash-exp
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Optional: Execution Configuration
MAX_RETRIES=3
TASK_TIMEOUT=600
PARALLEL_TASKS=4
LOG_LEVEL=INFO
"""
    
    (project_path / ".env").write_text(env_content)
    
    # Create sample documentation if requested
    if include_sample_docs:
        sample_prd = """# Sample Project: Task Management App

## Overview
Create a simple task management web application that allows users to create, update, and delete tasks.

## Features
- User authentication (register/login)
- Create, read, update, delete tasks
- Mark tasks as complete/incomplete
- Filter tasks by status
- Simple web interface

## Technical Requirements
- Backend: Python FastAPI
- Frontend: React with TypeScript
- Database: SQLite for development
- Authentication: JWT tokens

## API Endpoints
- POST /auth/register - User registration
- POST /auth/login - User login
- GET /tasks - List all tasks for user
- POST /tasks - Create new task
- PUT /tasks/{id} - Update task
- DELETE /tasks/{id} - Delete task

## Database Schema
Users table:
- id (primary key)
- username (unique)
- email (unique)
- password_hash
- created_at

Tasks table:
- id (primary key)
- user_id (foreign key)
- title
- description
- completed (boolean)
- created_at
- updated_at
"""
        
        (project_path / "docs" / "project_requirements.md").write_text(sample_prd)
        
        console.print(f"[green]✓ Sample documentation created in {project_name}/docs/[/green]")
    
    console.print(f"[green]✓ VIBE project '{project_name}' initialized successfully![/green]")
    console.print(f"[yellow]Next steps:[/yellow]")
    console.print(f"[yellow]1. cd {project_name}[/yellow]")
    console.print(f"[yellow]2. Edit .env file with your API keys[/yellow]")
    console.print(f"[yellow]3. Add your project documentation to the docs/ folder[/yellow]")
    console.print(f"[yellow]4. Run: vibe generate[/yellow]")


def _show_context_stats(config: Config, project_id: Optional[str]):
    """Show context statistics."""
    from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
    
    supervisor = MasterClaudeCliSupervisor(config)
    
    # Load context state if available
    if project_id:
        context_file = f"context_state_{project_id}.json"
        import asyncio
        asyncio.run(supervisor.context_manager.load_context_state(context_file))
    
    stats = supervisor.context_manager.get_context_stats()
    
    console.print("[blue]📊 Context Statistics[/blue]")
    console.print(f"Total tokens: {stats['total_tokens']:,}")
    console.print(f"Available tokens: {stats['available_tokens']:,}")
    console.print(f"Utilization: {stats['utilization']:.1%}")
    console.print(f"Active contexts: {stats['context_windows']}")
    console.print(f"Summaries: {stats['summaries']}")
    console.print(f"Compression ratio: {stats['compression_ratio']:.2f}x")


def _force_context_compression(config: Config, project_id: Optional[str], strategy: Optional[str]):
    """Force context compression."""
    from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
    from shared.agents.context_manager import ContextCompressionStrategy
    
    import asyncio
    
    async def compress():
        supervisor = MasterClaudeCliSupervisor(config)
        
        # Set strategy if provided
        if strategy:
            try:
                strategy_enum = ContextCompressionStrategy(strategy.lower())
                await supervisor.set_context_strategy(strategy_enum)
            except ValueError:
                console.print(f"[red]Invalid strategy: {strategy}[/red]")
                console.print("[yellow]Valid strategies: summarize, hierarchical, sliding_window, semantic_filtering, hybrid[/yellow]")
                return
        
        # Load context state if available
        if project_id:
            context_file = f"context_state_{project_id}.json"
            await supervisor.context_manager.load_context_state(context_file)
        
        # Force compression
        result = await supervisor.force_context_compression()
        
        console.print(f"[green]Compression completed![/green]")
        console.print(f"Tokens saved: {result['tokens_saved']}")
        console.print(f"Improvement: {result['compression_improvement']:.1%}")
    
    asyncio.run(compress())


def _export_context(config: Config, project_id: Optional[str], file_path: str):
    """Export project knowledge."""
    from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
    
    import asyncio
    
    async def export():
        supervisor = MasterClaudeCliSupervisor(config)
        
        # Load context state if available
        if project_id:
            context_file = f"context_state_{project_id}.json"
            await supervisor.context_manager.load_context_state(context_file)
        
        await supervisor.export_project_knowledge(file_path)
    
    asyncio.run(export())


def _import_context(config: Config, project_id: Optional[str], file_path: str):
    """Import project knowledge."""
    from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
    
    import asyncio
    
    async def import_knowledge():
        supervisor = MasterClaudeCliSupervisor(config)
        success = await supervisor.load_project_knowledge(file_path)
        
        if success and project_id:
            # Save updated context state
            context_file = f"context_state_{project_id}.json"
            await supervisor.context_manager.save_context_state(context_file)
    
    asyncio.run(import_knowledge())


def _preview_task_context(config: Config, project_id: Optional[str], task_id: str):
    """Preview context for a specific task."""
    from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
    from shared.core.schema import TaskSchemaValidator
    
    import asyncio
    
    async def preview():
        supervisor = MasterClaudeCliSupervisor(config)
        
        # Load context state if available
        if project_id:
            context_file = f"context_state_{project_id}.json"
            await supervisor.context_manager.load_context_state(context_file)
        
        # Load tasks.json to find the task
        try:
            tasks_plan = TaskSchemaValidator.validate_file("tasks.json")
            if not tasks_plan:
                console.print("[red]No valid tasks.json found[/red]")
                return
            
            # Find the task
            task = None
            for t in tasks_plan.tasks:
                if t.id == task_id:
                    task = t
                    break
            
            if not task:
                console.print(f"[red]Task {task_id} not found[/red]")
                return
            
            # Get context preview
            context_preview = await supervisor.get_task_context_preview(task, tasks_plan)
            
            console.print(f"[blue]📋 Context Preview for Task: {task_id}[/blue]")
            console.print(f"[dim]Length: {len(context_preview)} characters[/dim]")
            console.print("\n" + "="*60)
            console.print(context_preview)
            console.print("="*60)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(preview())


def _start_terminal_monitoring(config: Config):
    """Start terminal-based monitoring."""
    from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
    
    console.print("[blue]🖥️ Starting terminal monitoring...[/blue]")
    
    # 데모용 Master Claude 생성
    supervisor = MasterClaudeCliSupervisor(config, enable_monitoring=True)
    
    if supervisor.monitor:
        supervisor.monitor.start_live_dashboard()
        console.print("[green]✓ Terminal dashboard started[/green]")
        console.print("[yellow]Press Ctrl+C to stop monitoring[/yellow]")
        
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            supervisor.monitor.stop_monitoring()
            console.print("\n[blue]Terminal monitoring stopped[/blue]")
    else:
        console.print("[red]Monitoring not available[/red]")


def _start_web_monitoring(config: Config, host: str, port: int):
    """Start web-based monitoring."""
    from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
    from shared.monitoring import WebDashboard
    import asyncio
    
    console.print(f"[blue]🌐 Starting web monitoring at http://{host}:{port}...[/blue]")
    
    async def run_web_dashboard():
        # 데모용 Master Claude 생성
        supervisor = MasterClaudeCliSupervisor(config, enable_monitoring=True)
        
        if supervisor.monitor:
            # 웹 대시보드 생성 및 시작
            dashboard = WebDashboard(supervisor.monitor, host, port)
            await dashboard.start_server()
        else:
            console.print("[red]Monitoring not available[/red]")
    
    try:
        asyncio.run(run_web_dashboard())
    except KeyboardInterrupt:
        console.print("\n[blue]Web monitoring stopped[/blue]")


def _start_both_monitoring(config: Config, host: str, port: int):
    """Start both terminal and web monitoring."""
    from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
    from shared.monitoring import WebDashboard
    import asyncio
    import threading
    import time
    
    console.print(f"[blue]🔄 Starting both terminal and web monitoring...[/blue]")
    
    # 데모용 Master Claude 생성
    supervisor = MasterClaudeCliSupervisor(config, enable_monitoring=True)
    
    if not supervisor.monitor:
        console.print("[red]Monitoring not available[/red]")
        return
    
    # 웹 대시보드 백그라운드 시작
    dashboard = WebDashboard(supervisor.monitor, host, port)
    dashboard.start_server_background()
    
    console.print(f"[green]✓ Web dashboard: http://{host}:{port}[/green]")
    
    # 터미널 대시보드 시작
    supervisor.monitor.start_live_dashboard()
    console.print("[green]✓ Terminal dashboard started[/green]")
    console.print("[yellow]Press Ctrl+C to stop monitoring[/yellow]")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        supervisor.monitor.stop_monitoring()
        console.print("\n[blue]Monitoring stopped[/blue]")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()