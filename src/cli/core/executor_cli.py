"""Task Executor for CLI-based Claude execution."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TaskID

from shared.utils.config import Config
from shared.core.schema import TasksPlan, TaskSchemaValidator
from cli.agents.master_claude_cli_supervisor import MasterClaudeCliSupervisor
from shared.core.state_manager import ExecutorState
from shared.core.parallel_executor import ParallelTaskExecutor
from shared.core.consistency_manager import ConsistencyManager

logger = logging.getLogger(__name__)
console = Console()


class TaskExecutorCli:
    """Execute tasks using Claude CLI with parallel processing and consistency management."""
    
    def __init__(self, config: Config, use_master_supervision: bool = True, max_parallel_tasks: int = 4):
        self.config = config
        self.use_master_supervision = use_master_supervision
        self.max_parallel_tasks = max_parallel_tasks
        
        # Initialize Master Claude CLI Supervisor
        if use_master_supervision:
            self.master_supervisor = MasterClaudeCliSupervisor(config)
        else:
            self.master_supervisor = None
        
        # Initialize parallel executor and consistency manager
        self.parallel_executor = ParallelTaskExecutor(config, max_parallel_tasks)
        self.consistency_manager = ConsistencyManager(config)
        
        # State management
        self.state = ExecutorState()
        self.session_id = None
        
        # Execution tracking
        self.start_time = None
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.use_parallel_execution = True
    
    async def execute_tasks(self, tasks_file: str, workspace_path: str = "output") -> bool:
        """Execute all tasks from the tasks.json file using CLI.
        
        Args:
            tasks_file: Path to tasks.json file
            workspace_path: Path to workspace directory
            
        Returns:
            True if all tasks completed successfully, False otherwise
        """
        console.print("[bold blue]🚀 Starting CLI-based task execution[/bold blue]")
        
        # Convert workspace_path to absolute path
        if not os.path.isabs(workspace_path):
            workspace_path = os.path.abspath(workspace_path)
        
        console.print(f"[green]📁 Workspace: {workspace_path}[/green]")
        
        # Load and validate tasks
        tasks_plan = TaskSchemaValidator.validate_file(tasks_file)
        if not tasks_plan:
            console.print("[red]❌ Invalid tasks.json file[/red]")
            return False
        
        console.print(f"[green]📋 Loaded {len(tasks_plan.tasks)} tasks for CLI execution[/green]")
        
        # Initialize session
        self.start_time = datetime.now()
        
        # Create logs directory (safe with exist_ok=True)
        self.logs_dir = Path(workspace_path) / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create session log file
        session_timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.logs_dir / f"session_{session_timestamp}.md"
        self._init_session_log(tasks_plan, session_timestamp)
        
        # Save initial state and get session ID
        self.session_id = self.state.create_session(
            project_id=tasks_plan.project_id,
            tasks_file_path=tasks_file,
            total_tasks=len(tasks_plan.tasks)
        )
        
        try:
            # Initialize consistency patterns from workspace
            console.print("[blue]🔍 Analyzing workspace for consistency patterns[/blue]")
            self.consistency_manager.initialize_from_workspace(workspace_path)
            
            # Choose execution method based on configuration
            if self.use_parallel_execution and len(tasks_plan.tasks) > 1:
                console.print(f"[green]⚡ Using parallel execution with {self.max_parallel_tasks} workers[/green]")
                success, execution_stats = await self.parallel_executor.execute_tasks_parallel(
                    tasks_plan, workspace_path
                )
                
                # Update tracking from parallel execution stats
                self.completed_tasks = execution_stats.completed_tasks
                self.failed_tasks = execution_stats.failed_tasks
                
                # Display parallel execution report
                self._display_parallel_execution_report(execution_stats)
                
            else:
                console.print("[yellow]📋 Using sequential execution[/yellow]")
                success = await self._execute_tasks_with_cli_supervision(tasks_plan, workspace_path)
            
            # Perform final consistency check
            console.print("[blue]🔧 Performing final consistency validation[/blue]")
            consistency_report = await self._perform_final_consistency_check(tasks_plan, workspace_path)
            self._display_consistency_report(consistency_report)
            
            # Update final session state
            self.state.update_session_status(
                session_id=self.session_id,
                status="completed" if success else "failed",
                completed_tasks=self.completed_tasks,
                failed_tasks=self.failed_tasks
            )
            
            # Display final results
            self._display_final_results(success, tasks_plan, consistency_report)
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Fatal error during CLI execution: {e}[/red]")
            logger.error(f"Fatal execution error: {e}")
            
            self.state.update_session_status(
                session_id=self.session_id,
                status="failed",
                completed_tasks=self.completed_tasks,
                failed_tasks=self.failed_tasks
            )
            
            return False
        
        finally:
            # Log session summary
            self._log_session_summary(success, tasks_plan)
            
            # Cleanup
            if self.master_supervisor:
                self.master_supervisor.cleanup()
    
    async def _perform_final_consistency_check(self, tasks_plan: TasksPlan, workspace_path: str) -> Dict[str, Any]:
        """Perform final consistency check on all generated code."""
        
        all_reports = []
        overall_score = 0
        total_violations = 0
        
        for task in tasks_plan.tasks:
            try:
                report = self.consistency_manager.check_task_consistency(task, workspace_path)
                all_reports.append(report)
                overall_score += report['consistency_score']
                total_violations += report['total_violations']
            except Exception as e:
                logger.warning(f"Could not check consistency for task {task.id}: {e}")
        
        avg_score = overall_score / len(all_reports) if all_reports else 0
        
        return {
            'overall_consistency_score': avg_score,
            'total_violations': total_violations,
            'task_reports': all_reports,
            'recommendations': self._generate_overall_recommendations(all_reports)
        }
    
    def _generate_overall_recommendations(self, reports: List[Dict[str, Any]]) -> List[str]:
        """Generate overall recommendations from all consistency reports."""
        
        all_recommendations = []
        violation_types = {}
        
        for report in reports:
            all_recommendations.extend(report.get('recommendations', []))
            
            for violation_type, count in report.get('violations_by_type', {}).items():
                violation_types[violation_type] = violation_types.get(violation_type, 0) + count
        
        # Generate summary recommendations
        recommendations = []
        
        if violation_types.get('naming_convention', 0) > 5:
            recommendations.append("Establish and enforce consistent naming conventions across the project")
        
        if violation_types.get('coding_style', 0) > 3:
            recommendations.append("Set up automated code formatting (e.g., Black for Python, Prettier for JS)")
        
        if violation_types.get('import_structure', 0) > 2:
            recommendations.append("Configure import sorting tools (e.g., isort)")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations + recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:5]  # Top 5 recommendations
    
    def _display_parallel_execution_report(self, stats) -> None:
        """Display parallel execution statistics."""
        
        from rich.table import Table
        from rich.panel import Panel
        
        # Create execution summary table
        table = Table(title="Parallel Execution Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Tasks", str(stats.total_tasks))
        table.add_row("Completed", f"[green]{stats.completed_tasks}[/green]")
        table.add_row("Failed", f"[red]{stats.failed_tasks}[/red]")
        table.add_row("Parallel Tasks", str(stats.parallel_tasks_executed))
        table.add_row("Max Parallel Degree", str(stats.max_parallel_degree))
        table.add_row("Total Time", f"{stats.total_execution_time:.2f}s")
        
        console.print(table)
        
        # Display performance insights
        if hasattr(self.parallel_executor, 'get_execution_report'):
            report = self.parallel_executor.get_execution_report()
            insights = report.get('performance_insights', [])
            
            if insights:
                insights_text = "\n".join(f"• {insight}" for insight in insights)
                console.print(Panel(
                    insights_text,
                    title="Performance Insights",
                    border_style="green"
                ))
    
    def _display_consistency_report(self, consistency_report: Dict[str, Any]) -> None:
        """Display consistency validation report."""
        
        from rich.table import Table
        from rich.panel import Panel
        
        score = consistency_report['overall_consistency_score']
        violations = consistency_report['total_violations']
        
        # Color code the score
        if score >= 90:
            score_color = "green"
            score_icon = "✅"
        elif score >= 70:
            score_color = "yellow"
            score_icon = "⚠️"
        else:
            score_color = "red"
            score_icon = "❌"
        
        # Create consistency summary
        summary_text = f"{score_icon} Consistency Score: [{score_color}]{score:.1f}/100[/{score_color}]\n"
        summary_text += f"Total Violations: {violations}"
        
        console.print(Panel(
            summary_text,
            title="Code Consistency Report",
            border_style=score_color
        ))
        
        # Display recommendations if any
        recommendations = consistency_report.get('recommendations', [])
        if recommendations:
            rec_text = "\n".join(f"• {rec}" for rec in recommendations)
            console.print(Panel(
                rec_text,
                title="Recommendations",
                border_style="blue"
            ))
    
    def _display_final_results(self, success: bool, tasks_plan: TasksPlan, consistency_report: Dict[str, Any] = None) -> None:
        """Display enhanced final results with consistency information."""
        
        from rich.panel import Panel
        from rich.table import Table
        
        # Execution summary
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        success_rate = (self.completed_tasks / len(tasks_plan.tasks) * 100) if tasks_plan.tasks else 0
        
        # Create results table
        results_table = Table(title="Execution Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="white")
        
        results_table.add_row("Total Tasks", str(len(tasks_plan.tasks)))
        results_table.add_row("Completed", f"[green]{self.completed_tasks}[/green]")
        results_table.add_row("Failed", f"[red]{self.failed_tasks}[/red]")
        results_table.add_row("Success Rate", f"{success_rate:.1f}%")
        results_table.add_row("Duration", f"{duration:.2f}s")
        
        if consistency_report:
            score = consistency_report['overall_consistency_score']
            score_color = "green" if score >= 90 else "yellow" if score >= 70 else "red"
            results_table.add_row("Consistency Score", f"[{score_color}]{score:.1f}/100[/{score_color}]")
        
        console.print(results_table)
        
        # Overall status
        if success:
            if consistency_report and consistency_report['overall_consistency_score'] >= 90:
                status_text = "🎉 Project generation completed successfully with excellent code consistency!"
                status_color = "green"
            else:
                status_text = "✅ Project generation completed successfully!"
                status_color = "green"
        else:
            status_text = "❌ Project generation failed. Check logs for details."
            status_color = "red"
        
        console.print(Panel(
            status_text,
            border_style=status_color,
            title="Final Status"
        ))
    
    def _create_session_id(self, tasks_plan: TasksPlan) -> str:
        """Create a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cli_{tasks_plan.project_id}_{timestamp}"
    
    async def _execute_tasks_with_cli_supervision(self, tasks_plan: TasksPlan, workspace_path: str) -> bool:
        """Execute tasks with Master Claude CLI supervision."""
        
        console.print("[blue]🧠 Initializing Master Claude CLI Supervisor[/blue]")
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(tasks_plan.tasks)
        execution_order_ids = self._get_execution_order(dependency_graph)
        
        # Convert task IDs to task objects
        task_map = {task.id: task for task in tasks_plan.tasks}
        execution_queue = [task_map[task_id] for task_id in execution_order_ids if task_id in task_map]
        
        console.print(f"[green]📊 Execution order determined: {len(execution_queue)} tasks[/green]")
        
        # Track progress
        with Progress() as progress:
            main_task = progress.add_task(
                "[cyan]Executing CLI tasks...", 
                total=len(execution_queue)
            )
            
            for task in execution_queue:
                progress.update(main_task, description=f"[cyan]Executing CLI task: {task.id}")
                
                # Update task state
                self.state.update_task_status(
                    session_id=self.session_id,
                    task_id=task.id,
                    status="running"
                )
                
                console.print(f"\n[bold yellow]🔧 CLI Task: {task.id}[/bold yellow]")
                console.print(f"[dim]Description: {task.description}[/dim]")
                console.print(f"[dim]Type: {task.type} | Area: {task.project_area}[/dim]")
                
                try:
                    # Execute task with Master Claude CLI supervision
                    if self.master_supervisor:
                        result = await self.master_supervisor.execute_task_with_cli_supervision(
                            task, workspace_path, tasks_plan
                        )
                    else:
                        # Fallback: direct CLI execution without supervision
                        result = await self._execute_task_directly_cli(task, workspace_path)
                    
                    # Log task execution details
                    self._log_task_execution(task, result)
                    
                    # Process result
                    if result.get('overall_success', result.get('success', False)):
                        console.print(f"[green]✅ CLI Task {task.id} completed successfully[/green]")
                        self.completed_tasks += 1
                        
                        self.state.update_task_status(
                            session_id=self.session_id,
                            task_id=task.id,
                            status="completed",
                            output=json.dumps(result)
                        )
                    else:
                        console.print(f"[red]❌ CLI Task {task.id} failed[/red]")
                        self.failed_tasks += 1
                        
                        error_msg = result.get('error', 'Unknown error')
                        console.print(f"[red]Error: {error_msg}[/red]")
                        
                        self.state.update_task_status(
                            session_id=self.session_id,
                            task_id=task.id,
                            status="failed",
                            error=error_msg,
                            output=json.dumps(result)
                        )
                        
                        # Decide whether to continue or abort
                        if not self._should_continue_on_failure(task, result):
                            console.print("[red]🛑 Aborting execution due to critical failure[/red]")
                            return False
                
                except Exception as e:
                    console.print(f"[red]❌ CLI Task {task.id} crashed: {e}[/red]")
                    logger.error(f"Task {task.id} crashed: {e}")
                    self.failed_tasks += 1
                    
                    self.state.update_task_status(
                        session_id=self.session_id,
                        task_id=task.id,
                        status="failed",
                        error=str(e)
                    )
                    
                    if not self._should_continue_on_failure(task, {'error': str(e)}):
                        return False
                
                finally:
                    progress.advance(main_task)
        
        # Check overall success
        success_rate = self.completed_tasks / len(execution_queue) if execution_queue else 0
        return success_rate >= 0.8  # 80% success rate required
    
    def _init_session_log(self, tasks_plan: TasksPlan, session_timestamp: str):
        """Initialize session log file."""
        log_content = f"""# VIBE CLI Execution Log
**Session:** {session_timestamp}  
**Project:** {tasks_plan.project_id}  
**Started:** {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}  
**Total Tasks:** {len(tasks_plan.tasks)}

---

## Session Overview

### Tasks to Execute:
"""
        
        for task in tasks_plan.tasks:
            log_content += f"- **{task.id}**: {task.description} ({task.type})\n"
        
        log_content += "\n---\n\n"
        
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)
    
    def _log_task_execution(self, task, result, master_prompt=None, claude_response=None):
        """Log detailed task execution information."""
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        task_log = f"""## Task: {task.id}
**Time:** {timestamp}  
**Description:** {task.description}  
**Type:** {task.type}  
**Area:** {task.project_area}  
**Files:** {', '.join(task.files_to_create_or_modify)}  
**Status:** {'✅ SUCCESS' if result.get('overall_success', result.get('success', False)) else '❌ FAILED'}

### Master Prompt Analysis
"""
        
        if master_prompt:
            task_log += f"""```
{master_prompt[:500]}{'...' if len(master_prompt) > 500 else ''}
```

"""
        
        if claude_response and 'execution_log' in result:
            task_log += "### Claude CLI Execution Log\n"
            for log_entry in result['execution_log'][-5:]:  # Last 5 entries
                if 'content' in log_entry:
                    content = log_entry['content']
                    if len(content) > 200:
                        content = content[:200] + "..."
                    task_log += f"- **{log_entry.get('timestamp', 'Unknown')}**: {content}\n"
        
        # Log execution results
        task_log += f"""
### Execution Results
**Success:** {result.get('overall_success', result.get('success', False))}  
**Error:** {result.get('error', 'None')}  
"""
        
        if 'final_state' in result:
            task_log += f"**Final State:** Files created/modified in workspace\n"
        
        if 'verification_results' in result:
            verification = result['verification_results']
            task_log += f"""
### Verification
**Tests Passed:** {len(verification.get('tests_passed', []))}  
**Tests Failed:** {len(verification.get('tests_failed', []))}  
**Files Verified:** {len(verification.get('files_verified', []))}
"""
        
        task_log += "\n---\n\n"
        
        # Append to session log
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(task_log)
    
    def _log_session_summary(self, success: bool, tasks_plan: TasksPlan):
        """Log final session summary."""
        
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None
        
        summary_log = f"""## Session Summary

**Completed:** {end_time.strftime("%Y-%m-%d %H:%M:%S")}  
**Duration:** {duration}  
**Overall Success:** {'✅ YES' if success else '❌ NO'}  
**Tasks Completed:** {self.completed_tasks}/{len(tasks_plan.tasks)}  
**Tasks Failed:** {self.failed_tasks}/{len(tasks_plan.tasks)}  
**Success Rate:** {(self.completed_tasks / len(tasks_plan.tasks) * 100):.1f}%

### Files Generated
Check the workspace directory for generated files and project structure.

### Logs Location
- **Session Log:** `{self.session_log_file.name}`
- **System Logs:** Console output above

---

*Generated by VIBE CLI Execution System*
"""
        
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(summary_log)
        
        console.print(f"[blue]📝 Detailed logs saved to: {self.session_log_file}[/blue]")
    
    async def _execute_task_directly_cli(self, task, workspace_path: str) -> Dict[str, Any]:
        """Fallback: Execute task directly with CLI without Master supervision."""
        from ..agents.claude_cli_executor import ClaudeCliExecutor
        
        console.print(f"[yellow]⚠️ Executing task {task.id} directly with CLI (no Master supervision)[/yellow]")
        
        try:
            executor = ClaudeCliExecutor(self.config)
            
            # Simple context for direct execution
            master_context = f"Project task execution for {task.project_area} area"
            task_context = f"Implement: {task.description}"
            
            result = await executor.execute_task_with_curated_context(
                task, workspace_path, master_context, task_context
            )
            
            executor.cleanup()
            return result
            
        except Exception as e:
            logger.error(f"Direct CLI execution failed for task {task.id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task.id,
                'executor_type': 'cli_direct'
            }
    
    def _build_dependency_graph(self, tasks) -> Dict[str, List[str]]:
        """Build dependency graph from tasks."""
        graph = {}
        
        for task in tasks:
            graph[task.id] = task.dependencies.copy()
        
        return graph
    
    def _get_execution_order(self, dependency_graph: Dict[str, List[str]]) -> List:
        """Get topological order for task execution."""
        # Simple topological sort
        in_degree = {task_id: 0 for task_id in dependency_graph}
        
        # Calculate in-degrees
        for task_id, deps in dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[task_id] += 1
        
        # Find execution order
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Update in-degrees
            for task_id, deps in dependency_graph.items():
                if current in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        # Return the execution order as task IDs
        return execution_order
    
    def _should_continue_on_failure(self, task, result: Dict[str, Any]) -> bool:
        """Determine if execution should continue after a task failure."""
        
        # Critical tasks that should stop execution
        critical_keywords = ['auth', 'database', 'core', 'foundation']
        
        if any(keyword in task.id.lower() or keyword in task.description.lower() 
               for keyword in critical_keywords):
            console.print(f"[red]🚨 Critical task {task.id} failed - stopping execution[/red]")
            return False
        
        # Check failure rate
        total_attempted = self.completed_tasks + self.failed_tasks
        if total_attempted > 5 and (self.failed_tasks / total_attempted) > 0.3:
            console.print("[red]🚨 Too many failures (>30%) - stopping execution[/red]")
            return False
        
        console.print(f"[yellow]⚠️ Non-critical task {task.id} failed - continuing execution[/yellow]")
        return True
    
    def _display_final_results(self, success: bool, tasks_plan: TasksPlan):
        """Display final execution results."""
        
        elapsed_time = datetime.now() - self.start_time if self.start_time else None
        total_tasks = len(tasks_plan.tasks)
        
        console.print("\n" + "="*60)
        console.print("[bold blue]📊 CLI EXECUTION SUMMARY[/bold blue]")
        console.print("="*60)
        
        # Basic stats
        console.print(f"[blue]Project:[/blue] {tasks_plan.project_id}")
        console.print(f"[blue]Session:[/blue] {self.session_id}")
        console.print(f"[blue]Executor:[/blue] Claude CLI")
        
        if elapsed_time:
            console.print(f"[blue]Duration:[/blue] {elapsed_time}")
        
        # Task completion stats
        console.print(f"[green]✅ Completed:[/green] {self.completed_tasks}/{total_tasks}")
        console.print(f"[red]❌ Failed:[/red] {self.failed_tasks}/{total_tasks}")
        
        success_rate = (self.completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        console.print(f"[blue]Success Rate:[/blue] {success_rate:.1f}%")
        
        # Overall result
        if success:
            console.print("\n[bold green]🎉 CLI EXECUTION COMPLETED SUCCESSFULLY! 🎉[/bold green]")
            if self.master_supervisor:
                console.print("[green]✅ Master Claude CLI supervision was effective[/green]")
        else:
            console.print("\n[bold red]❌ CLI EXECUTION FAILED ❌[/bold red]")
            console.print("[yellow]💡 Check the logs above for detailed error information[/yellow]")
        
        # CLI-specific notes
        console.print("\n[dim]📝 CLI Execution Notes:[/dim]")
        console.print("[dim]• Used Claude CLI for direct workspace manipulation[/dim]")
        console.print("[dim]• Master Claude provided intelligent supervision[/dim]")
        console.print("[dim]• Sessions persisted within project areas[/dim]")
        
        console.print("="*60)
    
    def get_session_progress(self) -> Optional[Dict[str, Any]]:
        """Get current session progress."""
        if not self.session_id:
            return None
        
        return self.state.get_session_progress(self.session_id)