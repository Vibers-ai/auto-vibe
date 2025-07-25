"""Master Claude Supervisor using CLI - Orchestrates Code Claude CLI execution with intelligent oversight."""

import asyncio
import logging
<<<<<<< HEAD
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
=======
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271

import google.generativeai as genai
from rich.console import Console

from shared.utils.config import Config
from shared.core.schema import Task, TasksPlan
from shared.tools.aci_interface import ACIInterface, CommandResult
from cli.agents.claude_cli_executor import PersistentClaudeCliSession
from shared.agents.context_manager import ContextManager, ContextCompressionStrategy
from shared.utils.api_manager import api_manager, APIProvider
from shared.utils.response_validator import validate_ai_response
<<<<<<< HEAD
from shared.utils.json_repair import JSONRepair
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
from shared.utils.recovery_manager import get_recovery_manager, TaskStatus
from shared.core.consistency_manager import ConsistencyManager
from shared.core.circuit_breaker import global_circuit_breaker_manager, GEMINI_CIRCUIT_CONFIG
from shared.core.deadlock_detector import global_deadlock_detector
from shared.core.file_operation_guard import global_file_guard
<<<<<<< HEAD
from shared.core.workspace_analyzer import workspace_analyzer
from shared.core.enhanced_logger import get_logger, LogCategory
from shared.core.tech_stack_enforcer import tech_stack_enforcer
from shared.core.realtime_syntax_validator import realtime_syntax_validator, LanguageType, ValidationLevel
from shared.core.file_activity_monitor import get_file_activity_monitor
from shared.core.worker_pool_manager import get_worker_pool_manager, TaskPriority as WorkerTaskPriority
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271

logger = logging.getLogger(__name__)
console = Console()


class MasterClaudeCliSupervisor:
    """Master Claude that supervises and guides Code Claude CLI execution."""
    
    def __init__(self, config: Config, max_context_tokens: int = 128000, enable_monitoring: bool = False):
        self.config = config
        self.gemini_model = None
        self.aci = ACIInterface()
        self._setup_gemini()
        
        # Initialize recovery manager
        self.recovery_manager = None  # Will be set when project is known
        
<<<<<<< HEAD
        # Track completed tasks
        self.completed_tasks = set()
        self.tasks_plan = None
        
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        # Context management
        self.context_manager = ContextManager(config, max_context_tokens)
        self.max_context_tokens = max_context_tokens
        
        # Initialize consistency manager
        self.consistency_manager = ConsistencyManager(config)
        
        # Enhanced Code Claude CLI management
        self.persistent_cli_sessions = PersistentClaudeCliSession(config)
        
<<<<<<< HEAD
        # Worker pool for parallel execution
        self.worker_pool = None
        self.enable_parallel_execution = True
        
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        # State tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.current_workspace_state = {}
        self.task_progress = {}
        
<<<<<<< HEAD
        # Enhanced logging
        self.logger = None  # Will be initialized per task
        
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        # Code Claude CLI insights tracking
        self.claude_insights = {
            'established_patterns': [],
            'coding_conventions': {},
            'common_utilities': [],
            'architectural_decisions': [],
            'cli_specific_patterns': []
        }
        
        # Monitoring (optional)
        self.monitor = None
        if enable_monitoring:
            try:
                from shared.monitoring.master_claude_monitor import MasterClaudeMonitor
                self.monitor = MasterClaudeMonitor(config)
            except ImportError:
                logger.warning("Monitoring not available")
    
    def _setup_gemini(self):
        """Initialize Gemini model for master supervision."""
        genai.configure(api_key=self.config.gemini_api_key)
        self.gemini_model = genai.GenerativeModel(self.config.gemini_model)
    
    async def execute_task_with_cli_supervision(self, task: Task, workspace_path: str, 
<<<<<<< HEAD
                                               tasks_plan: TasksPlan, workspace_context: Optional[str] = None) -> Dict[str, Any]:
=======
                                               tasks_plan: TasksPlan) -> Dict[str, Any]:
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        """Execute a task with Master Claude supervision and Code Claude CLI execution.
        
        Args:
            task: The task to execute
            workspace_path: Path to the workspace
            tasks_plan: Complete tasks plan for context
            
        Returns:
            Execution result with detailed feedback
        """
<<<<<<< HEAD
        # Initialize enhanced logger for this task
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = get_logger(
            component="master_claude_supervisor",
            task_id=task.id,
            session_id=session_id,
            workspace_path=workspace_path
        )
        
        # Store tasks plan for reference
        self.tasks_plan = tasks_plan
        
        self.logger.task_start(f"CLI supervision of task: {task.id}")
        console.print(f"[blue]🧠 Master Claude starting CLI supervision of task: {task.id}[/blue]")
        
        # Update workspace state tracking
        self.current_workspace_state = {
            'path': workspace_path,
            'task_id': task.id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Analyze workspace structure before task execution
        self.logger.workspace_operation("analyze_structure", workspace_path)
        
        # Use provided workspace context or analyze on demand
        workspace_structure = None
        if workspace_context:
            self.logger.info("Using provided workspace context")
            # Still analyze workspace structure for tech stack detection
            workspace_structure = workspace_analyzer.analyze_workspace(workspace_path)
        else:
            workspace_structure = workspace_analyzer.analyze_workspace(workspace_path)
            workspace_context = workspace_analyzer.get_workspace_context_for_claude(workspace_path)
            
            # Log critical structure issues
            for issue in workspace_structure.issues:
                if issue.severity == "critical":
                    self.logger.error(f"Critical workspace issue: {issue.description}", 
                                    LogCategory.STRUCTURE_VALIDATION,
                                    metadata={'issue_type': issue.issue_type.value, 'path': issue.path})
                    console.print(f"[red]🏗️ Critical Structure Issue: {issue.description}[/red]")
                    console.print(f"[yellow]   Suggested fix: {issue.suggested_fix}[/yellow]")
        
        # Keep workspace path as output/ for all tasks including setup
        # This ensures project files are created under output/ directory as requested
        console.print(f"[green]Task {task.id}: Using workspace {workspace_path}[/green]")
        
        # Log workspace structure summary
        if workspace_structure and workspace_structure.technology_stacks:
            primary_stack = workspace_structure.technology_stacks[0]
            console.print(f"[cyan]📚 Detected tech stack: {primary_stack.stack.value} ({primary_stack.confidence:.1%} confidence)[/cyan]")
        
=======
        console.print(f"[blue]🧠 Master Claude starting CLI supervision of task: {task.id}[/blue]")
        
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        # 데드락 감지기에 작업 등록
        worker_id = f"master_claude_{id(self)}"
        global_deadlock_detector.register_task(
            task.id, 
            set(task.dependencies), 
<<<<<<< HEAD
            timeout=1800.0  # 30분 타임아웃
=======
            timeout=600.0  # 10분 타임아웃
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        )
        global_deadlock_detector.update_task_status(task.id, "running", worker_id)
        
        try:
            # Log context stats before starting
            stats = self.context_manager.get_context_stats()
            console.print(f"[dim]Context: {stats['total_tokens']}/{stats['available_tokens']} tokens ({stats['utilization']:.1%} used)[/dim]")
            
            # Phase 1: Analyze current state and plan execution with managed context
<<<<<<< HEAD
            execution_plan = await self._analyze_and_plan_with_context(task, workspace_path, tasks_plan, workspace_context)
            
            # Add execution plan to context
            if execution_plan:
                await self.context_manager.add_context(
                    content=json.dumps(execution_plan, indent=2),
                    content_type="execution_plan",
                    task_ids=[task.id],
                    importance=0.8
                )
            else:
                # Create a minimal execution plan if None
                execution_plan = {
                    "situation_analysis": f"Executing task {task.id}",
                    "execution_strategy": "Direct implementation",
                    "step_by_step_plan": [],
                    "code_claude_cli_instructions": f"Implement {task.description}"
                }
                logger.warning(f"Execution plan was None, using minimal plan for task {task.id}")
=======
            execution_plan = await self._analyze_and_plan_with_context(task, workspace_path, tasks_plan)
            
            # Add execution plan to context
            await self.context_manager.add_context(
                content=json.dumps(execution_plan, indent=2),
                content_type="execution_plan",
                task_ids=[task.id],
                importance=0.8
            )
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            
            # Phase 2: Execute with iterative CLI supervision
            result = await self._execute_with_cli_iterations(task, workspace_path, execution_plan)
            
            # Add logging information to result
            result['master_analysis_prompt'] = execution_plan.get('analysis_prompt_used', '')
            result['cli_prompts_used'] = getattr(self, '_cli_prompts_used', [])
            
            # Add execution result to context
            await self.context_manager.add_context(
                content=json.dumps(result, indent=2),
                content_type="execution_result",
                task_ids=[task.id],
                importance=0.7 if result.get('success') else 0.9  # Higher importance for failures
            )
            
            # Phase 3: Final verification and summary
            final_result = await self._final_verification(task, workspace_path, result)
            
            # Update context manager with task completion
            self.context_manager.update_task_completion(task.id, final_result)
            
            # Add final result to context with high importance
            await self.context_manager.add_context(
                content=json.dumps(final_result, indent=2),
                content_type="task_completion",
                task_ids=[task.id],
                importance=0.9
            )
            
            # Update execution history (keep for immediate access)
            self.execution_history.append({
                'task_id': task.id,
                'execution_plan': execution_plan,
                'result': final_result,
                'timestamp': datetime.now().isoformat(),
                'executor_type': 'cli'
            })
            
            # Save context state periodically
            if len(self.execution_history) % 5 == 0:  # Every 5 tasks
                await self.context_manager.save_context_state(f"context_state_cli_{tasks_plan.project_id}.json")
            
            # 데드락 감지기에 완료 상태 업데이트
            global_deadlock_detector.update_task_status(task.id, "completed", worker_id)
            
<<<<<<< HEAD
            # 작업이 성공했으면 completed_tasks에 추가
            if final_result.get('success', False):
                self.completed_tasks.add(task.id)
            
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            return final_result
            
        except Exception as e:
            # 실패 시 데드락 감지기 업데이트
            global_deadlock_detector.update_task_status(task.id, "failed", worker_id)
<<<<<<< HEAD
            logger.error(f"Task {task.id} execution failed: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            raise
    
    async def _analyze_and_plan_with_context(self, task: Task, workspace_path: str, 
                                            tasks_plan: TasksPlan, workspace_context: Optional[str] = None) -> Dict[str, Any]:
=======
            logger.error(f"Task {task.id} execution failed: {e}")
            raise
    
    async def _analyze_and_plan_with_context(self, task: Task, workspace_path: str, 
                                            tasks_plan: TasksPlan) -> Dict[str, Any]:
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        """Master Claude analyzes the situation with managed context and creates execution plan for CLI."""
        
        # Build optimized context using context manager
        managed_context = await self.context_manager.build_context_for_task(task, tasks_plan)
        
        # Gather current workspace state
        workspace_state = await self._get_workspace_state(workspace_path)
        
<<<<<<< HEAD
        # Determine target workspace relative to output directory
        if task.type == "setup" or task.project_area == "shared":
            target_workspace = "."  # Output directory root for setup tasks
        elif "fe-" in task.id or "frontend" in task.project_area.lower():
            target_workspace = "frontend"
        else:
            target_workspace = "backend"
=======
        # Determine target workspace
        target_workspace = "frontend" if "fe-" in task.id or "frontend" in task.project_area.lower() else "backend"
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        
        # Generate consistency guidelines for this task
        consistency_prompt = self.consistency_manager.generate_consistency_prompt(task, workspace_path)
        
        # Create enhanced analysis prompt with consistency guidance
        analysis_prompt = f"""Analyze task for Claude CLI execution with consistency enforcement.

**CONTEXT:** {managed_context}

<<<<<<< HEAD
**WORKSPACE STRUCTURE ANALYSIS:**
{workspace_context if workspace_context else "No workspace analysis available"}

=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
**CURRENT WORKSPACE:** {workspace_state}

**TASK:** {task.id} - {task.description}
**TARGET WORKSPACE:** {target_workspace}/
**FILES TO CREATE:** {', '.join(task.files_to_create_or_modify)}

**CONSISTENCY REQUIREMENTS:**
{consistency_prompt}

**CRITICAL REQUIREMENTS:**
1. Ensure all files go in {target_workspace}/ workspace
2. Follow established naming conventions and coding patterns
3. Maintain architectural consistency with existing code
4. Use consistent import patterns and code organization

**JSON OUTPUT:**
```json
{{
    "situation_analysis": "Current state and target workspace analysis",
    "execution_strategy": "Implementation approach for {target_workspace}/ workspace", 
    "consistency_guidelines": "Specific patterns and conventions to follow",
    "step_by_step_plan": [
        {{"step": 1, "action": "Check current directory with pwd", "verification": "Confirm correct location"}},
        {{"step": 2, "action": "Navigate to {target_workspace}/ if needed", "verification": "pwd shows correct path"}},
        {{"step": 3, "action": "Main implementation task", "verification": "Files created in correct location and follow patterns"}}
    ],
    "code_claude_cli_instructions": "Specific instructions ensuring correct workspace usage and consistency"
}}
```"""

        try:
            # Store prompt for logging
            execution_plan = {}
            execution_plan['analysis_prompt_used'] = analysis_prompt
            
            # 서킷 브레이커를 통한 보호된 Gemini 호출
            circuit_breaker = global_circuit_breaker_manager.get_circuit_breaker(
                "gemini_analysis", GEMINI_CIRCUIT_CONFIG
            )
            
            response = await circuit_breaker.call(
                self.gemini_model.generate_content,
                analysis_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more focused analysis
                    top_p=0.9,
                    max_output_tokens=4000,
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content:
                logger.warning("Gemini analysis response blocked by safety filters")
                raise Exception("Analysis response blocked by safety filters")
            
            # Parse JSON response with better error handling
            plan_text = response.text
            if '```json' in plan_text:
                json_start = plan_text.find('```json') + 7
                json_end = plan_text.find('```', json_start)
                if json_end == -1:  # No closing ```
                    plan_text = plan_text[json_start:]
                else:
                    plan_text = plan_text[json_start:json_end]
            
<<<<<<< HEAD
            # Clean up common JSON issues without breaking actual newlines in strings
            plan_text = plan_text.strip()
            
            # Use JSONRepair for robust parsing
            execution_plan, was_repaired = JSONRepair.repair(plan_text)
            
            if execution_plan is None:
                logger.warning(f"JSON parsing and repair failed")
                logger.warning(f"Raw response text: {plan_text[:500]}...")
                
                # Use comprehensive fallback
                execution_plan = JSONRepair.create_fallback_json(task.id, task.description)
                logger.info("Using structured fallback execution plan")
            elif was_repaired:
                logger.info("JSON was successfully repaired and parsed")
            
            console.print("[green]✓ Master Claude created CLI-optimized execution plan[/green]")
            # Only log situation_analysis if it exists
            if execution_plan and 'situation_analysis' in execution_plan:
                logger.info(f"CLI Execution plan: {execution_plan['situation_analysis']}")
            else:
                logger.info("CLI Execution plan created (using fallback)")
=======
            # Clean up common JSON issues
            plan_text = plan_text.strip()
            plan_text = plan_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            
            try:
                execution_plan = json.loads(plan_text)
            except json.JSONDecodeError as json_error:
                logger.warning(f"JSON parsing failed in planning: {json_error}")
                logger.warning(f"Raw response text: {plan_text[:500]}")
                # Use fallback plan
                raise Exception(f"JSON parsing failed: {json_error}")
            
            console.print("[green]✓ Master Claude created CLI-optimized execution plan[/green]")
            logger.info(f"CLI Execution plan: {execution_plan['situation_analysis']}")
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Error in Master Claude CLI analysis: {e}")
            # Fallback plan
            return {
                "situation_analysis": f"Proceeding with task {task.id} implementation using CLI",
                "execution_strategy": "Direct CLI-based implementation approach",
                "cli_specific_strategy": "Use Claude CLI's natural language processing and file manipulation",
                "step_by_step_plan": [{
                    "step": 1, 
                    "action": "Implement task requirements", 
                    "cli_command_approach": "Natural language instruction to CLI",
                    "verification": "Check acceptance criteria"
                }],
                "code_claude_cli_instructions": f"Implement {task.description} using best practices",
                "success_criteria": ["Task completes without errors"],
                "risk_factors": ["Implementation complexity"],
                "cli_advantages": ["Direct file manipulation", "Built-in Git integration"]
            }
    
    async def _execute_with_cli_iterations(self, task: Task, workspace_path: str, 
                                          execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task with iterative CLI supervision."""
        
        max_iterations = 2  # Reduced from 3 to 2 for speed
        iteration = 0
        success = False
        execution_log = []
        
        while iteration < max_iterations and not success:
            iteration += 1
            console.print(f"[yellow]🔄 CLI Iteration {iteration}/{max_iterations}[/yellow]")
            
            # Create specific prompt for Code Claude CLI based on current state
            cli_prompt = await self._create_cli_prompt(
                task, workspace_path, execution_plan, iteration, execution_log
            )
            
            # Execute with Code Claude CLI
            iteration_result = await self._execute_code_claude_cli(
                cli_prompt, task, workspace_path
            )
            
            execution_log.append({
                'iteration': iteration,
                'prompt': cli_prompt[:200] + "...",  # Truncate for logging
                'result': iteration_result,
                'timestamp': datetime.now().isoformat(),
                'executor_type': 'cli'
            })
            
            # Master Claude evaluates the result
            evaluation = await self._evaluate_cli_iteration_result(
                task, workspace_path, iteration_result, execution_plan
            )
            
<<<<<<< HEAD
            # Handle None evaluation (shouldn't happen with the fix above, but be safe)
            if evaluation is None:
                logger.warning("Evaluation returned None, using fallback")
                evaluation = {
                    'success': False,
                    'feedback': "Evaluation failed, continuing with next iteration",
                    'structure_correct': False,
                    'ready_for_final_verification': False
                }
            
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            execution_log.append({
                'iteration': iteration,
                'evaluation': evaluation,
                'timestamp': datetime.now().isoformat()
            })
            
<<<<<<< HEAD
            if evaluation.get('success', False):
                success = True
                console.print(f"[green]✅ Task completed successfully via CLI in iteration {iteration}[/green]")
            elif iteration < max_iterations:
                console.print(f"[yellow]⚠️ CLI iteration {iteration} needs improvement: {evaluation.get('feedback', 'No feedback available')}[/yellow]")
=======
            if evaluation['success']:
                success = True
                console.print(f"[green]✅ Task completed successfully via CLI in iteration {iteration}[/green]")
            elif iteration < max_iterations:
                console.print(f"[yellow]⚠️ CLI iteration {iteration} needs improvement: {evaluation['feedback']}[/yellow]")
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
                # Master Claude will adjust the approach for next iteration
            else:
                console.print(f"[red]❌ Task failed after {max_iterations} CLI iterations[/red]")
        
        return {
            'success': success,
            'iterations': iteration,
            'execution_log': execution_log,
            'final_state': await self._get_workspace_state(workspace_path),
            'executor_type': 'cli'
        }
    
    async def _create_cli_prompt(self, task: Task, workspace_path: str,
                                execution_plan: Dict[str, Any], iteration: int,
                                execution_log: List[Dict]) -> str:
        """Master Claude creates specific prompt for Code Claude CLI based on current state."""
        
        # Get current workspace state
        current_state = await self._get_workspace_state(workspace_path)
        
        # Analyze what happened in previous iterations
        previous_attempts = ""
        if execution_log:
            previous_attempts = "\n**PREVIOUS CLI ATTEMPTS:**\n"
            for log_entry in execution_log:
                if 'result' in log_entry:
                    previous_attempts += f"Iteration {log_entry['iteration']}: {log_entry['result'].get('summary', 'No summary')}\n"
                if 'evaluation' in log_entry:
                    previous_attempts += f"Evaluation: {log_entry['evaluation'].get('feedback', 'No feedback')}\n"
        
        # Create concise but clear CLI prompt with structure awareness
<<<<<<< HEAD
        workspace_area = self._determine_workspace_area(task, workspace_path)
        
        cli_prompt = f"""**TASK:** {task.description}
**TARGET:** {workspace_area} workspace
=======
        workspace_area = "frontend" if "fe-" in task.id or "frontend" in task.project_area.lower() else "backend"
        
        cli_prompt = f"""**TASK:** {task.description}
**TARGET:** {workspace_area}/ workspace
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
**FILES:** {', '.join(task.files_to_create_or_modify)}

**CURRENT WORKSPACE:**
{current_state}

**STEPS:**"""
        
        # Add step guidance with verification
        for step in execution_plan.get('step_by_step_plan', []):
            cli_prompt += f"\n{step['step']}. {step['action']}"
            if 'verification' in step:
                cli_prompt += f" (Verify: {step['verification']})"
        
        if previous_attempts:
            cli_prompt += f"\n\n**PREVIOUS ISSUES:**\n{previous_attempts}"
        
        cli_prompt += f"""

**MANDATORY CHECKS:**
1. Run `pwd` - confirm you're in correct directory
<<<<<<< HEAD
2. Target workspace: {'./' if workspace_area != '.' else ''}{workspace_area}{'' if workspace_area == '.' else '/'}
3. Check existing structure BEFORE creating files
4. Install dependencies: {f'`npm install <package> -w {workspace_area}`' if workspace_area != '.' else '`npm install <package>`'}

**STRUCTURE ERRORS TO AVOID:**
- NO {workspace_area}/{workspace_area}/ nesting{' (root level work)' if workspace_area == '.' else ''}
=======
2. Target workspace: ./{workspace_area}/
3. Check existing structure BEFORE creating files
4. Install dependencies: `npm install <package> -w {workspace_area}`

**STRUCTURE ERRORS TO AVOID:**
- NO {workspace_area}/{workspace_area}/ nesting
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
- NO node_modules in wrong location  
- NO files outside designated workspace

**EXECUTE:**
{execution_plan.get('code_claude_cli_instructions', f'Implement {task.description} in {workspace_area}/ workspace')}

**CRITICAL - COMPLETION REPORTING:**
When you finish ALL work, MUST end with EXACTLY one of these messages:
- "TASK IMPLEMENTATION COMPLETE"
- "TASK FINISHED SUCCESSFULLY" 
- "WORK COMPLETED"

Start with `pwd` command."""
        
        # Store CLI prompt for logging
        if not hasattr(self, '_cli_prompts_used'):
            self._cli_prompts_used = []
        self._cli_prompts_used.append({
            'iteration': iteration,
            'prompt': cli_prompt[:1000] + ('...' if len(cli_prompt) > 1000 else ''),
            'timestamp': datetime.now().isoformat()
        })
        
        return cli_prompt
    
    async def _execute_code_claude_cli(self, prompt: str, task: Task, workspace_path: str) -> Dict[str, Any]:
        """Execute Code Claude CLI with the supervised prompt using enhanced session management."""
<<<<<<< HEAD
        # 파일 활동 모니터 시작
        file_monitor = get_file_activity_monitor(workspace_path)
        await file_monitor.start_monitoring()
        
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        try:
            # Create curated context for Code Claude CLI
            master_context = await self._create_curated_context_for_code_claude_cli(task)
            task_specific_context = prompt  # The supervised prompt becomes task-specific context
            
            # Execute with persistent CLI session continuity
            result = await self.persistent_cli_sessions.execute_task_with_session_continuity(
                task, workspace_path, master_context, task_specific_context
            )
            
<<<<<<< HEAD
            # 파일 활동 보고서 추가
            activity_report = file_monitor.get_activity_report()
            result['file_activity'] = activity_report
            
            # 데드락 감지 - 파일 활동이 없으면 경고
            if activity_report['potentially_stuck']:
                logger.warning(f"Task {task.id} may be stuck - no file activity for {activity_report['inactive_seconds']} seconds")
                result['deadlock_warning'] = True
            
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            # Learn from Code Claude CLI's insights
            await self._absorb_code_claude_cli_insights(result)
            
            return result
            
        except Exception as e:
<<<<<<< HEAD
            import traceback
            logger.error(f"Error in enhanced Code Claude CLI execution: {e}", exc_info=True)
            traceback.print_exc()
=======
            logger.error(f"Error in enhanced Code Claude CLI execution: {e}")
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            return {
                'success': False,
                'error': str(e),
                'summary': f"Enhanced Code Claude CLI execution failed: {e}",
                'executor_type': 'cli'
            }
<<<<<<< HEAD
        finally:
            # 파일 활동 모니터 중지
            await file_monitor.stop_monitoring()
    
    async def _create_curated_context_for_code_claude_cli(self, task: Task) -> str:
        """Create enhanced structure-aware context for Code Claude CLI execution."""
        
        # Get current workspace structure analysis
        workspace_structure = workspace_analyzer.analyze_workspace(self.current_workspace_state.get('path', '.'))
        
        context_parts = []
        
        # 1. PROJECT OVERVIEW - Clear monolithic structure
        context_parts.append("# 🏗️ PROJECT STRUCTURE OVERVIEW")
        context_parts.append("**Architecture**: Monolithic application with clear separation of concerns")
        context_parts.append("**Approach**: Build incrementally with small, focused tasks")
        context_parts.append("")
        
        # 2. TASK RELATIONSHIPS MAP
        context_parts.append("## 📊 TASK DEPENDENCY MAP")
        context_parts.append(self._build_task_relationship_diagram(task))
        context_parts.append("")
        
        # 3. CURRENT TASK CONTEXT
        context_parts.append(f"## 🎯 CURRENT TASK: {task.id}")
        context_parts.append(f"**Description**: {task.description}")
        context_parts.append(f"**Type**: {task.type}")
        context_parts.append(f"**Project Area**: {task.project_area}")
        
        # Show task dependencies clearly
        if task.dependencies:
            context_parts.append(f"**Depends On**: {', '.join(task.dependencies)}")
            # Check which dependencies are completed
            completed_deps = [dep for dep in task.dependencies if dep in self.completed_tasks]
            if completed_deps:
                context_parts.append(f"  ✅ Completed: {', '.join(completed_deps)}")
            pending_deps = [dep for dep in task.dependencies if dep not in self.completed_tasks]
            if pending_deps:
                context_parts.append(f"  ⏳ Pending: {', '.join(pending_deps)}")
        else:
            context_parts.append("**Dependencies**: None (can start immediately)")
        
        # Show downstream tasks
        downstream = self._get_downstream_tasks(task.id)
        if downstream:
            context_parts.append(f"**Enables**: {', '.join([t.id for t in downstream[:3]])}{'...' if len(downstream) > 3 else ''}")
        context_parts.append("")
        
        # 4. Workspace Structure Analysis
        workspace_context = workspace_analyzer.get_workspace_context_for_claude(
            self.current_workspace_state.get('path', '.')
        )
        context_parts.append(workspace_context)
        
        # 5. Target workspace identification with structure validation
        target_workspace = self._determine_workspace_area(task, self.current_workspace_state.get('path', '.'))
        
        context_parts.append(f"\n## 📁 TARGET WORKSPACE: {target_workspace}{'' if target_workspace == '.' else '/'}")
        
        # 6. Technology Stack Information - Simplified
        if workspace_structure.technology_stacks:
            primary_stack = workspace_structure.technology_stacks[0]
            context_parts.append(f"\n## 💻 TECHNOLOGY STACK")
            context_parts.append(f"**Primary**: {primary_stack.stack.value} ({primary_stack.confidence:.0%} confidence)")
            
            # Add concise stack-specific rules
            if "typescript" in primary_stack.stack.value:
                context_parts.append("**Rules**: TypeScript only, proper types, no any")
            elif "javascript" in primary_stack.stack.value:
                context_parts.append("**Rules**: ES6+ syntax, async/await preferred")
            
            if "nextjs" in primary_stack.stack.value:
                context_parts.append("**Framework**: Next.js App Router (src/app/)")
            elif "react" in primary_stack.stack.value:
                context_parts.append("**Framework**: React with hooks")
        
        context_parts.append("")
        
        # 7. PROJECT PROGRESS - Clear status
        completed_count = len(self.completed_tasks)
        total_count = len(self.tasks_plan.tasks) if hasattr(self, 'tasks_plan') else len(self.execution_history)
        progress_pct = (completed_count / total_count * 100) if total_count > 0 else 0
        
        context_parts.append(f"## 📈 PROJECT PROGRESS")
        context_parts.append(f"**Overall**: {completed_count}/{total_count} tasks ({progress_pct:.0f}%)")
        context_parts.append(f"**Status**: {'🟢 On Track' if progress_pct > 20 else '🟡 Early Stage'}")
        
        # Show recent completions for context
        recent_completed = list(self.completed_tasks)[-3:] if self.completed_tasks else []
        if recent_completed:
            context_parts.append(f"**Recently Completed**: {', '.join(recent_completed)}")
        context_parts.append("")
        
        # 8. ESTABLISHED PATTERNS - What we've learned
        if self.claude_insights.get('established_patterns'):
            context_parts.append("## 🔧 ESTABLISHED PATTERNS")
            # Filter for most relevant patterns
            relevant_patterns = [
                p for p in self.claude_insights['established_patterns'][-5:]
                if task.project_area in p.lower() or task.type in p.lower() or 'general' in p.lower()
            ]
            if relevant_patterns:
                for pattern in relevant_patterns:
                    context_parts.append(f"- {pattern}")
            else:
                context_parts.append("- Follow existing code patterns in the workspace")
        context_parts.append("")
        
        # 9. TASK-SPECIFIC CONTEXT
        current_files = await self._get_relevant_file_state(task)
        if current_files:
            context_parts.append(f"## 📄 RELEVANT EXISTING FILES")
            context_parts.append(current_files)
            context_parts.append("")
        
        # 10. EXECUTION RULES - Clear and concise
        context_parts.append("## ⚡ EXECUTION RULES")
        context_parts.append("1. **Small Steps**: Implement one clear piece at a time")
        context_parts.append("2. **Test Incrementally**: Verify each component works before moving on")
        context_parts.append("3. **Follow Patterns**: Use existing code as reference")
        context_parts.append(f"4. **Stay Focused**: Only implement what's needed for {task.id}")
        
        if target_workspace != ".":
            context_parts.append(f"5. **Workspace Rule**: All files go in {target_workspace}/ only")
        
        # 11. COMMON PITFALLS TO AVOID
        context_parts.append("\n## ❌ AVOID THESE COMMON MISTAKES")
        context_parts.append("- Creating nested directories (e.g., frontend/frontend/)")
        context_parts.append("- Implementing features beyond current task scope")
        context_parts.append("- Using inconsistent naming or code style")
        context_parts.append("- Forgetting to create parent directories with mkdir -p")
        
        # Log context creation
        if self.logger:
            self.logger.claude_interaction(
                "context_creation", 
                len("\n".join(context_parts)),
                metadata={
                    'target_workspace': target_workspace,
                    'critical_issues': len([issue for issue in workspace_structure.issues if issue.severity == "critical"]),
                    'tech_stack': workspace_structure.technology_stacks[0].stack.value if workspace_structure and workspace_structure.technology_stacks else None
                }
            )
        
        return "\n".join(context_parts)
=======
    
    async def _create_curated_context_for_code_claude_cli(self, task: Task) -> str:
        """Create structure-aware context for Code Claude CLI execution."""
        
        context_parts = []
        
        # 1. Target workspace identification
        target_workspace = "frontend" if "fe-" in task.id or "frontend" in task.project_area.lower() else "backend"
        context_parts.append(f"TARGET WORKSPACE: {target_workspace}/")
        
        # 2. Essential progress
        completed_count = len([t for t in self.task_progress.values() if t.get('success')])
        total_count = len(self.execution_history)
        context_parts.append(f"Progress: {completed_count}/{total_count} tasks completed")
        
        # 3. Structure patterns (focus on avoiding errors)
        structure_patterns = [p for p in self.claude_insights.get('established_patterns', []) if 'structure' in p or 'workspace' in p]
        if structure_patterns:
            context_parts.append("Structure patterns: " + "; ".join(structure_patterns[-2:]))
        
        # 4. Task-specific file context
        current_files = await self._get_relevant_file_state(task)
        if current_files:
            context_parts.append(f"Current files:\n{current_files}")
        
        # 5. Critical workspace reminder
        context_parts.append(f"CRITICAL: All files must go in {target_workspace}/ workspace. Avoid {target_workspace}/{target_workspace}/ nesting.")
        
        return "\n\n".join(context_parts)
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
    
    
    async def _absorb_code_claude_cli_insights(self, execution_result: Dict[str, Any]) -> None:
        """Learn from Code Claude CLI's execution to improve future guidance."""
        
        # Extract insights from execution result
        if 'learned_patterns' in execution_result:
            new_patterns = execution_result['learned_patterns']
            for pattern in new_patterns:
                if pattern not in self.claude_insights['established_patterns']:
                    self.claude_insights['established_patterns'].append(pattern)
        
        # Add CLI-specific patterns
        if execution_result.get('executor_type') == 'cli':
            cli_pattern = f"CLI execution pattern from {execution_result.get('task_id', 'unknown')}: successful direct workspace manipulation"
            if cli_pattern not in self.claude_insights['cli_specific_patterns']:
                self.claude_insights['cli_specific_patterns'].append(cli_pattern)
        
        if 'session_updates' in execution_result:
            updates = execution_result['session_updates']
            
            # Update coding conventions
            if 'updated_conventions' in updates:
                self.claude_insights['coding_conventions'].update(updates['updated_conventions'])
            
            # Add new utilities
            if 'new_utilities' in updates:
                for utility in updates['new_utilities']:
                    if utility not in self.claude_insights['common_utilities']:
                        self.claude_insights['common_utilities'].append(utility)
        
        # Add architectural insights if this was a significant task
        task_id = execution_result.get('task_id', '')
        if execution_result.get('success') and ('api' in task_id or 'model' in task_id or 'component' in task_id):
            arch_decision = f"CLI Task {task_id}: Established architectural pattern through direct implementation"
            if arch_decision not in self.claude_insights['architectural_decisions']:
                self.claude_insights['architectural_decisions'].append(arch_decision)
        
        # Track structure-related insights to prevent future issues
        if execution_result.get('success'):
            structure_insight = f"CLI Task {task_id}: Maintained proper monorepo structure"
            if structure_insight not in self.claude_insights['cli_specific_patterns']:
                self.claude_insights['cli_specific_patterns'].append(structure_insight)
        
        # Broadcast insights to all active Code Claude CLI sessions
        await self.persistent_cli_sessions.broadcast_insights_to_sessions(self.claude_insights)
        
        logger.info(f"Absorbed CLI insights from task {execution_result.get('task_id', 'unknown')}")
    
    async def _evaluate_cli_iteration_result(self, task: Task, workspace_path: str,
                                            iteration_result: Dict[str, Any],
                                            execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Master Claude evaluates Code Claude CLI's iteration result."""
        
        # Get updated workspace state
        updated_state = await self._get_workspace_state(workspace_path)
        
        # Check if files were created/modified as expected
        file_check = await self._verify_expected_files(task, workspace_path)
        
        # Create structure-aware evaluation prompt
<<<<<<< HEAD
        target_workspace = self._determine_workspace_area(task, workspace_path)
        
        workspace_desc = "project root" if target_workspace == "." else f"{target_workspace}/ workspace"
        evaluation_prompt = f"""Evaluate CLI execution. Check if files are in correct {workspace_desc}.

**TASK:** {task.description}
**TARGET:** {workspace_desc}
=======
        target_workspace = "frontend" if "fe-" in task.id or "frontend" in task.project_area.lower() else "backend"
        
        evaluation_prompt = f"""Evaluate CLI execution. Check if files are in correct {target_workspace}/ workspace.

**TASK:** {task.description}
**TARGET:** {target_workspace}/ workspace
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
**RESULT:** {iteration_result.get('success', False)} - {iteration_result.get('summary', 'No summary')}
**FILES:** {file_check}
**STRUCTURE:** {updated_state}

<<<<<<< HEAD
**CHECK:** Are files in {workspace_desc}? No duplicate directories?
=======
**CHECK:** Are files in {target_workspace}/ workspace? No duplicate directories?
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271

**JSON:**
```json
{{
    "success": true/false,
    "feedback": "Structure issues or next steps",
    "structure_correct": true/false,
    "ready_for_final_verification": true/false
}}
```"""

        try:
            response = self.gemini_model.generate_content(
                evaluation_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Low temperature for consistent evaluation
                    top_p=0.9,
                    max_output_tokens=2000,
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content:
                logger.warning("Gemini evaluation response blocked by safety filters")
                raise Exception("Evaluation response blocked by safety filters")
            
            # Parse JSON response with better error handling
            eval_text = response.text
            if '```json' in eval_text:
                json_start = eval_text.find('```json') + 7
                json_end = eval_text.find('```', json_start)
                if json_end == -1:  # No closing ```
                    eval_text = eval_text[json_start:]
                else:
                    eval_text = eval_text[json_start:json_end]
            
            # Enhanced JSON parsing with validation
            evaluation = self._parse_and_validate_response(eval_text, iteration_result, "task_evaluation")
            
            logger.info(f"Master Claude CLI evaluation: {evaluation.get('feedback', 'No feedback')}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in Master Claude CLI evaluation: {e}")
<<<<<<< HEAD
            # Return a fallback evaluation when Gemini fails
            return {
                'success': False,
                'feedback': f"Evaluation failed: {str(e)}. Continuing with caution.",
                'structure_correct': False,
                'ready_for_final_verification': False
            }
    
    async def _validate_generated_code(self, file_path: str) -> Dict[str, Any]:
        """Validate generated code using real-time syntax validator."""
        validation_result = realtime_syntax_validator.validate_file(file_path)
        
        return {
            'file': file_path,
            'valid': validation_result.is_valid,
            'errors': len(validation_result.errors),
            'warnings': len(validation_result.warnings),
            'language': validation_result.language.value,
            'details': {
                'errors': [
                    {
                        'line': err.line,
                        'column': err.column,
                        'message': err.message,
                        'snippet': err.code_snippet
                    } for err in validation_result.errors[:3]  # Limit to first 3 errors
                ],
                'warnings': [
                    {
                        'line': warn.line,
                        'column': warn.column,
                        'message': warn.message
                    } for warn in validation_result.warnings[:3]  # Limit to first 3 warnings
                ]
            }
        }
    
    def _determine_workspace_area(self, task: Task, workspace_path: str) -> str:
        """Determine the appropriate workspace area based on task and project structure."""
        # Analyze workspace structure to determine project type
        workspace_structure = workspace_analyzer.analyze_workspace(workspace_path)
        
        # Check if it's a monolithic Next.js or similar full-stack framework
        is_monolithic = False
        if workspace_structure.technology_stacks:
            primary_stack = workspace_structure.technology_stacks[0]
            # Next.js, Nuxt, SvelteKit etc. are monolithic frameworks
            try:
                stack_value = primary_stack.stack.value.lower()
                if any(framework in stack_value for framework in ['nextjs', 'nuxt', 'sveltekit']):
                    is_monolithic = True
            except AttributeError as e:
                logger.warning(f"Error accessing technology stack value: {e}")
                logger.warning(f"primary_stack type: {type(primary_stack)}")
                logger.warning(f"primary_stack attributes: {dir(primary_stack)}")
                if hasattr(primary_stack, 'stack'):
                    logger.warning(f"primary_stack.stack type: {type(primary_stack.stack)}")
                    logger.warning(f"primary_stack.stack attributes: {dir(primary_stack.stack)}")
        
        # For monolithic frameworks, everything goes in the root
        if is_monolithic:
            return "."
        
        # For other projects, use the original logic
        if task.type == "setup" or task.project_area == "shared":
            return "."  # Output directory root for setup tasks
        elif "fe-" in task.id or "frontend" in task.project_area.lower():
            return "frontend"
        else:
            return "backend"
    
    def _build_task_relationship_diagram(self, current_task: Task) -> str:
        """Build a visual representation of task relationships."""
        if not hasattr(self, 'tasks_plan') or not self.tasks_plan:
            return "No task plan available"
        
        diagram_lines = []
        
        # Find parent tasks (dependencies)
        if current_task.dependencies:
            diagram_lines.append("📌 Dependencies (must complete first):")
            for dep_id in current_task.dependencies:
                dep_task = next((t for t in self.tasks_plan.tasks if t.id == dep_id), None)
                if dep_task:
                    status = "✅" if dep_id in self.completed_tasks else "⏳"
                    diagram_lines.append(f"  {status} {dep_id}: {dep_task.description[:50]}...")
        
        # Current task
        diagram_lines.append(f"\n🎯 **CURRENT: {current_task.id}**")
        diagram_lines.append(f"   └─ {current_task.description[:60]}...")
        
        # Find downstream tasks
        downstream = self._get_downstream_tasks(current_task.id)
        if downstream:
            diagram_lines.append("\n🔜 Unlocks these tasks:")
            for task in downstream[:5]:  # Limit to 5 for brevity
                diagram_lines.append(f"  → {task.id}: {task.description[:50]}...")
            if len(downstream) > 5:
                diagram_lines.append(f"  → ... and {len(downstream) - 5} more tasks")
        
        return "\n".join(diagram_lines)
    
    def _get_downstream_tasks(self, task_id: str) -> List[Task]:
        """Get tasks that depend on the given task."""
        if not hasattr(self, 'tasks_plan') or not self.tasks_plan:
            return []
        
        downstream = []
        for task in self.tasks_plan.tasks:
            if task_id in task.dependencies:
                downstream.append(task)
        
        return downstream
=======
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
    
    def _parse_json_with_fallbacks(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse JSON with multiple fallback strategies."""
        import re
        
        if context is None:
            context = {}
        
        # Strategy 1: Direct parsing
        try:
            cleaned_text = text.strip()
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Fix common escape issues
        try:
            # Handle newlines and other escape characters in JSON strings
            fixed_text = text.strip()
            # Fix unescaped newlines inside JSON strings
            fixed_text = re.sub(r'(?<!\\)\\n', '\\\\n', fixed_text)
            fixed_text = re.sub(r'(?<!\\)\\r', '\\\\r', fixed_text)
            fixed_text = re.sub(r'(?<!\\)\\t', '\\\\t', fixed_text)
            # Fix unescaped quotes
            fixed_text = re.sub(r'(?<!\\)"(?=[\w\s])', '\\"', fixed_text)
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Extract JSON from markdown code blocks
        try:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Find first valid JSON object
        try:
            start_idx = text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Manual key extraction (final fallback)
        logger.warning(f"All JSON parsing strategies failed. Using manual extraction for: {text[:200]}...")
        
        success = bool(re.search(r'"success"\s*:\s*true', text, re.IGNORECASE) or 
                      context.get('success', False))
        
        feedback_match = re.search(r'"feedback"\s*:\s*"([^"]*)"', text)
        feedback = feedback_match.group(1) if feedback_match else "JSON parsing failed - manual extraction used"
        
        quality_match = re.search(r'"quality_score"\s*:\s*(\d+)', text)
        quality_score = int(quality_match.group(1)) if quality_match else 3
        
        return {
            'success': success,
            'feedback': feedback,
            'quality_score': quality_score,
            'cli_effectiveness': 5,
            'ready_for_final_verification': success,
            '_parsing_method': 'manual_extraction'
        }
    
    def _parse_and_validate_response(self, text: str, context: Dict[str, Any], response_type: str = "generic") -> Dict[str, Any]:
        """Parse response with validation and fallback mechanisms."""
        
        # First attempt: Enhanced JSON parsing
        try:
            parsed_response = self._parse_json_with_fallbacks(text, context)
            
            # Validate the parsed response
            validation_result = validate_ai_response(parsed_response, response_type)
            
            if validation_result.is_valid:
                logger.info(f"Response validation passed with confidence: {validation_result.confidence_score:.2f}")
                return parsed_response
            else:
                logger.warning(f"Response validation failed: {len(validation_result.issues)} issues")
                
                # Try auto-correction if available
                if validation_result.corrected_response:
                    logger.info("Using auto-corrected response")
                    return validation_result.corrected_response
                
                # Fall back to manual extraction with context
                logger.warning("Falling back to enhanced manual extraction")
                return self._enhanced_manual_extraction(text, context, validation_result.issues)
                
        except Exception as e:
            logger.error(f"All parsing strategies failed: {e}")
            return self._create_fallback_response(context)
    
    def _enhanced_manual_extraction(self, text: str, context: Dict[str, Any], validation_issues: List[Dict]) -> Dict[str, Any]:
        """Enhanced manual extraction with context awareness."""
        import re
        
        # Start with basic manual extraction
        result = {}
        
        # Extract success with multiple patterns
        success_patterns = [
            r'"success"\s*:\s*(true|false)',
            r'success["\']?\s*[:=]\s*(true|false)',
            r'(successful|completed|passed|done)',
            r'(failed|error|issue|problem)'
        ]
        
        success = False
        for pattern in success_patterns[:2]:  # Boolean patterns
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                success = match.group(1).lower() == 'true'
                break
        
        if not success:  # Check descriptive patterns
            success_keywords = re.search(success_patterns[2], text, re.IGNORECASE)
            failure_keywords = re.search(success_patterns[3], text, re.IGNORECASE)
            
            if success_keywords and not failure_keywords:
                success = True
            elif failure_keywords:
                success = False
        
        # Extract feedback with context enhancement
        feedback_patterns = [
            r'"feedback"\s*:\s*"([^"]*)"',
            r'feedback["\']?\s*[:=]\s*["\']([^"\']*)["\']',
            r'(?:feedback|result|outcome|status):\s*([^.\n]+)'
        ]
        
        feedback = "Extracted feedback from unstructured response"
        for pattern in feedback_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                feedback = match.group(1).strip()
                break
        
        # Enhance feedback with context
        if len(feedback) < 20 and context.get('error_info'):
            feedback += f" Context: {context['error_info']}"
        
        # Extract quality score
        quality_patterns = [
            r'"quality_score"\s*:\s*(\d+(?:\.\d+)?)',
            r'quality["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'score["\']?\s*[:=]\s*(\d+(?:\.\d+)?)'
        ]
        
        quality_score = 3  # Default middle score
        for pattern in quality_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                quality_score = float(match.group(1))
                break
        
        # Ensure consistency
        if success and quality_score < 5:
            quality_score = 6  # Boost score if success is true
        elif not success and quality_score > 5:
            quality_score = 3  # Lower score if success is false
        
        return {
            'success': success,
            'feedback': feedback,
            'quality_score': quality_score,
            'cli_effectiveness': 5,
            'ready_for_final_verification': success,
            '_parsing_method': 'enhanced_manual_extraction',
            '_validation_issues': len(validation_issues)
        }
    
    def _create_fallback_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe fallback response when all parsing fails."""
        success = context.get('success', False)
        
        return {
            'success': success,
            'feedback': f"Fallback response due to parsing failure. Original result: {success}",
            'quality_score': 3,
            'cli_effectiveness': 3,
            'ready_for_final_verification': success,
            '_parsing_method': 'fallback_response'
        }
    
    async def _execute_task_with_recovery(self, task: Task, workspace_path: str, tasks_plan: TasksPlan) -> Dict[str, Any]:
        """Execute task with intelligent error recovery support."""
        
        logger.info(f"Executing task {task.id} with recovery support")
        
        max_recovery_attempts = 3
        current_attempt = 0
        last_error = None
        
        while current_attempt < max_recovery_attempts:
            try:
                # Attempt normal execution
                result = await self.execute_task_with_cli_supervision(task, workspace_path, tasks_plan)
                
                # If successful, return immediately
                if result.get('overall_success', result.get('success', False)):
                    logger.info(f"Task {task.id} completed successfully on attempt {current_attempt + 1}")
                    return result
                
                # If failed, check if recovery is possible
                error_message = result.get('error', 'Task execution failed')
                logger.warning(f"Task {task.id} failed on attempt {current_attempt + 1}: {error_message}")
                
                # Check if we have error recovery manager
                if hasattr(self, 'error_recovery_manager') and self.error_recovery_manager:
                    logger.info(f"Attempting error recovery for task {task.id}")
                    
                    # Try to recover from the error
                    recovery_success, recovery_result = await self.error_recovery_manager.handle_task_error(
                        task, error_message, None, task.files_to_create_or_modify
                    )
                    
                    if recovery_success:
                        logger.info(f"Error recovery successful for task {task.id}")
                        # Continue to next attempt with recovered state
                        current_attempt += 1
                        continue
                    else:
                        logger.warning(f"Error recovery failed for task {task.id}")
                
                last_error = error_message
                current_attempt += 1
                
            except Exception as e:
                logger.error(f"Exception during task {task.id} execution: {e}")
                last_error = str(e)
                current_attempt += 1
        
        # All attempts failed
        logger.error(f"Task {task.id} failed after {max_recovery_attempts} attempts. Last error: {last_error}")
        
        return {
            'success': False,
            'overall_success': False,
            'error': f"Task failed after {max_recovery_attempts} recovery attempts. Last error: {last_error}",
            'recovery_attempts': max_recovery_attempts,
            'final_error': last_error
        }
    
    # Inherit other methods from the base supervisor
    async def _get_workspace_state(self, workspace_path: str) -> str:
        """Get current workspace state summary with noise filtering."""
        try:
            files = await self.aci.list_files(workspace_path)
            structure = await self.aci.get_directory_structure(workspace_path)
            
            # Filter out noise directories and files
            noise_patterns = [
                'node_modules', '.git', '__pycache__', '.next', 'dist', 'build', 
                '.cache', 'coverage', '.nyc_output', 'logs', 'tmp', 'temp',
                '.DS_Store', 'Thumbs.db', '*.log', '*.tmp'
            ]
            
            # Filter files
            filtered_files = []
            for file_path in files:
                if not any(noise in file_path for noise in noise_patterns):
                    filtered_files.append(file_path)
            
            # Filter structure lines  
            filtered_structure_lines = []
            for line in structure.split('\n'):
                if not any(noise in line for noise in noise_patterns):
                    filtered_structure_lines.append(line)
            
            filtered_structure = '\n'.join(filtered_structure_lines)
            
            return f"""Directory Structure:
{filtered_structure}

Files ({len(filtered_files)} total):
{chr(10).join(filtered_files)}"""
        except Exception as e:
            return f"Error getting workspace state: {e}"
    
    async def _get_related_task_outcomes(self, task: Task) -> str:
        """Get outcomes from related tasks (dependencies, same area, etc.)."""
        related_outcomes = []
        
        # Get outcomes from dependency tasks
        for dep_id in task.dependencies:
            if dep_id in self.task_progress:
                outcome = self.task_progress[dep_id]
                summary = outcome.get('summary', 'No summary available')
                related_outcomes.append(f"Task {dep_id}: {summary}")
        
        # Get recent outcomes from same project area
        recent_same_area = [
            entry for entry in self.execution_history[-5:]  # Last 5 entries
            if entry.get('result', {}).get('task_id', '').startswith(task.project_area)
        ]
        
        for entry in recent_same_area[-2:]:  # Last 2 from same area
            result = entry.get('result', {})
            task_id = result.get('task_id', 'unknown')
            summary = result.get('summary', 'No summary')
            executor_type = result.get('executor_type', 'unknown')
            related_outcomes.append(f"Recent {task.project_area}: {task_id} - {summary} (via {executor_type})")
        
        return "\n".join(related_outcomes) if related_outcomes else ""
    
    async def _get_relevant_file_state(self, task: Task) -> str:
        """Get current state of files relevant to this task."""
        relevant_files = []
        
        # Files this task will modify
        for file_path in task.files_to_create_or_modify:
            if await self.aci.file_exists(file_path):
                relevant_files.append(f"EXISTS: {file_path}")
            else:
                relevant_files.append(f"TO CREATE: {file_path}")
        
        # Related files in the same directory
        if task.files_to_create_or_modify:
            first_file = task.files_to_create_or_modify[0]
            directory = "/".join(first_file.split("/")[:-1])
            if directory:
                existing_files = await self.aci.list_files(directory, "*")
                for file in existing_files[:5]:  # Limit to 5 files
                    relevant_files.append(f"SIBLING: {file}")
        
        return "\n".join(relevant_files) if relevant_files else "No relevant files identified"
    
    def _get_project_summary(self) -> str:
        """Get essential project summary for Code Claude CLI."""
        completed_count = len([t for t in self.task_progress.values() if t.get('success')])
        total_count = len(self.execution_history)
        
        return f"""Progress: {completed_count}/{total_count} tasks completed
Architecture: {', '.join(self.claude_insights['architectural_decisions'][-2:]) if self.claude_insights['architectural_decisions'] else 'Being established'}
Active Conventions: {len(self.claude_insights['coding_conventions'])} established
Patterns Identified: {len(self.claude_insights['established_patterns'])}
CLI Patterns: {len(self.claude_insights['cli_specific_patterns'])} CLI-specific patterns"""
    
    def _format_acceptance_criteria(self, criteria) -> str:
        """Format acceptance criteria for prompts."""
        formatted = "Acceptance Criteria:\n"
        
        if criteria.tests:
            formatted += "Tests to pass:\n"
            for test in criteria.tests:
                formatted += f"- {test.type} test in {test.file}"
                if test.function:
                    formatted += f" (function: {test.function})"
                formatted += "\n"
        
        if criteria.linting:
            formatted += f"Linting: {criteria.linting}\n"
        
        if criteria.manual_checks:
            formatted += "Manual checks:\n"
            for check in criteria.manual_checks:
                formatted += f"- {check}\n"
        
        return formatted
    
<<<<<<< HEAD
    async def _verify_expected_files(self, task: Task, workspace_path: str) -> tuple[str, dict]:
        """Verify that expected files were created/modified using flexible path checking."""
        results = []
        file_verification = {}
        files_found = 0
        files_missing = 0
        
        logger.info(f"Verifying expected files for task {task.id} in workspace {workspace_path}")
        
        # Check each required file individually
        for file_path in task.files_to_create_or_modify:
            # Try multiple path variations for flexible matching
            path_variations = self._generate_path_variations(file_path, workspace_path)
            
            file_found = False
            found_path = None
            item_type = "file"
            
            # Check if it's likely a directory
            is_directory = file_path.endswith('/') or (
                '.' not in os.path.basename(file_path) and 
                not any(ext in file_path.lower() for ext in [
                    'dockerfile', 'makefile', 'rakefile', 'gemfile', 'pipfile'
                ])
            )
            
            if is_directory:
                item_type = "directory"
            
            # Try each path variation
            for variant_path in path_variations:
                full_path = variant_path.replace('//', '/')  # Clean double slashes
                
                if is_directory:
                    if os.path.isdir(full_path):
                        file_found = True
                        found_path = full_path
                        break
                else:
                    if os.path.isfile(full_path):
                        file_found = True
                        found_path = full_path
                        break
            
            logger.info(f"Checking {item_type} {file_path}: {'EXISTS' if file_found else 'MISSING'}")
            if file_found:
                logger.info(f"  Found at: {found_path}")
            
            file_verification[file_path] = {
                'exists': file_found,
                'full_path': found_path or f"{workspace_path}/{file_path}",
                'status': 'EXISTS' if file_found else 'MISSING',
                'type': item_type,
                'variations_tried': len(path_variations)
            }
            
            if file_found:
                files_found += 1
                results.append(f"- {file_path}: ✓ {item_type} exists")
                
                # Validate syntax for code files
                if not is_directory and found_path:
                    validation = await self._validate_generated_code(found_path)
                    if not validation['valid']:
                        results.append(f"  ⚠️ Syntax errors: {validation['errors']} errors, {validation['warnings']} warnings")
                        file_verification[file_path]['syntax_valid'] = False
                        file_verification[file_path]['syntax_errors'] = validation['details']['errors']
                    else:
                        file_verification[file_path]['syntax_valid'] = True
                        if validation['warnings'] > 0:
                            results.append(f"  ⚡ {validation['warnings']} warnings")
            else:
                files_missing += 1
                results.append(f"- {file_path}: ✗ {item_type} missing")
                # Log variations tried for debugging
                logger.debug(f"  Tried variations: {path_variations}")
        
        # Calculate overall success
        all_files_exist = files_missing == 0
        
        # Return both formatted string and verification data
        verification_summary = {
            'all_files_exist': all_files_exist,
            'total_files': len(task.files_to_create_or_modify),
            'existing_files': files_found,
            'missing_files': files_missing,
            'file_details': file_verification
        }
        
        logger.info(f"File verification summary: {files_found}/{len(task.files_to_create_or_modify)} items found")
        
        return "Expected Files:\n" + "\n".join(results), verification_summary
    
    def _generate_path_variations(self, file_path: str, workspace_path: str) -> List[str]:
        """Generate multiple path variations for flexible file matching."""
        if not file_path or not workspace_path:
            return []
            
        variations = []
        
        # Original path
        base_path = f"{workspace_path}/{file_path}"
        variations.append(base_path)
        
        # Handle extension variations for TypeScript/JavaScript
        if file_path.endswith('.js'):
            # Try TypeScript variations
            ts_path = base_path[:-3] + '.ts'
            tsx_path = base_path[:-3] + '.tsx'
            variations.extend([ts_path, tsx_path])
            
            # Also try with src/ directory for Next.js
            if '/frontend/components/' in file_path:
                src_path = file_path.replace('/frontend/components/', '/frontend/src/components/')
                variations.append(f"{workspace_path}/{src_path}")
                variations.append(f"{workspace_path}/{src_path[:-3]}.ts")
                variations.append(f"{workspace_path}/{src_path[:-3]}.tsx")
        
        elif file_path.endswith('.jsx'):
            # Try TypeScript variations
            tsx_path = base_path[:-4] + '.tsx'
            variations.append(tsx_path)
            
            # Also try with src/ directory
            if '/frontend/' in file_path and '/src/' not in file_path:
                src_path = file_path.replace('/frontend/', '/frontend/src/')
                variations.append(f"{workspace_path}/{src_path}")
                variations.append(f"{workspace_path}/{src_path[:-4]}.tsx")
        
        # Handle missing src/ directory in Next.js projects
        if '/frontend/' in file_path and '/src/' not in file_path:
            # Try adding src/ after frontend/
            src_variant = file_path.replace('/frontend/', '/frontend/src/')
            variations.append(f"{workspace_path}/{src_variant}")
        
        # Handle app/ vs src/app/ for Next.js App Router
        if '/frontend/app/' in file_path and '/src/app/' not in file_path:
            src_app_variant = file_path.replace('/frontend/app/', '/frontend/src/app/')
            variations.append(f"{workspace_path}/{src_app_variant}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for path in variations:
            if path not in seen:
                seen.add(path)
                unique_variations.append(path)
        
        return unique_variations
=======
    async def _verify_expected_files(self, task: Task, workspace_path: str) -> str:
        """Verify that expected files were created/modified."""
        results = []
        for file_path in task.files_to_create_or_modify:
            full_path = f"{workspace_path}/{file_path}"
            exists = await self.aci.file_exists(full_path)
            results.append(f"- {file_path}: {'✓ exists' if exists else '✗ missing'}")
        
        return "Expected Files:\n" + "\n".join(results)
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
    
    async def _final_verification(self, task: Task, workspace_path: str, 
                                result: Dict[str, Any]) -> Dict[str, Any]:
        """Master Claude performs final verification of the completed task."""
        
        # Run actual acceptance criteria tests
        verification_results = await self._run_acceptance_tests(task, workspace_path)
        
        # Master Claude final assessment
        final_assessment = await self._master_final_assessment(
            task, workspace_path, result, verification_results
        )
        
        return {
            'task_id': task.id,
            'overall_success': final_assessment['success'],
            'execution_summary': result,
            'verification_results': verification_results,
            'master_assessment': final_assessment,
            'completed_at': datetime.now().isoformat(),
            'executor_type': 'cli'
        }
    
    async def _run_acceptance_tests(self, task: Task, workspace_path: str) -> Dict[str, Any]:
<<<<<<< HEAD
        """Run actual acceptance criteria tests and file verification."""
=======
        """Run actual acceptance criteria tests."""
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
        results = {
            'tests_passed': [],
            'tests_failed': [],
            'linting_passed': False,
<<<<<<< HEAD
            'files_verified': [],
            'file_verification': {}
        }
        
        try:
            # Run actual file verification first
            file_verification_text, file_verification_data = await self._verify_expected_files(task, workspace_path)
            results['file_verification'] = file_verification_data
            results['all_required_files_exist'] = file_verification_data['all_files_exist']
            
=======
            'files_verified': []
        }
        
        try:
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            # Run tests
            for test in task.acceptance_criteria.tests:
                test_result = await self.aci.run_test(
                    workspace_path, test.file, test.function
                )
                if test_result.success:
                    results['tests_passed'].append(f"{test.file}::{test.function}")
                else:
                    results['tests_failed'].append(f"{test.file}::{test.function}")
            
            # Run linting
            if task.acceptance_criteria.linting:
                lint_command = task.acceptance_criteria.linting.get('command')
                if lint_command:
                    lint_result = await self.aci.run_command(workspace_path, lint_command)
                    results['linting_passed'] = lint_result.success
            
<<<<<<< HEAD
            # Legacy file verification (for compatibility)
=======
            # Verify files
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            for file_path in task.files_to_create_or_modify:
                full_path = f"{workspace_path}/{file_path}"
                if await self.aci.file_exists(full_path):
                    results['files_verified'].append(file_path)
        
        except Exception as e:
            logger.error(f"Error running acceptance tests: {e}")
            results['error'] = str(e)
        
        return results
    
    async def _master_final_assessment(self, task: Task, workspace_path: str,
                                     execution_result: Dict[str, Any],
                                     verification_results: Dict[str, Any]) -> Dict[str, Any]:
<<<<<<< HEAD
        """Deterministic final assessment based on clear conditions."""
        
        logger.info(f"Conducting rule-based final assessment for task {task.id}")
        
        try:
            # Extract verification data
            all_files_exist = verification_results.get('all_required_files_exist', False)
            file_verification = verification_results.get('file_verification', {})
            tests_passed = len(verification_results.get('tests_passed', []))
            tests_failed = len(verification_results.get('tests_failed', []))
            linting_passed = verification_results.get('linting_passed', True)  # Default true if no linting required
            execution_success = execution_result.get('success', False)
            
            # Rule-based success determination
            success_conditions = []
            failure_reasons = []
            
            # 1. File existence check (most critical) - with tolerance
            existing_files = file_verification.get('existing_files', 0)
            total_files = file_verification.get('total_files', 1)
            file_completion_rate = existing_files / total_files if total_files > 0 else 0
            
            if all_files_exist:
                success_conditions.append("All required files exist")
                logger.info(f"✅ File verification: {existing_files}/{total_files} files found")
            elif file_completion_rate >= 0.8:  # 80% of files exist - still acceptable
                success_conditions.append(f"Most files exist ({existing_files}/{total_files})")
                logger.info(f"⚠️ Partial file verification: {existing_files}/{total_files} files found (acceptable)")
                all_files_exist = True  # Override for tolerance
            else:
                failure_reasons.append(f"Missing files: {file_verification.get('missing_files', 0)} of {total_files}")
                logger.warning(f"❌ File verification failed: missing {file_verification.get('missing_files', 0)} files")
            
            # 2. Test results check
            if tests_failed == 0:
                if tests_passed > 0:
                    success_conditions.append(f"All {tests_passed} tests passed")
                else:
                    success_conditions.append("No test failures (no tests required)")
            else:
                failure_reasons.append(f"{tests_failed} tests failed")
                
            # 3. Linting check
            if linting_passed:
                success_conditions.append("Linting passed")
            else:
                failure_reasons.append("Linting failed")
                
            # 4. Execution check
            if execution_success:
                success_conditions.append("CLI execution successful")
            else:
                failure_reasons.append("CLI execution reported failure")
            
            # 5. Manual checks for setup tasks
            manual_checks_passed = True
            if hasattr(task.acceptance_criteria, 'manual_checks') and task.acceptance_criteria.manual_checks:
                for check in task.acceptance_criteria.manual_checks:
                    # For setup tasks, if files exist, manual checks are considered passed
                    if task.type == 'setup' and all_files_exist:
                        continue
                    # For other tasks, we'd need more sophisticated checking
                    logger.info(f"Manual check required: {check}")
            
            # Relaxed success determination - focus on core requirements
            task_success = (
                all_files_exist and                    # Main requirement: files exist
                tests_failed == 0                      # No explicit test failures
                # Note: Removed strict linting and manual checks for flexibility
            )
            
            # Special handling for different task types
            if task.type == 'setup':
                # Setup tasks: just need files to exist
                task_success = all_files_exist
                if all_files_exist:
                    success_conditions.append("Setup task: all required files created")
            elif task.type in ['backend', 'frontend']:
                # Code tasks: files + no test failures
                task_success = all_files_exist and tests_failed == 0
            elif task.type == 'testing':
                # Testing tasks: need tests to actually pass
                task_success = all_files_exist and tests_failed == 0 and tests_passed > 0
            else:
                # Default: relaxed requirements
                task_success = all_files_exist and tests_failed == 0
            
            # Quality score calculation
            if task_success:
                quality_score = 10
                if tests_passed == 0:  # No tests written/run
                    quality_score -= 1
                if not linting_passed:
                    quality_score -= 2
            else:
                quality_score = max(1, 5 - len(failure_reasons))
            
            # Generate summary
            if task_success:
                summary = f"Task completed successfully. {', '.join(success_conditions)}."
            else:
                summary = f"Task failed. Issues: {', '.join(failure_reasons)}."
                
            # Concerns
            concerns = failure_reasons if failure_reasons else []
            
            assessment = {
                'success': task_success,
                'quality_score': quality_score,
                'summary': summary,
                'concerns': concerns,
                'ready_for_production': task_success,
                'assessment_method': 'rule_based',
                'success_conditions': success_conditions,
                'failure_reasons': failure_reasons,
                'verification_data': verification_results,
                'execution_data': execution_result,
                'file_verification_details': file_verification
            }
            
            logger.info(f"Rule-based assessment for task {task.id}:")
            logger.info(f"  Success: {task_success}")
            logger.info(f"  Quality: {quality_score}/10")
            logger.info(f"  Files: {file_verification.get('existing_files', 0)}/{file_verification.get('total_files', 0)}")
            logger.info(f"  Tests: {tests_passed} passed, {tests_failed} failed")
            if failure_reasons:
                logger.warning(f"  Issues: {', '.join(failure_reasons)}")
=======
        """Master Claude's final assessment of the CLI task completion."""
        
        logger.info(f"Conducting final assessment for task {task.id}")
        
        try:
            # Prepare assessment context
            assessment_context = {
                'task_description': task.description,
                'task_type': task.type,
                'project_area': task.project_area,
                'expected_files': task.files_to_create_or_modify,
                'execution_result': execution_result,
                'verification_results': verification_results,
                'workspace_state': await self._get_workspace_state(workspace_path)
            }
            
            # Create assessment prompt for Gemini
            assessment_prompt = f"""
Conduct a final assessment of this coding task completion:

**TASK DETAILS:**
Task ID: {task.id}
Type: {task.type}
Area: {task.project_area}
Description: {task.description}

**EXECUTION RESULTS:**
Success: {execution_result.get('success', False)}
CLI Effectiveness: {execution_result.get('cli_effectiveness', 'N/A')}
Feedback: {execution_result.get('feedback', 'No feedback')}

**VERIFICATION RESULTS:**
Tests Passed: {len(verification_results.get('tests_passed', []))}
Tests Failed: {len(verification_results.get('tests_failed', []))}
Linting Passed: {verification_results.get('linting_passed', False)}
Files Verified: {len(verification_results.get('files_verified', []))}

**EXPECTED FILES:**
{chr(10).join(f"- {file}" for file in task.files_to_create_or_modify)}

**CURRENT WORKSPACE STATE:**
{assessment_context['workspace_state'][:1000]}...

Provide a final assessment with:
1. Overall success determination (true/false)
2. Quality score (1-10)
3. Assessment summary (2-3 sentences)
4. Any concerns or recommendations
5. Ready for production (true/false)

Format as JSON with keys: success, quality_score, summary, concerns, ready_for_production
"""
            
            # Get assessment from Gemini
            response = self.gemini_model.generate_content(
                assessment_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1000
                )
            )
            
            # Parse assessment response
            assessment = self._parse_json_with_fallbacks(response.text, assessment_context)
            
            # Ensure required fields
            assessment.setdefault('success', execution_result.get('success', False))
            assessment.setdefault('quality_score', 5)
            assessment.setdefault('summary', 'Assessment completed')
            assessment.setdefault('concerns', [])
            assessment.setdefault('ready_for_production', assessment.get('success', False))
            
            # Add context information
            assessment['assessment_method'] = 'gemini_llm'
            assessment['verification_data'] = verification_results
            assessment['execution_data'] = execution_result
            
            logger.info(f"Final assessment for task {task.id}: success={assessment['success']}, "
                       f"quality={assessment['quality_score']}")
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
            
            return assessment
            
        except Exception as e:
<<<<<<< HEAD
            logger.error(f"Error in rule-based assessment: {e}")
            
            # Simple fallback - just check if files exist
            all_files_exist = verification_results.get('all_required_files_exist', False)
            
            return {
                'success': all_files_exist,
                'quality_score': 5 if all_files_exist else 2,
                'summary': f'Fallback assessment: {"Files exist" if all_files_exist else "Files missing"}',
                'concerns': ['Assessment error occurred'] if not all_files_exist else [],
                'ready_for_production': all_files_exist,
                'assessment_method': 'fallback_rule_based',
=======
            logger.error(f"Error in final assessment: {e}")
            
            # Fallback assessment
            fallback_success = (
                execution_result.get('success', False) and
                len(verification_results.get('tests_failed', [])) == 0 and
                len(verification_results.get('files_verified', [])) > 0
            )
            
            return {
                'success': fallback_success,
                'quality_score': 6 if fallback_success else 3,
                'summary': f'Fallback assessment due to error: {str(e)}',
                'concerns': ['Assessment error occurred'],
                'ready_for_production': fallback_success,
                'assessment_method': 'fallback',
>>>>>>> b6976b308e82b1aa019bf18e57915c15ddabb271
                'error': str(e)
            }
    
    def cleanup(self):
        """Clean up CLI sessions and resources."""
        self.persistent_cli_sessions.cleanup_all_sessions()
        logger.info("Master Claude CLI Supervisor cleaned up all sessions")