"""Master Claude Supervisor using CLI - Orchestrates Code Claude CLI execution with intelligent oversight."""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

import google.generativeai as genai
from rich.console import Console

from shared.utils.config import Config
from shared.core.schema import Task, TasksPlan
from shared.tools.aci_interface import ACIInterface, CommandResult
from cli.agents.claude_cli_executor import PersistentClaudeCliSession
from shared.agents.context_manager import ContextManager, ContextCompressionStrategy
from shared.utils.api_manager import api_manager, APIProvider
from shared.utils.response_validator import validate_ai_response
from shared.utils.recovery_manager import get_recovery_manager, TaskStatus
from shared.core.consistency_manager import ConsistencyManager

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
        
        # Context management
        self.context_manager = ContextManager(config, max_context_tokens)
        self.max_context_tokens = max_context_tokens
        
        # Initialize consistency manager
        self.consistency_manager = ConsistencyManager(config)
        
        # Enhanced Code Claude CLI management
        self.persistent_cli_sessions = PersistentClaudeCliSession(config)
        
        # State tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.current_workspace_state = {}
        self.task_progress = {}
        
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
                                               tasks_plan: TasksPlan) -> Dict[str, Any]:
        """Execute a task with Master Claude supervision and Code Claude CLI execution.
        
        Args:
            task: The task to execute
            workspace_path: Path to the workspace
            tasks_plan: Complete tasks plan for context
            
        Returns:
            Execution result with detailed feedback
        """
        console.print(f"[blue]🧠 Master Claude starting CLI supervision of task: {task.id}[/blue]")
        
        # Log context stats before starting
        stats = self.context_manager.get_context_stats()
        console.print(f"[dim]Context: {stats['total_tokens']}/{stats['available_tokens']} tokens ({stats['utilization']:.1%} used)[/dim]")
        
        # Phase 1: Analyze current state and plan execution with managed context
        execution_plan = await self._analyze_and_plan_with_context(task, workspace_path, tasks_plan)
        
        # Add execution plan to context
        self.context_manager.add_context(
            content=json.dumps(execution_plan, indent=2),
            content_type="execution_plan",
            task_ids=[task.id],
            importance=0.8
        )
        
        # Phase 2: Execute with iterative CLI supervision
        result = await self._execute_with_cli_iterations(task, workspace_path, execution_plan)
        
        # Add logging information to result
        result['master_analysis_prompt'] = execution_plan.get('analysis_prompt_used', '')
        result['cli_prompts_used'] = getattr(self, '_cli_prompts_used', [])
        
        # Add execution result to context
        self.context_manager.add_context(
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
        self.context_manager.add_context(
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
        
        return final_result
    
    async def _analyze_and_plan_with_context(self, task: Task, workspace_path: str, 
                                            tasks_plan: TasksPlan) -> Dict[str, Any]:
        """Master Claude analyzes the situation with managed context and creates execution plan for CLI."""
        
        # Build optimized context using context manager
        managed_context = await self.context_manager.build_context_for_task(task, tasks_plan)
        
        # Gather current workspace state
        workspace_state = await self._get_workspace_state(workspace_path)
        
        # Determine target workspace
        target_workspace = "frontend" if "fe-" in task.id or "frontend" in task.project_area.lower() else "backend"
        
        # Generate consistency guidelines for this task
        consistency_prompt = self.consistency_manager.generate_consistency_prompt(task, workspace_path)
        
        # Create enhanced analysis prompt with consistency guidance
        analysis_prompt = f"""Analyze task for Claude CLI execution with consistency enforcement.

**CONTEXT:** {managed_context}

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
            execution_plan['analysis_prompt_used'] = analysis_prompt
            
            response = self.gemini_model.generate_content(
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
            
            execution_log.append({
                'iteration': iteration,
                'evaluation': evaluation,
                'timestamp': datetime.now().isoformat()
            })
            
            if evaluation['success']:
                success = True
                console.print(f"[green]✅ Task completed successfully via CLI in iteration {iteration}[/green]")
            elif iteration < max_iterations:
                console.print(f"[yellow]⚠️ CLI iteration {iteration} needs improvement: {evaluation['feedback']}[/yellow]")
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
        workspace_area = "frontend" if "fe-" in task.id or "frontend" in task.project_area.lower() else "backend"
        
        cli_prompt = f"""**TASK:** {task.description}
**TARGET:** {workspace_area}/ workspace
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
2. Target workspace: ./{workspace_area}/
3. Check existing structure BEFORE creating files
4. Install dependencies: `npm install <package> -w {workspace_area}`

**STRUCTURE ERRORS TO AVOID:**
- NO {workspace_area}/{workspace_area}/ nesting
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
        try:
            # Create curated context for Code Claude CLI
            master_context = await self._create_curated_context_for_code_claude_cli(task)
            task_specific_context = prompt  # The supervised prompt becomes task-specific context
            
            # Execute with persistent CLI session continuity
            result = await self.persistent_cli_sessions.execute_task_with_session_continuity(
                task, workspace_path, master_context, task_specific_context
            )
            
            # Learn from Code Claude CLI's insights
            await self._absorb_code_claude_cli_insights(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced Code Claude CLI execution: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': f"Enhanced Code Claude CLI execution failed: {e}",
                'executor_type': 'cli'
            }
    
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
        target_workspace = "frontend" if "fe-" in task.id or "frontend" in task.project_area.lower() else "backend"
        
        evaluation_prompt = f"""Evaluate CLI execution. Check if files are in correct {target_workspace}/ workspace.

**TASK:** {task.description}
**TARGET:** {target_workspace}/ workspace
**RESULT:** {iteration_result.get('success', False)} - {iteration_result.get('summary', 'No summary')}
**FILES:** {file_check}
**STRUCTURE:** {updated_state}

**CHECK:** Are files in {target_workspace}/ workspace? No duplicate directories?

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
    
    async def _execute_task_with_recovery(self, task: Task, workspace_path: str) -> Dict[str, Any]:
        """Execute task with recovery support."""
        if not self.recovery_manager:
            project_id = Path(workspace_path).name
            self.recovery_manager = get_recovery_manager(project_id)
        
        # Create checkpoint before execution
        checkpoint_id = self.recovery_manager.create_checkpoint(
            task.id, 
            TaskStatus.IN_PROGRESS,
            {"workspace_path": workspace_path, "task_description": task.description}
        )
        
        try:
            # Execute with API rate limiting
            result = await api_manager.execute_with_retry(
                self._execute_task_internal,
                APIProvider.CLAUDE,
                task,
                workspace_path
            )
            
            # Mark as completed on success
            if result.get('success', False):
                self.recovery_manager.update_checkpoint(checkpoint_id, TaskStatus.COMPLETED)
                self.recovery_manager.mark_task_completed(task.id)
            else:
                self.recovery_manager.update_checkpoint(
                    checkpoint_id, 
                    TaskStatus.FAILED,
                    error_info={"result": result}
                )
            
            return result
            
        except Exception as e:
            # Record failure with error details
            self.recovery_manager.update_checkpoint(
                checkpoint_id,
                TaskStatus.FAILED,
                error_info={"exception": str(e), "type": type(e).__name__}
            )
            
            # Attempt recovery if possible
            if self.recovery_manager.can_recover_from_failure(task.id):
                logger.info(f"Attempting recovery for task {task.id}")
                recovery_checkpoint = await self.recovery_manager.recover_from_failure(task.id)
                
                if recovery_checkpoint:
                    # Retry with recovery context
                    return await self._execute_task_internal(task, workspace_path)
            
            raise
    
    async def _execute_task_internal(self, task: Task, workspace_path: str) -> Dict[str, Any]:
        """Internal task execution method."""
        # This should contain the actual task execution logic
        # For now, we'll call the existing method
        return await self._execute_single_cli_task(task, workspace_path)
    
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
    
    async def _verify_expected_files(self, task: Task, workspace_path: str) -> str:
        """Verify that expected files were created/modified."""
        results = []
        for file_path in task.files_to_create_or_modify:
            full_path = f"{workspace_path}/{file_path}"
            exists = await self.aci.file_exists(full_path)
            results.append(f"- {file_path}: {'✓ exists' if exists else '✗ missing'}")
        
        return "Expected Files:\n" + "\n".join(results)
    
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
        """Run actual acceptance criteria tests."""
        results = {
            'tests_passed': [],
            'tests_failed': [],
            'linting_passed': False,
            'files_verified': []
        }
        
        try:
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
            
            # Verify files
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
        """Master Claude's final assessment of the CLI task completion."""
        
        assessment_prompt = f"""You are the Master Supervisor making a final assessment of a completed coding task executed via Claude CLI.

**TASK COMPLETED:** {task.description}

**CLI EXECUTION SUMMARY:**
{execution_result}

**VERIFICATION RESULTS:**
Tests passed: {verification_results.get('tests_passed', [])}
Tests failed: {verification_results.get('tests_failed', [])}
Linting passed: {verification_results.get('linting_passed', False)}
Files verified: {verification_results.get('files_verified', [])}

**CLI EXECUTION NOTES:**
Executor Type: {execution_result.get('executor_type', 'unknown')}
Iterations: {execution_result.get('iterations', 1)}

**FINAL ASSESSMENT REQUIRED:**
Provide a comprehensive assessment in JSON format:

```json
{{
    "success": true/false,
    "quality_score": 1-10,
    "completion_percentage": 0-100,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "recommendations": ["recommendation1", "recommendation2"],
    "cli_effectiveness": 1-10,
    "cli_advantages_realized": ["advantage1", "advantage2"],
    "ready_for_next_task": true/false,
    "summary": "Brief summary of the CLI task completion"
}}
```"""

        try:
            response = self.gemini_model.generate_content(
                assessment_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
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
                logger.warning("Gemini response blocked by safety filters")
                raise Exception("Response blocked by safety filters")
            
            # Parse response with better error handling
            assessment_text = response.text
            if '```json' in assessment_text:
                json_start = assessment_text.find('```json') + 7
                json_end = assessment_text.find('```', json_start)
                if json_end == -1:  # No closing ```
                    assessment_text = assessment_text[json_start:]
                else:
                    assessment_text = assessment_text[json_start:json_end]
            
            # Clean up common JSON issues
            assessment_text = assessment_text.strip()
            assessment_text = assessment_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            
            try:
                return json.loads(assessment_text)
            except json.JSONDecodeError as json_error:
                logger.warning(f"JSON parsing failed in final assessment: {json_error}")
                logger.warning(f"Raw response text: {assessment_text[:500]}")
                # Use fallback assessment
                raise Exception(f"JSON parsing failed: {json_error}")
            
        except Exception as e:
            logger.error(f"Error in final CLI assessment: {e}")
            return {
                'success': len(verification_results.get('tests_failed', [])) == 0,
                'summary': 'CLI assessment completed with basic verification',
                'quality_score': 7,
                'cli_effectiveness': 7,
                'executor_type': 'cli'
            }
    
    def cleanup(self):
        """Clean up CLI sessions and resources."""
        self.persistent_cli_sessions.cleanup_all_sessions()
        logger.info("Master Claude CLI Supervisor cleaned up all sessions")