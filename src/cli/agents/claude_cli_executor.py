"""Claude CLI Executor - Uses Claude Code CLI instead of SDK."""

import asyncio
import logging
import subprocess
import json
import tempfile
import os
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
from pathlib import Path

from shared.utils.config import Config
from shared.core.schema import Task, TasksPlan
from shared.tools.aci_interface import ACIInterface

logger = logging.getLogger(__name__)


class ClaudeCliExecutor:
    """Enhanced Claude CLI Executor with worker pool support."""
    
    def __init__(self, config: Config, worker_id: str = "default"):
        self.config = config
        self.worker_id = worker_id
        self.aci = ACIInterface()
        self.completion_detector = ClaudeCompletionDetector()
        self.execution_stats = {
            'tasks_executed': 0,
            'total_execution_time': 0.0,
            'successful_tasks': 0,
            'failed_tasks': 0
        }
        logger.info(f"Claude CLI Executor {worker_id} initialized")
    
    async def execute_task_with_curated_context(
        self,
        task: Task,
        workspace_path: str,
        curated_context: str,
        timeout: int = 600
    ) -> Dict[str, Any]:
        """Execute a task using Claude CLI with curated context."""
        
        start_time = time.time()
        logger.info(f"Worker {self.worker_id} executing task {task.id}")
        
        try:
            # Prepare CLI prompt
            cli_prompt = self._create_cli_prompt(task, workspace_path, curated_context)
            
            # Execute with Claude CLI
            result = await self._execute_claude_cli(cli_prompt, workspace_path, timeout)
            
            # Update stats
            execution_time = time.time() - start_time
            self.execution_stats['tasks_executed'] += 1
            self.execution_stats['total_execution_time'] += execution_time
            
            if result.get('success', False):
                self.execution_stats['successful_tasks'] += 1
            else:
                self.execution_stats['failed_tasks'] += 1
            
            result['worker_id'] = self.worker_id
            result['execution_time'] = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_stats['failed_tasks'] += 1
            self.execution_stats['total_execution_time'] += execution_time
            
            logger.error(f"Worker {self.worker_id} failed to execute task {task.id}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'worker_id': self.worker_id,
                'execution_time': execution_time,
                'task_id': task.id
            }
    
    def _create_cli_prompt(self, task: Task, workspace_path: str, curated_context: str) -> str:
        """Create optimized CLI prompt for task execution."""
        
        prompt = f"""You are Code Claude, an expert software developer working via CLI.

**WORKSPACE:** {workspace_path}
**TASK ID:** {task.id}
**TASK TYPE:** {task.type}
**PROJECT AREA:** {task.project_area}

**MASTER'S CURATED CONTEXT:**
{curated_context}

**TASK DESCRIPTION:**
{task.description}

**FILES TO CREATE/MODIFY:**
{', '.join(task.files_to_create_or_modify)}

**ACCEPTANCE CRITERIA:**
{json.dumps(task.acceptance_criteria.dict() if hasattr(task.acceptance_criteria, 'dict') else task.acceptance_criteria, indent=2)}

**DEPENDENCIES (COMPLETED):**
{', '.join(task.dependencies) if task.dependencies else 'None'}

**TECHNICAL DETAILS:**
{json.dumps(task.technical_details.dict() if hasattr(task.technical_details, 'dict') else task.technical_details or {}, indent=2)}

**INSTRUCTIONS:**
1. Follow the exact task description and acceptance criteria
2. Use the curated context to maintain consistency with the project
3. Create/modify only the specified files
4. Follow established patterns and conventions from the context
5. Write clean, well-documented, production-ready code
6. Run tests if specified in acceptance criteria
7. Signal completion clearly when done

**COMPLETION SIGNAL:**
When the task is fully complete, output: "TASK IMPLEMENTATION COMPLETE"

Begin implementation now:
"""
        return prompt
    
    async def _execute_claude_cli(
        self, 
        prompt: str, 
        workspace_path: str, 
        timeout: int
    ) -> Dict[str, Any]:
        """Execute Claude CLI with the given prompt."""
        
        # Create temporary prompt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name
        
        try:
            # Prepare Claude CLI command
            cmd = [
                "claude",
                "--headless",
                "--project", workspace_path,
                "--file", prompt_file
            ]
            
            # Execute Claude CLI process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace_path
            )
            
            # Monitor output for completion
            stdout_lines = []
            stderr_lines = []
            completion_detected = False
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                if stdout:
                    stdout_text = stdout.decode('utf-8', errors='ignore')
                    stdout_lines = stdout_text.split('\n')
                    
                    # Check for completion signals
                    for line in stdout_lines:
                        detection_result = self.completion_detector.analyze_output(line)
                        if detection_result.get('completion_found'):
                            completion_detected = True
                            break
                
                if stderr:
                    stderr_text = stderr.decode('utf-8', errors='ignore')
                    stderr_lines = stderr_text.split('\n')
                
                # Process completed successfully
                success = process.returncode == 0 or completion_detected
                
                return {
                    'success': success,
                    'returncode': process.returncode,
                    'stdout': stdout_lines,
                    'stderr': stderr_lines,
                    'completion_detected': completion_detected,
                    'completion_signals': self.completion_detector.completion_signals
                }
                
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                await process.wait()
                
                return {
                    'success': False,
                    'error': f'Claude CLI execution timed out after {timeout} seconds',
                    'timeout': True
                }
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(prompt_file)
            except Exception:
                pass
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker execution statistics."""
        return {
            'worker_id': self.worker_id,
            'stats': self.execution_stats.copy()
        }


class ClaudeCompletionDetector:
    """Detect Claude CLI completion signals from output messages."""
    
    def __init__(self):
        self.completion_signals = []
        # Explicit completion messages from Master prompt (highest priority)
        self.explicit_completion_patterns = [
            r"TASK IMPLEMENTATION COMPLETE",
            r"TASK FINISHED SUCCESSFULLY", 
            r"WORK COMPLETED"
        ]
        # General completion indicators (lower priority)
        self.task_completion_patterns = [
            r"task.*complete",
            r"implementation.*complete", 
            r"successfully.*implemented",
            r"finished.*implementing",
            r"done.*creating",
            r"created.*successfully",
            r"task.*finished",
            r"work.*complete",
            r"all.*files.*created",
            r"implementation.*ready",
            # Git commit messages
            r"committed.*changes",
            r"git.*commit",
            # Test completion
            r"tests.*pass",
            r"all.*tests.*successful",
            # Build/compilation success
            r"build.*successful",
            r"compilation.*successful",
            # Final status messages
            r"ready.*for.*review",
            r"task.*accomplished",
        ]
        
    def analyze_output(self, output_line: str) -> dict:
        """Analyze Claude CLI output for completion signals."""
        import re
        
        line_upper = output_line.upper()
        line_lower = output_line.lower()
        
        # Check for explicit completion messages first (highest priority)
        explicit_completion = False
        for pattern in self.explicit_completion_patterns:
            if re.search(pattern, line_upper):
                explicit_completion = True
                self.completion_signals.append({
                    'pattern': pattern,
                    'line': output_line.strip(),
                    'timestamp': time.time(),
                    'type': 'explicit',
                    'priority': 'high'
                })
                break
        
        # Check for general completion patterns (lower priority)
        general_completion = False
        if not explicit_completion:
            for pattern in self.task_completion_patterns:
                if re.search(pattern, line_lower):
                    general_completion = True
                    self.completion_signals.append({
                        'pattern': pattern,
                        'line': output_line.strip(),
                        'timestamp': time.time(),
                        'type': 'general',
                        'priority': 'medium'
                    })
                    break
        
        completion_found = explicit_completion or general_completion
        
        # Check for error indicators
        error_indicators = ['error', 'failed', 'exception', 'traceback']
        has_error = any(indicator in line_lower for indicator in error_indicators)
        
        # Explicit completion = immediate finish, general = need 2 signals
        has_explicit = any(s.get('type') == 'explicit' for s in self.completion_signals)
        
        return {
            'completion_detected': completion_found,
            'explicit_completion': explicit_completion,
            'error_detected': has_error,
            'total_completion_signals': len(self.completion_signals),
            'recent_signals': self.completion_signals[-3:] if self.completion_signals else [],
            'appears_finished': has_explicit or len(self.completion_signals) >= 2,
            'immediate_finish': has_explicit  # New: for immediate termination
        }


class FileSystemMonitor:
    """Monitor file system changes to track Claude CLI progress."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.initial_snapshot = self._get_filesystem_snapshot()
        self.last_change_time = time.time()
        self.completion_indicators = 0  # Track completion signals
        
    def _get_filesystem_snapshot(self) -> Dict[str, Any]:
        """Get current state of filesystem."""
        snapshot = {
            'files': [],
            'directories': [],
            'total_size': 0,
            'file_count': 0,
            'dir_count': 0
        }
        
        try:
            if not self.workspace_path.exists():
                return snapshot
                
            for item in self.workspace_path.rglob('*'):
                rel_path = str(item.relative_to(self.workspace_path))
                
                if item.is_file():
                    try:
                        size = item.stat().st_size
                        mtime = item.stat().st_mtime
                        snapshot['files'].append({
                            'path': rel_path,
                            'size': size,
                            'mtime': mtime
                        })
                        snapshot['total_size'] += size
                        snapshot['file_count'] += 1
                    except (OSError, PermissionError):
                        continue
                elif item.is_dir():
                    snapshot['directories'].append(rel_path)
                    snapshot['dir_count'] += 1
                    
        except Exception as e:
            logger.debug(f"Error scanning filesystem: {e}")
            
        return snapshot
    
    def check_for_changes(self) -> Dict[str, Any]:
        """Check for filesystem changes since last check."""
        current_snapshot = self._get_filesystem_snapshot()
        
        changes = {
            'new_files': [],
            'modified_files': [],
            'new_directories': [],
            'size_change': current_snapshot['total_size'] - self.initial_snapshot['total_size'],
            'file_count_change': current_snapshot['file_count'] - self.initial_snapshot['file_count'],
            'dir_count_change': current_snapshot['dir_count'] - self.initial_snapshot['dir_count'],
            'has_changes': False
        }
        
        # Check for new/modified files
        current_files = {f['path']: f for f in current_snapshot['files']}
        initial_files = {f['path']: f for f in self.initial_snapshot['files']}
        
        for path, file_info in current_files.items():
            if path not in initial_files:
                changes['new_files'].append(path)
                changes['has_changes'] = True
            elif file_info['mtime'] > initial_files[path]['mtime']:
                changes['modified_files'].append(path)
                changes['has_changes'] = True
        
        # Check for new directories
        current_dirs = set(current_snapshot['directories'])
        initial_dirs = set(self.initial_snapshot['directories'])
        changes['new_directories'] = list(current_dirs - initial_dirs)
        
        if changes['new_directories']:
            changes['has_changes'] = True
            
        if changes['has_changes']:
            self.last_change_time = time.time()
            
        return changes
    
    def get_activity_summary(self) -> Dict[str, Any]:
        """Get summary of file system activity."""
        current_snapshot = self._get_filesystem_snapshot()
        time_since_last_change = time.time() - self.last_change_time
        
        # Conservative completion detection - only for clearly finished tasks
        files_created = current_snapshot['file_count'] - self.initial_snapshot['file_count']
        dirs_created = current_snapshot['dir_count'] - self.initial_snapshot['dir_count']
        
        # More strict criteria for completion
        appears_complete = (
            files_created >= 2 and  # At least 2 files created (more substantial work)
            time_since_last_change > 45 and  # No changes for 45 seconds (longer wait)
            current_snapshot['total_size'] > self.initial_snapshot['total_size'] + 1024  # At least 1KB of content
        )
        
        if appears_complete:
            self.completion_indicators += 1
        
        return {
            'total_files': current_snapshot['file_count'],
            'total_directories': current_snapshot['dir_count'],
            'total_size_mb': round(current_snapshot['total_size'] / (1024 * 1024), 2),
            'files_created': files_created,
            'dirs_created': current_snapshot['dir_count'] - self.initial_snapshot['dir_count'],
            'size_change_mb': round((current_snapshot['total_size'] - self.initial_snapshot['total_size']) / (1024 * 1024), 2),
            'seconds_since_last_change': round(time_since_last_change, 1),
            'appears_active': time_since_last_change < 25,  # Conservative: 25 seconds for safety
            'appears_complete': appears_complete,
            'completion_signals': self.completion_indicators
        }


class ClaudeCliExecutor:
    """Enhanced Claude executor that uses Claude Code CLI."""
    
    def __init__(self, config: Config):
        self.config = config
        self.aci = ACIInterface()
        
        # Session state for this executor instance
        self.session_context = {
            'project_patterns': [],
            'coding_style': {},
            'established_conventions': {},
            'recent_files_state': {},
            'common_utilities': []
        }
        
        # CLI configuration
        self.claude_cli_path = self._find_claude_cli()
        self.session_dir = None
        self._setup_session_directory()
    
    def _find_claude_cli(self) -> str:
        """Find Claude CLI executable."""
        # Use configured path if available
        if self.config.claude_cli_path:
            logger.info(f"Using configured Claude CLI path: {self.config.claude_cli_path}")
            return self.config.claude_cli_path
        
        # First try to find claude using 'which' command
        try:
            which_result = subprocess.run(['which', 'claude'], 
                                        capture_output=True, text=True, timeout=5)
            if which_result.returncode == 0 and which_result.stdout.strip():
                claude_path = which_result.stdout.strip()
                logger.info(f"Found Claude CLI via 'which' command: {claude_path}")
                return claude_path
        except Exception as e:
            logger.debug(f"'which claude' failed: {e}")
        
        # Try common locations (prioritize /usr/bin/claude like TS example)
        possible_paths = [
            '/usr/bin/claude',  # Primary path from TS example
            '/usr/local/bin/claude',
            os.path.expanduser('~/.local/bin/claude'),
            'claude',  # Try directly in PATH
            'npx claude'
        ]
        
        for path in possible_paths:
            try:
                # Test if the executable exists and is accessible
                if path.startswith('/'):
                    # For absolute paths, check if file exists and is executable
                    if os.path.isfile(path) and os.access(path, os.X_OK):
                        logger.info(f"Found Claude CLI at: {path}")
                        return path
                else:
                    # For relative paths, try running with --help
                    result = subprocess.run([path, '--help'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        logger.info(f"Found Claude CLI at: {path}")
                        return path
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
                logger.debug(f"Failed to test {path}: {e}")
                continue
        
        # If still not found, try just 'claude' in PATH
        logger.warning("Claude CLI not found in standard locations, trying 'claude' directly")
        return 'claude'
    
    def _setup_session_directory(self):
        """Create a temporary directory for this session."""
        self.session_dir = tempfile.mkdtemp(prefix='vibe_claude_session_')
        logger.info(f"Created Claude CLI session directory: {self.session_dir}")
    
    async def execute_task_with_curated_context(self, 
                                              task: Task, 
                                              workspace_path: str,
                                              master_context: str,
                                              task_specific_context: str) -> Dict[str, Any]:
        """Execute task with carefully curated context using Claude CLI.
        
        Args:
            task: Task to execute
            workspace_path: Working directory
            master_context: Curated context from Master Claude
            task_specific_context: Specific context for this task
            
        Returns:
            Execution result with detailed feedback
        """
        logger.info(f"Executing task {task.id} with Claude CLI")
        
        # Build comprehensive but focused prompt
        execution_prompt = self._build_contextual_prompt(
            task, workspace_path, master_context, task_specific_context
        )
        
        execution_log = []
        success = False
        error_message = None
        
        try:
            # Execute with Claude CLI
            async for message in self._execute_with_claude_cli(execution_prompt, task, workspace_path):
                execution_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'content': message,
                    'type': 'claude_response'
                })
                
                # Process responses and update session context
                await self._update_session_context(message, task)
            
            # Verify completion
            success = await self._verify_task_completion(task, workspace_path)
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Task {task.id} failed: {e}")
            execution_log.append({
                'timestamp': datetime.now().isoformat(),
                'content': f"Execution failed: {e}",
                'type': 'error'
            })
        
        return {
            'task_id': task.id,
            'success': success,
            'error': error_message,
            'execution_log': execution_log,
            'session_updates': self._get_session_updates(),
            'learned_patterns': self._extract_learned_patterns(),
            'completed_at': datetime.now().isoformat()
        }
    
    def _build_contextual_prompt(self, task: Task, workspace_path: str,
                               master_context: str, task_specific_context: str) -> str:
        """Build a comprehensive prompt with all necessary context."""
        
        prompt = f"""You are an expert software developer working on task: {task.id}

**MASTER CLAUDE'S CURATED CONTEXT:**
{master_context}

**TASK-SPECIFIC CONTEXT:**
{task_specific_context}

**SESSION CONTEXT (Patterns from previous tasks in this session):**
"""
        
        # Add session-specific context
        if self.session_context['project_patterns']:
            prompt += "\nEstablished Patterns:\n"
            for pattern in self.session_context['project_patterns']:
                prompt += f"- {pattern}\n"
        
        if self.session_context['coding_style']:
            prompt += "\nCoding Style Conventions:\n"
            for key, value in self.session_context['coding_style'].items():
                prompt += f"- {key}: {value}\n"
        
        if self.session_context['common_utilities']:
            prompt += "\nCommon Utilities Available:\n"
            for utility in self.session_context['common_utilities']:
                prompt += f"- {utility}\n"
        
        # Add current task details
        prompt += f"""

**CURRENT TASK DETAILS:**
- Description: {task.description}
- Type: {task.type}
- Project Area: {task.project_area}
- Files to modify: {', '.join(task.files_to_create_or_modify)}

**ACCEPTANCE CRITERIA:**
"""
        
        # Add acceptance criteria
        criteria = task.acceptance_criteria
        if criteria.tests:
            prompt += "Tests that must pass:\n"
            for test in criteria.tests:
                prompt += f"- {test.type} test in {test.file}"
                if test.function:
                    prompt += f" (function: {test.function})"
                prompt += "\n"
        
        if criteria.linting:
            prompt += f"Linting requirements: {criteria.linting}\n"
        
        if criteria.manual_checks:
            prompt += "Manual verification:\n"
            for check in criteria.manual_checks:
                prompt += f"- {check}\n"
        
        prompt += """

**INSTRUCTIONS:**
1. Follow the established patterns and conventions shown above
2. Build upon the work already completed (as described in the context)
3. Ensure consistency with the existing codebase style
4. Implement the task requirements precisely
5. Run tests and verify the acceptance criteria
6. Provide clear status updates as you work

IMPORTANT: You have context from previous tasks, so build upon existing work rather than recreating everything from scratch.

Begin implementation:"""
        
        return prompt
    
    async def _execute_with_claude_cli(self, prompt: str, task: Task, workspace_path: str) -> AsyncGenerator[str, None]:
        """Execute with Claude CLI."""
        try:
            # Ensure workspace directory exists
            os.makedirs(workspace_path, exist_ok=True)
            logger.info(f"Ensured workspace directory exists: {workspace_path}")
            
            # Prepare Claude CLI command with prompt as argument (non-interactive)
            cmd = [self.claude_cli_path]
            
            # Add skip permissions flag if configured (always add it for automation)
            if self.config.claude_cli_skip_permissions:
                cmd.append('--dangerously-skip-permissions')
                logger.info("Added --dangerously-skip-permissions flag for automated execution")
            
            # Add print flag for non-interactive output
            cmd.append('--print')
            
            # Add the prompt directly as command line argument
            cmd.append(prompt)
            
            logger.info("Using command-line prompt (non-interactive mode)")
            
            # Set up environment (like TS example) - no API key needed for logged-in CLI
            env = os.environ.copy()
            env['TERM'] = 'xterm-256color'  # Like TS example
            
            # Remove any ANTHROPIC_API_KEY to force use of CLI authentication
            if 'ANTHROPIC_API_KEY' in env:
                del env['ANTHROPIC_API_KEY']
                logger.info("Removed ANTHROPIC_API_KEY to use CLI authentication")
            
            logger.info("Using Claude CLI local authentication (already logged in)")
            
            # Validate Claude CLI before execution
            if not os.path.isfile(self.claude_cli_path):
                raise RuntimeError(f"Claude CLI not found at path: {self.claude_cli_path}")
            if not os.access(self.claude_cli_path, os.X_OK):
                raise RuntimeError(f"Claude CLI not executable at path: {self.claude_cli_path}")
            
            logger.info(f"Executing Claude CLI: {' '.join(cmd)}")
            logger.info(f"Claude CLI path verified: {self.claude_cli_path}")
            logger.info(f"Working directory: {workspace_path}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            if self.config.claude_cli_skip_permissions:
                logger.info("⚠️  Permission checks are skipped for automated execution")
            
            # Test Claude CLI with simple command first
            try:
                test_result = subprocess.run([self.claude_cli_path, '--help'], 
                                           capture_output=True, text=True, timeout=10)
                logger.info(f"Claude CLI test command return code: {test_result.returncode}")
                if test_result.stdout:
                    logger.info(f"Claude CLI help output (first 200 chars): {test_result.stdout[:200]}")
                if test_result.stderr:
                    logger.info(f"Claude CLI help stderr: {test_result.stderr[:200]}")
                    
                # Test with a simple prompt to verify Claude CLI works (skip API key test)
                logger.info("Skipping simple test - using CLI authentication")
                    
            except Exception as e:
                logger.warning(f"Claude CLI test command failed: {e}")
            
            # Execute CLI process without stdin (command-line prompt)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            logger.info(f"Claude process PID: {process.pid}")
            logger.info("Executing Claude CLI with command-line prompt (no stdin needed)")
            
            # Initialize file system monitoring and completion detection
            fs_monitor = FileSystemMonitor(workspace_path)
            completion_detector = ClaudeCompletionDetector()
            logger.info("Started file system monitoring and completion detection")
            
            # Stream output with balanced timeout and file system monitoring
            start_time = asyncio.get_event_loop().time()
            timeout_seconds = 180  # 3 minutes timeout (balanced: not too short, not too long)
            last_fs_check = start_time
            fs_check_interval = 8  # Check filesystem every 8 seconds (balanced frequency)
            
            while True:
                try:
                    current_time = asyncio.get_event_loop().time()
                    
                    # Check for overall timeout
                    if current_time - start_time > timeout_seconds:
                        logger.warning(f"Claude CLI execution timeout after {timeout_seconds} seconds")
                        activity = fs_monitor.get_activity_summary()
                        yield f"TIMEOUT: Claude CLI execution exceeded {timeout_seconds} seconds"
                        yield f"Final activity: {activity['files_created']} files, {activity['dirs_created']} dirs created"
                        process.terminate()
                        break
                    
                    # Periodic file system check
                    if current_time - last_fs_check >= fs_check_interval:
                        changes = fs_monitor.check_for_changes()
                        activity = fs_monitor.get_activity_summary()
                        
                        if changes['has_changes']:
                            new_files_msg = f"+{len(changes['new_files'])} files" if changes['new_files'] else ""
                            new_dirs_msg = f"+{len(changes['new_directories'])} dirs" if changes['new_directories'] else ""
                            mod_files_msg = f"~{len(changes['modified_files'])} modified" if changes['modified_files'] else ""
                            
                            change_parts = [part for part in [new_files_msg, new_dirs_msg, mod_files_msg] if part]
                            change_summary = ", ".join(change_parts)
                            
                            yield f"📁 File Activity: {change_summary} (Total: {activity['total_files']} files, {activity['total_size_mb']}MB)"
                            
                            # Show some new files as examples
                            if changes['new_files']:
                                example_files = changes['new_files'][:3]
                                yield f"📄 New files: {', '.join(example_files)}" + (f" (+{len(changes['new_files'])-3} more)" if len(changes['new_files']) > 3 else "")
                        
                        elif activity['appears_active']:
                            yield f"📁 Activity Status: {activity['total_files']} files, {activity['total_size_mb']}MB (Last change: {activity['seconds_since_last_change']}s ago)"
                        else:
                            # Combined completion detection: Claude messages + file system
                            claude_signals = completion_detector.completion_signals
                            has_claude_completion = len(claude_signals) >= 1
                            
                            # More confident completion detection with Claude's input
                            if has_claude_completion and activity['files_created'] >= 1:
                                yield f"🎯 Combined completion: Claude signals ({len(claude_signals)}) + files created ({activity['files_created']})"
                                yield f"⚡ Confident completion detected - terminating process"
                                process.terminate()
                                break
                            elif activity['completion_signals'] >= 3:  # Pure file system signals
                                yield f"🎯 File system completion: {activity['files_created']} files created, {activity['seconds_since_last_change']}s idle"
                                yield f"⚡ File system completion detected - terminating process"
                                process.terminate()
                                break
                            elif activity['seconds_since_last_change'] > 75:  # Longer wait without Claude signals
                                if activity['files_created'] >= 2 and current_time - start_time > 75:
                                    yield f"🎯 Conservative completion: {activity['files_created']} files created, {activity['seconds_since_last_change']}s idle"
                                    yield f"⚡ Timeout-based completion detected - terminating process"
                                    process.terminate()
                                    break
                                elif current_time - start_time > 120:  # Extended patience
                                    yield f"⚠️ No file changes for {activity['seconds_since_last_change']}s - Claude might be thinking or stuck"
                        
                        last_fs_check = current_time
                    
                    # Read line with shorter timeout
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=2.0)
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8').strip()
                    if line_str:
                        # Analyze Claude's output for completion signals
                        completion_analysis = completion_detector.analyze_output(line_str)
                        
                        # Yield the output
                        yield f"Claude CLI: {line_str}"
                        
                        # Check for completion signals from Claude's messages
                        if completion_analysis['completion_detected']:
                            signal_type = completion_analysis['recent_signals'][-1].get('type', 'general')
                            yield f"🎯 Claude completion signal detected ({signal_type}): {completion_analysis['recent_signals'][-1]['pattern']}"
                        
                        # Immediate termination for explicit completion messages
                        if completion_analysis['immediate_finish']:
                            yield f"✅ EXPLICIT COMPLETION MESSAGE DETECTED!"
                            yield f"⚡ Claude sent explicit completion signal - terminating immediately"
                            process.terminate()
                            break
                        
                        # Regular completion logic for general signals
                        elif completion_analysis['appears_finished']:
                            yield f"✅ Claude indicates task completion ({completion_analysis['total_completion_signals']} signals)"
                            yield f"⚡ Claude completion message detected - terminating process"
                            process.terminate()
                            break
                        
                except asyncio.TimeoutError:
                    # No output for 5 seconds, check if process is still alive
                    if process.returncode is not None:
                        logger.info("Process has finished")
                        break
                    else:
                        # Continue waiting, file system monitoring will show activity
                        continue
            
            # Wait for completion and check for errors
            await process.wait()
            
            # Final file system summary
            final_activity = fs_monitor.get_activity_summary()
            yield f"🎯 Task completed with {final_activity['files_created']} files created, {final_activity['dirs_created']} directories created"
            yield f"📊 Total workspace: {final_activity['total_files']} files, {final_activity['total_size_mb']}MB"
            
            if process.returncode != 0:
                stderr_output = await process.stderr.read()
                stdout_output = await process.stdout.read()
                
                error_msg = stderr_output.decode('utf-8', errors='replace') if stderr_output else ""
                stdout_msg = stdout_output.decode('utf-8', errors='replace') if stdout_output else ""
                
                # Log detailed error information
                logger.error(f"Claude CLI failed with return code {process.returncode}")
                logger.error(f"Command: {' '.join(cmd)}")
                logger.error(f"Working directory: {workspace_path}")
                logger.error(f"STDERR: {error_msg}")
                logger.error(f"STDOUT: {stdout_msg}")
                
                # Yield detailed error information
                yield f"ERROR - Return code: {process.returncode}"
                if error_msg:
                    yield f"ERROR - STDERR: {error_msg}"
                if stdout_msg:
                    yield f"ERROR - STDOUT: {stdout_msg}"
                
                # Create comprehensive error message
                full_error = f"Return code {process.returncode}"
                if error_msg.strip():
                    full_error += f" - STDERR: {error_msg.strip()}"
                if stdout_msg.strip():
                    full_error += f" - STDOUT: {stdout_msg.strip()}"
                
                raise RuntimeError(f"Claude CLI failed: {full_error}")
            
            yield f"Task {task.id} completed via Claude CLI"
            
        except Exception as e:
            logger.error(f"Error in Claude CLI execution: {e}")
            yield f"Error: {e}"
            raise
    
    async def _execute_with_claude_cli_alternative(self, prompt: str, task: Task, workspace_path: str) -> AsyncGenerator[str, None]:
        """Alternative execution method using stdin for prompt."""
        try:
            # Prepare Claude CLI command (minimal options)
            cmd = [self.claude_cli_path, '--print']
            
            # Add skip permissions flag if configured
            if self.config.claude_cli_skip_permissions:
                cmd.insert(1, '--dangerously-skip-permissions')  # Insert after claude command
                logger.info("Added --dangerously-skip-permissions flag for automated execution")
            
            # Set up environment - force CLI authentication
            env = os.environ.copy()
            env['TERM'] = 'xterm-256color'  # Like main method
            
            # Remove any ANTHROPIC_API_KEY to force use of CLI authentication
            if 'ANTHROPIC_API_KEY' in env:
                del env['ANTHROPIC_API_KEY']
                logger.info("Removed ANTHROPIC_API_KEY to use CLI authentication")
            
            logger.info(f"Executing Claude CLI with stdin: {' '.join(cmd)} in {workspace_path}")
            if self.config.claude_cli_skip_permissions:
                logger.info("⚠️  Permission checks are skipped for automated execution")
            
            # Execute CLI process with stdin
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace_path,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send prompt to stdin
            process.stdin.write(prompt.encode('utf-8'))
            process.stdin.write('\n'.encode('utf-8'))
            await process.stdin.drain()
            process.stdin.close()
            
            # Stream output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                line = line.decode('utf-8').strip()
                if line:
                    yield f"Claude CLI: {line}"
                    await asyncio.sleep(0.1)
            
            # Wait for completion
            await process.wait()
            
            if process.returncode != 0:
                stderr_output = await process.stderr.read()
                stdout_output = await process.stdout.read()
                
                error_msg = stderr_output.decode('utf-8', errors='replace') if stderr_output else ""
                stdout_msg = stdout_output.decode('utf-8', errors='replace') if stdout_output else ""
                
                # Log detailed error information
                logger.error(f"Claude CLI (alternative) failed with return code {process.returncode}")
                logger.error(f"Command: {' '.join(cmd)}")
                logger.error(f"Working directory: {workspace_path}")
                logger.error(f"STDERR: {error_msg}")
                logger.error(f"STDOUT: {stdout_msg}")
                
                # Yield detailed error information
                yield f"ERROR - Return code: {process.returncode}"
                if error_msg:
                    yield f"ERROR - STDERR: {error_msg}"
                if stdout_msg:
                    yield f"ERROR - STDOUT: {stdout_msg}"
                
                # Create comprehensive error message
                full_error = f"Return code {process.returncode}"
                if error_msg.strip():
                    full_error += f" - STDERR: {error_msg.strip()}"
                if stdout_msg.strip():
                    full_error += f" - STDOUT: {stdout_msg.strip()}"
                
                raise RuntimeError(f"Claude CLI failed: {full_error}")
            
            yield f"Task {task.id} completed successfully"
            
        except Exception as e:
            logger.error(f"Error in alternative Claude CLI execution: {e}")
            yield f"Error: {e}"
            raise
    
    async def _update_session_context(self, message: str, task: Task) -> None:
        """Update session context based on Claude's responses."""
        
        # Extract patterns from Claude's work
        message_lower = message.lower()
        
        # Detect new patterns
        if 'pattern' in message_lower or 'convention' in message_lower:
            pattern = f"Pattern from {task.id}: {message[:100]}..."
            if pattern not in self.session_context['project_patterns']:
                self.session_context['project_patterns'].append(pattern)
        
        # Detect coding style decisions
        if 'style' in message_lower or 'format' in message_lower:
            style_key = f"{task.type}_style"
            self.session_context['coding_style'][style_key] = message[:50] + "..."
        
        # Detect utilities or common code
        if 'utility' in message_lower or 'helper' in message_lower:
            utility = f"Utility from {task.id}: {message[:80]}..."
            if utility not in self.session_context['common_utilities']:
                self.session_context['common_utilities'].append(utility)
    
    async def _verify_task_completion(self, task: Task, workspace_path: str) -> bool:
        """Verify task completion using ACI tools."""
        try:
            # Check if files were created/modified
            for file_path in task.files_to_create_or_modify:
                full_path = f"{workspace_path}/{file_path}"
                if not await self.aci.file_exists(full_path):
                    logger.warning(f"Expected file not found: {file_path}")
                    return False
            
            # Run tests if specified
            criteria = task.acceptance_criteria
            if criteria.tests:
                for test in criteria.tests:
                    test_result = await self.aci.run_test(
                        workspace_path, test.file, test.function
                    )
                    if not test_result.success:
                        logger.warning(f"Test failed: {test.file}::{test.function}")
                        return False
            
            # Run linting if specified
            if criteria.linting and 'command' in criteria.linting:
                lint_result = await self.aci.run_command(
                    workspace_path, criteria.linting['command']
                )
                if lint_result.return_code != 0:
                    logger.warning(f"Linting failed: {criteria.linting['command']}")
                    return False
            
            logger.info(f"Task {task.id} verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying task completion: {e}")
            return False
    
    def _get_session_updates(self) -> Dict[str, Any]:
        """Get updates to session context for Master Claude."""
        return {
            'new_patterns': self.session_context['project_patterns'][-3:],  # Last 3 patterns
            'updated_conventions': self.session_context['coding_style'],
            'new_utilities': self.session_context['common_utilities'][-2:],  # Last 2 utilities
        }
    
    def _extract_learned_patterns(self) -> List[str]:
        """Extract patterns learned during this task for Master Claude."""
        patterns = []
        
        # Extract from session context
        if self.session_context['project_patterns']:
            patterns.extend(self.session_context['project_patterns'][-2:])
        
        # Add any specific insights
        patterns.append("Task completed with Claude CLI")
        patterns.append("Built upon existing project patterns")
        
        return patterns
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of this execution session for Master Claude."""
        return {
            'total_patterns_learned': len(self.session_context['project_patterns']),
            'coding_conventions_established': len(self.session_context['coding_style']),
            'utilities_identified': len(self.session_context['common_utilities']),
            'session_context': self.session_context,
            'cli_executor': True
        }
    
    def inject_master_insights(self, insights: Dict[str, Any]) -> None:
        """Receive insights from Master Claude to improve future executions."""
        
        # Update session context with Master's insights
        if 'patterns' in insights:
            self.session_context['project_patterns'].extend(insights['patterns'])
        
        if 'conventions' in insights:
            self.session_context['coding_style'].update(insights['conventions'])
        
        if 'utilities' in insights:
            self.session_context['common_utilities'].extend(insights['utilities'])
        
        logger.info("Received insights from Master Claude for improved CLI execution")
    
    def cleanup(self):
        """Clean up session directory and resources."""
        if self.session_dir and os.path.exists(self.session_dir):
            import shutil
            shutil.rmtree(self.session_dir)
            logger.info(f"Cleaned up session directory: {self.session_dir}")


class PersistentClaudeCliSession:
    """Manages persistent Claude CLI sessions for better context continuity."""
    
    def __init__(self, config: Config):
        self.config = config
        self.active_sessions: Dict[str, ClaudeCliExecutor] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def get_or_create_session(self, project_area: str) -> ClaudeCliExecutor:
        """Get existing session or create new one for project area."""
        
        if project_area not in self.active_sessions:
            # Create new session
            executor = ClaudeCliExecutor(self.config)
            self.active_sessions[project_area] = executor
            self.session_metadata[project_area] = {
                'created_at': datetime.now().isoformat(),
                'task_count': 0,
                'last_used': datetime.now().isoformat(),
                'executor_type': 'cli'
            }
            
            logger.info(f"Created new Claude CLI session for {project_area}")
        else:
            # Update last used time
            self.session_metadata[project_area]['last_used'] = datetime.now().isoformat()
        
        return self.active_sessions[project_area]
    
    async def execute_task_with_session_continuity(self, 
                                                  task: Task,
                                                  workspace_path: str,
                                                  master_context: str,
                                                  task_specific_context: str) -> Dict[str, Any]:
        """Execute task using appropriate persistent CLI session."""
        
        # Get or create session for this project area
        executor = await self.get_or_create_session(task.project_area)
        
        # Execute task
        result = await executor.execute_task_with_curated_context(
            task, workspace_path, master_context, task_specific_context
        )
        
        # Update session metadata
        self.session_metadata[task.project_area]['task_count'] += 1
        
        return result
    
    def get_session_insights(self) -> Dict[str, Any]:
        """Get insights from all active CLI sessions."""
        insights = {}
        
        for area, executor in self.active_sessions.items():
            insights[area] = executor.get_session_summary()
        
        return insights
    
    async def cleanup_idle_sessions(self, max_idle_hours: int = 2) -> None:
        """Clean up sessions that haven't been used recently."""
        current_time = datetime.now()
        to_remove = []
        
        for area, metadata in self.session_metadata.items():
            last_used = datetime.fromisoformat(metadata['last_used'])
            idle_hours = (current_time - last_used).total_seconds() / 3600
            
            if idle_hours > max_idle_hours:
                to_remove.append(area)
        
        for area in to_remove:
            if area in self.active_sessions:
                self.active_sessions[area].cleanup()
                del self.active_sessions[area]
            del self.session_metadata[area]
            logger.info(f"Cleaned up idle CLI session for {area}")
    
    async def broadcast_insights_to_sessions(self, master_insights: Dict[str, Any]) -> None:
        """Broadcast Master Claude's insights to all active CLI sessions."""
        for executor in self.active_sessions.values():
            executor.inject_master_insights(master_insights)
    
    def cleanup_all_sessions(self):
        """Clean up all active sessions."""
        for executor in self.active_sessions.values():
            executor.cleanup()
        self.active_sessions.clear()
        self.session_metadata.clear()
        logger.info("Cleaned up all CLI sessions")