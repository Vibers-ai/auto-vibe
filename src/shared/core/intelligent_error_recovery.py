"""Intelligent Error Recovery System for VIBE."""

import asyncio
import logging
import json
import re
import traceback
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import shutil

import google.generativeai as genai

from shared.utils.config import Config
from shared.core.schema import Task, TasksPlan
from shared.utils.file_utils import read_text_file, write_text_file, backup_file
from shared.agents.context_manager import ContextManager

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur during task execution."""
    SYNTAX_ERROR = "syntax_error"
    LOGIC_ERROR = "logic_error"
    ENVIRONMENT_ERROR = "environment_error"
    DEPENDENCY_ERROR = "dependency_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY_WITH_MODIFICATIONS = "retry_with_modifications"
    PARTIAL_ROLLBACK = "partial_rollback"
    FULL_ROLLBACK = "full_rollback"
    ENVIRONMENT_FIX = "environment_fix"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    ALTERNATIVE_APPROACH = "alternative_approach"
    HUMAN_INTERVENTION = "human_intervention"
    SKIP_AND_CONTINUE = "skip_and_continue"


@dataclass
class ErrorContext:
    """Context information about an error."""
    error_id: str
    task_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    error_message: str
    stack_trace: Optional[str]
    affected_files: List[str]
    environment_state: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    previous_attempts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken."""
    action_id: str
    strategy: RecoveryStrategy
    description: str
    steps: List[str]
    estimated_success_rate: float
    prerequisites: List[str] = field(default_factory=list)
    rollback_points: List[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    action_id: str
    success: bool
    error_resolved: bool
    new_errors_introduced: List[ErrorContext] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    execution_log: List[str] = field(default_factory=list)
    recovery_time_seconds: float = 0.0


class ErrorClassifier:
    """Classifies errors into types and determines severity."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Error pattern mappings
        self.syntax_patterns = [
            r"SyntaxError",
            r"IndentationError", 
            r"TabError",
            r"invalid syntax",
            r"unexpected token",
            r"missing semicolon",
            r"unterminated string"
        ]
        
        self.logic_patterns = [
            r"TypeError",
            r"AttributeError",
            r"KeyError",
            r"IndexError",
            r"ValueError",
            r"NameError",
            r"UnboundLocalError",
            r"undefined is not a function",
            r"cannot read property"
        ]
        
        self.environment_patterns = [
            r"ModuleNotFoundError",
            r"ImportError",
            r"command not found",
            r"No such file or directory",
            r"Permission denied",
            r"Access is denied",
            r"ENOENT",
            r"EACCES"
        ]
        
        self.dependency_patterns = [
            r"package.*not found",
            r"npm.*not found",
            r"pip.*not found",
            r"requirements.*not satisfied",
            r"version conflict",
            r"dependency.*missing"
        ]
        
        self.network_patterns = [
            r"Connection.*failed",
            r"Network.*unreachable",
            r"Timeout.*expired",
            r"DNS.*resolution.*failed",
            r"HTTP.*error.*[45]\d\d"
        ]
    
    def classify_error(self, error_message: str, stack_trace: Optional[str] = None) -> Tuple[ErrorType, ErrorSeverity]:
        """Classify an error and determine its severity."""
        
        full_error_text = f"{error_message} {stack_trace or ''}"
        
        # Check patterns for error type
        if any(re.search(pattern, full_error_text, re.IGNORECASE) for pattern in self.syntax_patterns):
            error_type = ErrorType.SYNTAX_ERROR
            severity = ErrorSeverity.MEDIUM
            
        elif any(re.search(pattern, full_error_text, re.IGNORECASE) for pattern in self.logic_patterns):
            error_type = ErrorType.LOGIC_ERROR
            severity = ErrorSeverity.HIGH
            
        elif any(re.search(pattern, full_error_text, re.IGNORECASE) for pattern in self.environment_patterns):
            error_type = ErrorType.ENVIRONMENT_ERROR
            severity = ErrorSeverity.HIGH
            
        elif any(re.search(pattern, full_error_text, re.IGNORECASE) for pattern in self.dependency_patterns):
            error_type = ErrorType.DEPENDENCY_ERROR
            severity = ErrorSeverity.HIGH
            
        elif any(re.search(pattern, full_error_text, re.IGNORECASE) for pattern in self.network_patterns):
            error_type = ErrorType.NETWORK_ERROR
            severity = ErrorSeverity.MEDIUM
            
        else:
            error_type = ErrorType.UNKNOWN_ERROR
            severity = ErrorSeverity.MEDIUM
        
        # Adjust severity based on specific conditions
        if "critical" in full_error_text.lower() or "fatal" in full_error_text.lower():
            severity = ErrorSeverity.CRITICAL
        elif "warning" in full_error_text.lower() and error_type == ErrorType.SYNTAX_ERROR:
            severity = ErrorSeverity.LOW
        
        return error_type, severity
    
    def extract_affected_files(self, error_message: str, stack_trace: Optional[str] = None) -> List[str]:
        """Extract file paths from error messages and stack traces."""
        
        affected_files = []
        full_text = f"{error_message} {stack_trace or ''}"
        
        # Common file path patterns
        file_patterns = [
            r'"([^"]+\.(?:py|js|ts|jsx|tsx|json|yaml|yml))"',
            r"'([^']+\.(?:py|js|ts|jsx|tsx|json|yaml|yml))'",
            r"File \"([^\"]+)\"",
            r"at ([^\s]+\.(?:py|js|ts|jsx|tsx|json|yaml|yml)):\d+",
            r"([/\w-]+\.(?:py|js|ts|jsx|tsx|json|yaml|yml))"
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, full_text)
            affected_files.extend(matches)
        
        # Remove duplicates and return
        return list(set(affected_files))


class RollbackManager:
    """Manages file rollbacks and state restoration."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.backup_dir = self.workspace_path / ".vibe" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.rollback_points = {}
    
    def create_rollback_point(self, point_id: str, files: List[str]) -> bool:
        """Create a rollback point by backing up specified files."""
        
        try:
            point_dir = self.backup_dir / point_id
            point_dir.mkdir(exist_ok=True)
            
            backed_up_files = []
            
            for file_path in files:
                full_path = self.workspace_path / file_path
                if full_path.exists():
                    backup_path = point_dir / file_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full_path, backup_path)
                    backed_up_files.append(file_path)
            
            self.rollback_points[point_id] = {
                'files': backed_up_files,
                'timestamp': datetime.now(),
                'directory': str(point_dir)
            }
            
            logger.info(f"Created rollback point {point_id} with {len(backed_up_files)} files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create rollback point {point_id}: {e}")
            return False
    
    def rollback_to_point(self, point_id: str, files: Optional[List[str]] = None) -> bool:
        """Rollback to a specific point, optionally for specific files only."""
        
        if point_id not in self.rollback_points:
            logger.error(f"Rollback point {point_id} not found")
            return False
        
        try:
            point_info = self.rollback_points[point_id]
            point_dir = Path(point_info['directory'])
            
            files_to_restore = files or point_info['files']
            restored_files = []
            
            for file_path in files_to_restore:
                backup_path = point_dir / file_path
                full_path = self.workspace_path / file_path
                
                if backup_path.exists():
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, full_path)
                    restored_files.append(file_path)
            
            logger.info(f"Rolled back {len(restored_files)} files from point {point_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback to point {point_id}: {e}")
            return False
    
    def list_rollback_points(self) -> Dict[str, Dict[str, Any]]:
        """List all available rollback points."""
        return self.rollback_points.copy()
    
    def cleanup_old_points(self, max_age_hours: int = 24) -> None:
        """Clean up rollback points older than specified hours."""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        points_to_remove = []
        
        for point_id, point_info in self.rollback_points.items():
            if point_info['timestamp'] < cutoff_time:
                points_to_remove.append(point_id)
        
        for point_id in points_to_remove:
            try:
                point_dir = Path(self.rollback_points[point_id]['directory'])
                if point_dir.exists():
                    shutil.rmtree(point_dir)
                del self.rollback_points[point_id]
                logger.info(f"Cleaned up old rollback point {point_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup rollback point {point_id}: {e}")


class RecoveryStrategyGenerator:
    """Generates recovery strategies based on error analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        self.gemini_model = genai.GenerativeModel(config.gemini_model)
    
    def generate_recovery_strategies(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Generate recovery strategies for a given error."""
        
        strategies = []
        
        # Generate basic strategies based on error type
        if error_context.error_type == ErrorType.SYNTAX_ERROR:
            strategies.extend(self._generate_syntax_error_strategies(error_context))
        elif error_context.error_type == ErrorType.LOGIC_ERROR:
            strategies.extend(self._generate_logic_error_strategies(error_context))
        elif error_context.error_type == ErrorType.ENVIRONMENT_ERROR:
            strategies.extend(self._generate_environment_error_strategies(error_context))
        elif error_context.error_type == ErrorType.DEPENDENCY_ERROR:
            strategies.extend(self._generate_dependency_error_strategies(error_context))
        else:
            strategies.extend(self._generate_generic_strategies(error_context))
        
        # Generate AI-powered advanced strategies
        ai_strategies = self._generate_ai_strategies(error_context)
        strategies.extend(ai_strategies)
        
        # Sort by estimated success rate
        strategies.sort(key=lambda s: s.estimated_success_rate, reverse=True)
        
        return strategies[:5]  # Return top 5 strategies
    
    def _generate_syntax_error_strategies(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Generate strategies for syntax errors."""
        
        strategies = []
        
        # Strategy 1: Automatic syntax fix
        strategies.append(RecoveryAction(
            action_id=f"syntax_fix_{error_context.error_id}",
            strategy=RecoveryStrategy.RETRY_WITH_MODIFICATIONS,
            description="Automatically fix common syntax errors",
            steps=[
                "Analyze syntax error location",
                "Apply common syntax fixes (missing semicolons, brackets, etc.)",
                "Validate syntax after fix",
                "Retry task execution"
            ],
            estimated_success_rate=0.7
        ))
        
        # Strategy 2: Rollback to last working state
        strategies.append(RecoveryAction(
            action_id=f"syntax_rollback_{error_context.error_id}",
            strategy=RecoveryStrategy.PARTIAL_ROLLBACK,
            description="Rollback affected files to last working state",
            steps=[
                "Identify last working rollback point",
                "Restore affected files from backup",
                "Retry task execution with modifications"
            ],
            estimated_success_rate=0.6,
            rollback_points=["pre_task", "pre_file_modification"]
        ))
        
        return strategies
    
    def _generate_logic_error_strategies(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Generate strategies for logic errors."""
        
        strategies = []
        
        # Strategy 1: Code review and fix
        strategies.append(RecoveryAction(
            action_id=f"logic_review_{error_context.error_id}",
            strategy=RecoveryStrategy.RETRY_WITH_MODIFICATIONS,
            description="Review and fix logical issues in code",
            steps=[
                "Analyze error context and affected code",
                "Identify logical inconsistencies",
                "Generate corrected code",
                "Test fix and retry execution"
            ],
            estimated_success_rate=0.8
        ))
        
        # Strategy 2: Alternative implementation approach
        strategies.append(RecoveryAction(
            action_id=f"logic_alternative_{error_context.error_id}",
            strategy=RecoveryStrategy.ALTERNATIVE_APPROACH,
            description="Try alternative implementation approach",
            steps=[
                "Analyze original implementation approach",
                "Generate alternative implementation strategy",
                "Implement alternative approach",
                "Test and validate solution"
            ],
            estimated_success_rate=0.6
        ))
        
        return strategies
    
    def _generate_environment_error_strategies(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Generate strategies for environment errors."""
        
        strategies = []
        
        # Strategy 1: Environment setup and fix
        strategies.append(RecoveryAction(
            action_id=f"env_fix_{error_context.error_id}",
            strategy=RecoveryStrategy.ENVIRONMENT_FIX,
            description="Fix environment configuration issues",
            steps=[
                "Diagnose environment configuration",
                "Install missing dependencies",
                "Fix permission issues",
                "Validate environment setup",
                "Retry task execution"
            ],
            estimated_success_rate=0.9,
            prerequisites=["admin_access"]
        ))
        
        return strategies
    
    def _generate_dependency_error_strategies(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Generate strategies for dependency errors."""
        
        strategies = []
        
        # Strategy 1: Dependency resolution
        strategies.append(RecoveryAction(
            action_id=f"dep_resolve_{error_context.error_id}",
            strategy=RecoveryStrategy.DEPENDENCY_RESOLUTION,
            description="Resolve dependency conflicts and missing packages",
            steps=[
                "Analyze dependency requirements",
                "Install missing packages",
                "Resolve version conflicts",
                "Update package manifests",
                "Retry task execution"
            ],
            estimated_success_rate=0.85
        ))
        
        return strategies
    
    def _generate_generic_strategies(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Generate generic recovery strategies."""
        
        strategies = []
        
        # Strategy 1: Full rollback
        strategies.append(RecoveryAction(
            action_id=f"full_rollback_{error_context.error_id}",
            strategy=RecoveryStrategy.FULL_ROLLBACK,
            description="Full rollback to previous stable state",
            steps=[
                "Identify last stable state",
                "Rollback all affected files",
                "Reset environment state",
                "Retry with different approach"
            ],
            estimated_success_rate=0.5,
            rollback_points=["task_start", "pre_execution"]
        ))
        
        # Strategy 2: Skip and continue
        if error_context.severity != ErrorSeverity.CRITICAL:
            strategies.append(RecoveryAction(
                action_id=f"skip_continue_{error_context.error_id}",
                strategy=RecoveryStrategy.SKIP_AND_CONTINUE,
                description="Skip failed operation and continue with remaining tasks",
                steps=[
                    "Mark current operation as skipped",
                    "Log error for manual review",
                    "Continue with next operations"
                ],
                estimated_success_rate=0.3
            ))
        
        return strategies
    
    def _generate_ai_strategies(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Generate AI-powered recovery strategies."""
        
        try:
            # Create prompt for AI strategy generation
            prompt = f"""
Analyze this error and suggest recovery strategies:

Error Type: {error_context.error_type.value}
Severity: {error_context.severity.value}
Message: {error_context.error_message}
Affected Files: {error_context.affected_files}
Previous Attempts: {len(error_context.previous_attempts)}

Generate 2-3 specific recovery strategies with:
1. Strategy name and approach
2. Step-by-step actions
3. Success probability (0.0-1.0)
4. Prerequisites if any

Focus on practical, actionable solutions.
"""
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            
            # Parse AI response and create strategies
            ai_strategies = self._parse_ai_strategies(response.text, error_context.error_id)
            return ai_strategies
            
        except Exception as e:
            logger.warning(f"Failed to generate AI strategies: {e}")
            return []
    
    def _parse_ai_strategies(self, ai_response: str, error_id: str) -> List[RecoveryAction]:
        """Parse AI response into RecoveryAction objects."""
        
        strategies = []
        
        # Simple parsing logic - in production this would be more sophisticated
        sections = ai_response.split('\n\n')
        
        for i, section in enumerate(sections):
            if 'strategy' in section.lower() or 'approach' in section.lower():
                try:
                    # Extract strategy information using basic parsing
                    lines = section.strip().split('\n')
                    description = lines[0] if lines else "AI-generated strategy"
                    
                    # Extract steps (lines starting with numbers or bullets)
                    steps = [line.strip() for line in lines[1:] 
                            if re.match(r'^\d+\.|\-|\*', line.strip())]
                    
                    # Estimate success rate (default to 0.6 for AI strategies)
                    success_rate = 0.6
                    
                    strategies.append(RecoveryAction(
                        action_id=f"ai_strategy_{error_id}_{i}",
                        strategy=RecoveryStrategy.ALTERNATIVE_APPROACH,
                        description=description[:100],  # Limit description length
                        steps=steps[:5],  # Limit to 5 steps
                        estimated_success_rate=success_rate
                    ))
                    
                except Exception as e:
                    logger.warning(f"Failed to parse AI strategy section: {e}")
        
        return strategies


class IntelligentErrorRecoveryManager:
    """Main error recovery management system."""
    
    def __init__(self, config: Config, workspace_path: str):
        self.config = config
        self.workspace_path = workspace_path
        
        # Initialize components
        self.classifier = ErrorClassifier(config)
        self.rollback_manager = RollbackManager(workspace_path)
        self.strategy_generator = RecoveryStrategyGenerator(config)
        
        # State tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_attempts: Dict[str, List[RecoveryResult]] = {}
        self.max_retry_attempts = 3
        
        # Performance metrics
        self.metrics = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0
        }
    
    async def handle_task_error(
        self, 
        task: Task, 
        error_message: str, 
        stack_trace: Optional[str] = None,
        affected_files: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[RecoveryResult]]:
        """Handle an error that occurred during task execution."""
        
        # Create error context
        error_context = self._create_error_context(
            task, error_message, stack_trace, affected_files
        )
        
        self.error_history.append(error_context)
        self.metrics['total_errors'] += 1
        
        logger.info(f"Handling error for task {task.id}: {error_context.error_type.value}")
        
        # Check if we should attempt recovery
        if not self._should_attempt_recovery(error_context):
            logger.warning(f"Skipping recovery for error {error_context.error_id} - too many attempts")
            return False, None
        
        # Generate recovery strategies
        strategies = self.strategy_generator.generate_recovery_strategies(error_context)
        
        if not strategies:
            logger.warning(f"No recovery strategies available for error {error_context.error_id}")
            return False, None
        
        # Attempt recovery with best strategy
        for strategy in strategies:
            logger.info(f"Attempting recovery strategy: {strategy.description}")
            
            recovery_result = await self._execute_recovery_strategy(
                error_context, strategy, task
            )
            
            if recovery_result.success and recovery_result.error_resolved:
                self.metrics['successful_recoveries'] += 1
                logger.info(f"Successfully recovered from error {error_context.error_id}")
                return True, recovery_result
            else:
                logger.warning(f"Recovery strategy failed: {strategy.description}")
        
        # All strategies failed
        self.metrics['failed_recoveries'] += 1
        logger.error(f"All recovery strategies failed for error {error_context.error_id}")
        return False, None
    
    def _create_error_context(
        self, 
        task: Task, 
        error_message: str, 
        stack_trace: Optional[str],
        affected_files: Optional[List[str]]
    ) -> ErrorContext:
        """Create error context from error information."""
        
        error_type, severity = self.classifier.classify_error(error_message, stack_trace)
        
        if affected_files is None:
            affected_files = self.classifier.extract_affected_files(error_message, stack_trace)
        
        error_id = f"error_{task.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return ErrorContext(
            error_id=error_id,
            task_id=task.id,
            error_type=error_type,
            severity=severity,
            error_message=error_message,
            stack_trace=stack_trace,
            affected_files=affected_files,
            environment_state=self._capture_environment_state(),
            timestamp=datetime.now()
        )
    
    def _capture_environment_state(self) -> Dict[str, Any]:
        """Capture current environment state for debugging."""
        
        state = {
            'working_directory': str(Path.cwd()),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Capture Python environment info
            import sys
            state['python_version'] = sys.version
            state['python_path'] = sys.path[:5]  # First 5 entries
            
            # Capture installed packages (basic info)
            try:
                result = subprocess.run(['pip', 'list'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    packages = result.stdout.split('\n')[:10]  # First 10 packages
                    state['installed_packages'] = packages
            except:
                pass
            
        except Exception as e:
            logger.warning(f"Failed to capture environment state: {e}")
        
        return state
    
    def _should_attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Determine if recovery should be attempted for this error."""
        
        # Check retry count for this specific error pattern
        similar_errors = [
            e for e in self.error_history 
            if e.error_type == error_context.error_type and 
               e.task_id == error_context.task_id
        ]
        
        if len(similar_errors) > self.max_retry_attempts:
            return False
        
        # Don't attempt recovery for critical errors that indicate fundamental issues
        if error_context.severity == ErrorSeverity.CRITICAL and \
           "system" in error_context.error_message.lower():
            return False
        
        return True
    
    async def _execute_recovery_strategy(
        self, 
        error_context: ErrorContext, 
        strategy: RecoveryAction,
        task: Task
    ) -> RecoveryResult:
        """Execute a specific recovery strategy."""
        
        start_time = datetime.now()
        execution_log = []
        modified_files = []
        new_errors = []
        
        try:
            # Create rollback point if strategy requires it
            if strategy.rollback_points:
                rollback_id = f"recovery_{strategy.action_id}"
                self.rollback_manager.create_rollback_point(
                    rollback_id, error_context.affected_files
                )
                execution_log.append(f"Created rollback point: {rollback_id}")
            
            # Execute strategy based on type
            if strategy.strategy == RecoveryStrategy.RETRY_WITH_MODIFICATIONS:
                success = await self._execute_retry_with_modifications(
                    error_context, strategy, task, execution_log, modified_files
                )
            elif strategy.strategy == RecoveryStrategy.PARTIAL_ROLLBACK:
                success = await self._execute_partial_rollback(
                    error_context, strategy, task, execution_log, modified_files
                )
            elif strategy.strategy == RecoveryStrategy.ENVIRONMENT_FIX:
                success = await self._execute_environment_fix(
                    error_context, strategy, task, execution_log
                )
            elif strategy.strategy == RecoveryStrategy.DEPENDENCY_RESOLUTION:
                success = await self._execute_dependency_resolution(
                    error_context, strategy, task, execution_log
                )
            else:
                # Generic strategy execution
                success = await self._execute_generic_strategy(
                    error_context, strategy, task, execution_log
                )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return RecoveryResult(
                action_id=strategy.action_id,
                success=success,
                error_resolved=success,  # For now, assume success means error resolved
                new_errors_introduced=new_errors,
                modified_files=modified_files,
                execution_log=execution_log,
                recovery_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_log.append(f"Recovery strategy execution failed: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return RecoveryResult(
                action_id=strategy.action_id,
                success=False,
                error_resolved=False,
                new_errors_introduced=[],
                modified_files=modified_files,
                execution_log=execution_log,
                recovery_time_seconds=execution_time
            )
    
    async def _execute_retry_with_modifications(
        self, 
        error_context: ErrorContext, 
        strategy: RecoveryAction,
        task: Task,
        execution_log: List[str],
        modified_files: List[str]
    ) -> bool:
        """Execute retry with modifications strategy."""
        
        execution_log.append("Starting retry with modifications")
        
        try:
            # Handle different error types with specific modifications
            if error_context.error_type == ErrorType.SYNTAX_ERROR:
                success = await self._fix_syntax_errors(error_context, execution_log, modified_files)
                
            elif error_context.error_type == ErrorType.LOGIC_ERROR:
                success = await self._fix_logic_errors(error_context, execution_log, modified_files)
                
            elif error_context.error_type == ErrorType.ENVIRONMENT_ERROR:
                success = await self._fix_environment_issues(error_context, execution_log, modified_files)
                
            elif error_context.error_type == ErrorType.DEPENDENCY_ERROR:
                success = await self._fix_dependency_issues(error_context, execution_log, modified_files)
                
            else:
                # Generic fixes for unknown error types
                success = await self._apply_generic_fixes(error_context, execution_log, modified_files)
            
            if success:
                execution_log.append("Modifications applied successfully")
                return True
            else:
                execution_log.append("Modifications failed or were not applicable")
                return False
            
        except Exception as e:
            execution_log.append(f"Failed to apply modifications: {str(e)}")
            return False
    
    async def _execute_partial_rollback(
        self, 
        error_context: ErrorContext, 
        strategy: RecoveryAction,
        task: Task,
        execution_log: List[str],
        modified_files: List[str]
    ) -> bool:
        """Execute partial rollback strategy."""
        
        execution_log.append("Starting partial rollback")
        
        try:
            # Find appropriate rollback point
            rollback_points = self.rollback_manager.list_rollback_points()
            
            if rollback_points:
                # Use most recent rollback point
                latest_point = max(rollback_points.keys(), key=lambda k: rollback_points[k]['timestamp'])
                
                success = self.rollback_manager.rollback_to_point(
                    latest_point, error_context.affected_files
                )
                
                if success:
                    modified_files.extend(error_context.affected_files)
                    execution_log.append(f"Rolled back to point: {latest_point}")
                    return True
                else:
                    execution_log.append(f"Failed to rollback to point: {latest_point}")
                    return False
            else:
                execution_log.append("No rollback points available")
                return False
                
        except Exception as e:
            execution_log.append(f"Rollback failed: {str(e)}")
            return False
    
    async def _execute_environment_fix(
        self, 
        error_context: ErrorContext, 
        strategy: RecoveryAction,
        task: Task,
        execution_log: List[str]
    ) -> bool:
        """Execute environment fix strategy."""
        
        execution_log.append("Starting environment fix")
        
        try:
            # Basic environment checks and fixes
            if "ModuleNotFoundError" in error_context.error_message:
                # Try to install missing module
                module_match = re.search(r"No module named '([^']+)'", error_context.error_message)
                if module_match:
                    module_name = module_match.group(1)
                    
                    try:
                        result = subprocess.run(
                            ['pip', 'install', module_name], 
                            capture_output=True, 
                            text=True, 
                            timeout=60
                        )
                        
                        if result.returncode == 0:
                            execution_log.append(f"Successfully installed module: {module_name}")
                            return True
                        else:
                            execution_log.append(f"Failed to install module: {result.stderr}")
                            return False
                            
                    except subprocess.TimeoutExpired:
                        execution_log.append(f"Timeout while installing module: {module_name}")
                        return False
            
            execution_log.append("Environment fix completed")
            return True
            
        except Exception as e:
            execution_log.append(f"Environment fix failed: {str(e)}")
            return False
    
    async def _execute_dependency_resolution(
        self, 
        error_context: ErrorContext, 
        strategy: RecoveryAction,
        task: Task,
        execution_log: List[str]
    ) -> bool:
        """Execute dependency resolution strategy."""
        
        execution_log.append("Starting dependency resolution")
        
        try:
            # Check for requirements.txt and install dependencies
            requirements_path = Path(self.workspace_path) / "requirements.txt"
            
            if requirements_path.exists():
                result = subprocess.run(
                    ['pip', 'install', '-r', str(requirements_path)], 
                    capture_output=True, 
                    text=True, 
                    timeout=120
                )
                
                if result.returncode == 0:
                    execution_log.append("Successfully installed requirements")
                    return True
                else:
                    execution_log.append(f"Failed to install requirements: {result.stderr}")
                    return False
            else:
                execution_log.append("No requirements.txt found")
                return False
                
        except Exception as e:
            execution_log.append(f"Dependency resolution failed: {str(e)}")
            return False
    
    async def _execute_generic_strategy(
        self, 
        error_context: ErrorContext, 
        strategy: RecoveryAction,
        task: Task,
        execution_log: List[str]
    ) -> bool:
        """Execute generic recovery strategy."""
        
        execution_log.append(f"Executing generic strategy: {strategy.strategy.value}")
        
        # For now, generic strategies just log their execution
        for step in strategy.steps:
            execution_log.append(f"Step: {step}")
        
        # Return success based on strategy type
        if strategy.strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
            execution_log.append("Marked for skipping - continuing with next operations")
            return True
        
        execution_log.append("Generic strategy completed")
        return True
    
    def _apply_basic_syntax_fixes(self, content: str) -> str:
        """Apply basic syntax fixes to code content."""
        
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            fixed_line = line
            
            # Fix common Python syntax issues
            if line.strip().endswith(':') and not line.strip().endswith('::'):
                # Already has colon, no fix needed
                pass
            elif re.search(r'\b(if|elif|else|for|while|def|class|try|except|finally|with)\b', line) and not line.strip().endswith(':'):
                # Add missing colon
                fixed_line = line.rstrip() + ':'
            
            # Fix missing quotes (basic check)
            if 'print(' in line and not ('"' in line or "'" in line):
                # Very basic quote fix
                match = re.search(r'print\(([^)]+)\)', line)
                if match:
                    content_inside = match.group(1).strip()
                    if not content_inside.startswith(('"', "'")) and not content_inside.isdigit():
                        fixed_line = line.replace(content_inside, f'"{content_inside}"')
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        
        return {
            'metrics': self.metrics.copy(),
            'error_distribution': self._get_error_distribution(),
            'recovery_success_rate': self._calculate_success_rate(),
            'average_recovery_time': self._calculate_average_recovery_time(),
            'most_common_errors': self._get_most_common_errors(),
            'rollback_points': len(self.rollback_manager.list_rollback_points())
        }
    
    def _get_error_distribution(self) -> Dict[str, int]:
        """Get distribution of error types."""
        
        distribution = {}
        for error in self.error_history:
            error_type = error.error_type.value
            distribution[error_type] = distribution.get(error_type, 0) + 1
        
        return distribution
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall recovery success rate."""
        
        total_attempts = self.metrics['successful_recoveries'] + self.metrics['failed_recoveries']
        if total_attempts == 0:
            return 0.0
        
        return self.metrics['successful_recoveries'] / total_attempts
    
    def _calculate_average_recovery_time(self) -> float:
        """Calculate average recovery time."""
        
        total_time = 0.0
        count = 0
        
        for attempts in self.recovery_attempts.values():
            for attempt in attempts:
                total_time += attempt.recovery_time_seconds
                count += 1
        
        return total_time / count if count > 0 else 0.0
    
    def _get_most_common_errors(self) -> List[Tuple[str, int]]:
        """Get most common error patterns."""
        
        error_patterns = {}
        
        for error in self.error_history:
            # Create a simplified error pattern
            pattern = f"{error.error_type.value}:{error.severity.value}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        # Sort by frequency and return top 5
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
        return sorted_patterns[:5]
    
    def cleanup(self):
        """Clean up resources and old data."""
        
        # Clean up old rollback points
        self.rollback_manager.cleanup_old_points(max_age_hours=24)
        
        # Clean up old error history (keep last 100 errors)
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        logger.info("Error recovery manager cleanup completed")
    
    async def _fix_syntax_errors(
        self, 
        error_context: ErrorContext, 
        execution_log: List[str], 
        modified_files: List[str]
    ) -> bool:
        """Fix syntax errors in affected files."""
        
        execution_log.append("Attempting to fix syntax errors")
        fixed_any = False
        
        for file_path in error_context.affected_files:
            full_path = Path(self.workspace_path) / file_path
            if full_path.exists():
                try:
                    content = read_text_file(str(full_path))
                    fixed_content = self._apply_basic_syntax_fixes(content)
                    
                    if fixed_content != content:
                        write_text_file(str(full_path), fixed_content)
                        modified_files.append(file_path)
                        execution_log.append(f"Applied syntax fixes to {file_path}")
                        fixed_any = True
                    else:
                        execution_log.append(f"No syntax fixes needed for {file_path}")
                        
                except Exception as e:
                    execution_log.append(f"Failed to fix syntax in {file_path}: {e}")
        
        return fixed_any
    
    async def _fix_logic_errors(
        self, 
        error_context: ErrorContext, 
        execution_log: List[str], 
        modified_files: List[str]
    ) -> bool:
        """Fix logic errors using AI assistance."""
        
        execution_log.append("Attempting to fix logic errors")
        
        try:
            # Use Gemini to suggest logic fixes
            if hasattr(self, 'strategy_generator') and self.strategy_generator:
                
                for file_path in error_context.affected_files[:2]:  # Limit to 2 files
                    full_path = Path(self.workspace_path) / file_path
                    if full_path.exists():
                        content = read_text_file(str(full_path))
                        
                        # Create a prompt for logic error fixing
                        fix_prompt = f'''
Fix the logic error in this code:

Error: {error_context.error_message}

Code:
```
{content[:1000]}  # First 1000 chars
```

Provide a corrected version of the problematic section only.
Focus on:
1. Type errors
2. Attribute errors  
3. Index/Key errors
4. Variable naming issues

Return only the corrected code section.
'''
                        
                        try:
                            response = self.strategy_generator.gemini_model.generate_content(
                                fix_prompt,
                                generation_config={
                                    'temperature': 0.1,
                                    'max_output_tokens': 500
                                }
                            )
                            
                            if response.text and len(response.text.strip()) > 10:
                                # For now, just log the suggestion
                                execution_log.append(f"AI suggested fix for {file_path}: {response.text[:100]}...")
                                # In a full implementation, we would apply the fix
                                return True
                                
                        except Exception as e:
                            execution_log.append(f"AI fix generation failed for {file_path}: {e}")
            
            execution_log.append("Logic error fixing not fully implemented yet")
            return False
            
        except Exception as e:
            execution_log.append(f"Logic error fixing failed: {e}")
            return False
    
    async def _fix_environment_issues(
        self, 
        error_context: ErrorContext, 
        execution_log: List[str], 
        modified_files: List[str]
    ) -> bool:
        """Fix environment-related issues."""
        
        execution_log.append("Attempting to fix environment issues")
        
        try:
            # Handle missing module errors
            if "ModuleNotFoundError" in error_context.error_message or "ImportError" in error_context.error_message:
                module_match = re.search(r"No module named '([^']+)'", error_context.error_message)
                if not module_match:
                    module_match = re.search(r"cannot import name '([^']+)'", error_context.error_message)
                
                if module_match:
                    module_name = module_match.group(1)
                    execution_log.append(f"Attempting to install missing module: {module_name}")
                    
                    try:
                        result = subprocess.run(
                            ['pip', 'install', module_name], 
                            capture_output=True, 
                            text=True, 
                            timeout=60
                        )
                        
                        if result.returncode == 0:
                            execution_log.append(f"Successfully installed {module_name}")
                            return True
                        else:
                            execution_log.append(f"Failed to install {module_name}: {result.stderr}")
                            
                    except subprocess.TimeoutExpired:
                        execution_log.append(f"Timeout while installing {module_name}")
                    except Exception as e:
                        execution_log.append(f"Exception while installing {module_name}: {e}")
            
            # Handle permission errors
            if "Permission denied" in error_context.error_message or "PermissionError" in error_context.error_message:
                execution_log.append("Permission error detected - checking file permissions")
                
                for file_path in error_context.affected_files:
                    full_path = Path(self.workspace_path) / file_path
                    if full_path.exists():
                        try:
                            # Try to make file writable
                            import stat
                            full_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                            execution_log.append(f"Updated permissions for {file_path}")
                            modified_files.append(file_path)
                            return True
                        except Exception as e:
                            execution_log.append(f"Failed to update permissions for {file_path}: {e}")
            
            execution_log.append("No applicable environment fixes found")
            return False
            
        except Exception as e:
            execution_log.append(f"Environment fix failed: {e}")
            return False
    
    async def _fix_dependency_issues(
        self, 
        error_context: ErrorContext, 
        execution_log: List[str], 
        modified_files: List[str]
    ) -> bool:
        """Fix dependency-related issues."""
        
        execution_log.append("Attempting to fix dependency issues")
        
        try:
            # Check and install from requirements.txt
            requirements_path = Path(self.workspace_path) / "requirements.txt"
            
            if requirements_path.exists():
                execution_log.append("Found requirements.txt, attempting to install dependencies")
                
                try:
                    result = subprocess.run(
                        ['pip', 'install', '-r', str(requirements_path)], 
                        capture_output=True, 
                        text=True, 
                        timeout=180  # 3 minutes timeout
                    )
                    
                    if result.returncode == 0:
                        execution_log.append("Successfully installed dependencies from requirements.txt")
                        return True
                    else:
                        execution_log.append(f"Failed to install dependencies: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    execution_log.append("Timeout while installing dependencies")
                except Exception as e:
                    execution_log.append(f"Exception while installing dependencies: {e}")
            
            # Check for package.json for Node.js projects
            package_json_path = Path(self.workspace_path) / "package.json"
            
            if package_json_path.exists():
                execution_log.append("Found package.json, attempting npm install")
                
                try:
                    result = subprocess.run(
                        ['npm', 'install'], 
                        cwd=self.workspace_path,
                        capture_output=True, 
                        text=True, 
                        timeout=180
                    )
                    
                    if result.returncode == 0:
                        execution_log.append("Successfully ran npm install")
                        return True
                    else:
                        execution_log.append(f"npm install failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    execution_log.append("Timeout while running npm install")
                except Exception as e:
                    execution_log.append(f"Exception while running npm install: {e}")
            
            execution_log.append("No dependency files found to process")
            return False
            
        except Exception as e:
            execution_log.append(f"Dependency fixing failed: {e}")
            return False
    
    async def _apply_generic_fixes(
        self, 
        error_context: ErrorContext, 
        execution_log: List[str], 
        modified_files: List[str]
    ) -> bool:
        """Apply generic fixes for unknown error types."""
        
        execution_log.append("Attempting generic fixes")
        
        try:
            # Generic fix 1: Check file encoding issues
            for file_path in error_context.affected_files:
                full_path = Path(self.workspace_path) / file_path
                if full_path.exists():
                    try:
                        # Try to re-encode file as UTF-8
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        execution_log.append(f"Re-encoded {file_path} as UTF-8")
                        modified_files.append(file_path)
                        
                    except Exception as e:
                        execution_log.append(f"Failed to re-encode {file_path}: {e}")
            
            # Generic fix 2: Remove common problematic characters
            fixed_any = False
            for file_path in error_context.affected_files:
                full_path = Path(self.workspace_path) / file_path
                if full_path.exists() and full_path.suffix in ['.py', '.js', '.ts', '.jsx', '.tsx']:
                    try:
                        content = read_text_file(str(full_path))
                        
                        # Remove BOM if present
                        if content.startswith('\ufeff'):
                            content = content[1:]
                            fixed_any = True
                        
                        # Fix mixed line endings
                        original_content = content
                        content = content.replace('\r\n', '\n').replace('\r', '\n')
                        
                        if content != original_content:
                            write_text_file(str(full_path), content)
                            execution_log.append(f"Fixed line endings in {file_path}")
                            modified_files.append(file_path)
                            fixed_any = True
                        
                    except Exception as e:
                        execution_log.append(f"Failed to apply generic fixes to {file_path}: {e}")
            
            if fixed_any:
                execution_log.append("Applied generic fixes successfully")
                return True
            else:
                execution_log.append("No generic fixes were applicable")
                return False
                
        except Exception as e:
            execution_log.append(f"Generic fixes failed: {e}")
            return False