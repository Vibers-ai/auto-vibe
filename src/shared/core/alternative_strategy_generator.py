"""Alternative Strategy Generator for VIBE

This module generates alternative implementation strategies when primary approaches fail,
using pattern analysis, similar code detection, and AI-assisted suggestions.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
from pathlib import Path
import ast
import difflib

from shared.core.failure_pattern_db import FailurePatternDB, FailureContext, FailureType
from shared.core.schema import Task
from shared.core.enhanced_logger import get_logger, LogCategory

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of alternative strategies."""
    REFACTOR = "refactor"              # Refactor existing approach
    ALTERNATIVE_LIB = "alternative_lib" # Use different library
    WORKAROUND = "workaround"          # Temporary workaround
    SIMPLIFIED = "simplified"          # Simplified implementation
    INCREMENTAL = "incremental"        # Break into smaller steps
    FALLBACK = "fallback"              # Fallback to basic implementation
    PARALLEL = "parallel"              # Try multiple approaches
    CUSTOM = "custom"                  # Custom strategy


@dataclass
class ImplementationStrategy:
    """Represents an alternative implementation strategy."""
    strategy_id: str
    strategy_type: StrategyType
    description: str
    confidence_score: float
    
    # Implementation details
    approach: Dict[str, Any]
    code_snippets: List[str] = field(default_factory=list)
    required_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Dependencies and requirements
    new_dependencies: List[str] = field(default_factory=list)
    removed_dependencies: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Risk assessment
    risks: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    
    # Examples and references
    similar_implementations: List[Dict[str, Any]] = field(default_factory=list)
    documentation_links: List[str] = field(default_factory=list)


@dataclass
class StrategyContext:
    """Context for strategy generation."""
    task: Task
    failure_context: FailureContext
    previous_attempts: List[Dict[str, Any]]
    current_code: Optional[str] = None
    target_file: Optional[str] = None
    project_structure: Optional[Dict[str, Any]] = None
    available_libraries: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


class AlternativeStrategyGenerator:
    """Generates alternative implementation strategies."""
    
    def __init__(self):
        # Pattern database for learning from failures
        self.failure_db = FailurePatternDB()
        
        # Strategy templates
        self.strategy_templates = self._initialize_strategy_templates()
        
        # Code pattern recognizers
        self.code_patterns = self._initialize_code_patterns()
        
        # Library alternatives mapping
        self.library_alternatives = self._initialize_library_alternatives()
        
        # Statistics
        self.stats = {
            'strategies_generated': 0,
            'successful_strategies': 0,
            'failed_strategies': 0,
            'most_effective_types': {}
        }
        
        # Enhanced logger
        self.logger = get_logger(
            component="strategy_generator",
            session_id=f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def generate_strategies(self, context: StrategyContext) -> List[ImplementationStrategy]:
        """Generate alternative strategies based on context."""
        strategies = []
        
        # Analyze failure patterns
        failure_patterns = self.failure_db.get_pattern_suggestions(context.failure_context)
        
        # Generate strategies based on failure type
        if context.failure_context.failure_type == FailureType.IMPORT_ERROR:
            strategies.extend(self._generate_import_strategies(context))
        elif context.failure_context.failure_type == FailureType.SYNTAX_ERROR:
            strategies.extend(self._generate_syntax_strategies(context))
        elif context.failure_context.failure_type == FailureType.TIMEOUT:
            strategies.extend(self._generate_performance_strategies(context))
        elif context.failure_context.failure_type == FailureType.RUNTIME_ERROR:
            strategies.extend(self._generate_runtime_strategies(context))
        
        # Generate generic strategies
        strategies.extend(self._generate_generic_strategies(context))
        
        # Learn from similar successful implementations
        similar_strategies = self._find_similar_successful_implementations(context)
        strategies.extend(similar_strategies)
        
        # Rank strategies by confidence
        strategies.sort(key=lambda s: s.confidence_score, reverse=True)
        
        # Limit to top strategies
        strategies = strategies[:5]
        
        self.stats['strategies_generated'] += len(strategies)
        
        self.logger.log(
            level=logging.INFO,
            category=LogCategory.TASK_EXECUTION,
            message=f"Generated {len(strategies)} alternative strategies",
            metadata={
                'task_id': context.task.id,
                'failure_type': context.failure_context.failure_type.value,
                'top_strategy': strategies[0].strategy_type.value if strategies else None
            }
        )
        
        return strategies
    
    def evaluate_strategy(self, strategy: ImplementationStrategy, 
                         success: bool, execution_time: float,
                         error_info: Optional[Dict[str, Any]] = None):
        """Evaluate strategy effectiveness after execution."""
        if success:
            self.stats['successful_strategies'] += 1
            # Update effectiveness metrics
            strategy_type = strategy.strategy_type.value
            if strategy_type not in self.stats['most_effective_types']:
                self.stats['most_effective_types'][strategy_type] = {'success': 0, 'total': 0}
            self.stats['most_effective_types'][strategy_type]['success'] += 1
            self.stats['most_effective_types'][strategy_type]['total'] += 1
        else:
            self.stats['failed_strategies'] += 1
            if error_info:
                # Learn from failure
                self._learn_from_strategy_failure(strategy, error_info)
        
        self.logger.log(
            level=logging.INFO,
            category=LogCategory.ERROR_RECOVERY,
            message=f"Strategy evaluation: {'success' if success else 'failed'}",
            metadata={
                'strategy_id': strategy.strategy_id,
                'strategy_type': strategy.strategy_type.value,
                'execution_time': execution_time,
                'confidence_score': strategy.confidence_score
            }
        )
    
    def get_strategy_explanation(self, strategy: ImplementationStrategy) -> str:
        """Generate human-readable explanation of strategy."""
        explanation = f"## {strategy.description}\n\n"
        
        explanation += f"**Strategy Type:** {strategy.strategy_type.value}\n"
        explanation += f"**Confidence:** {strategy.confidence_score:.2f}\n"
        explanation += f"**Estimated Effort:** {strategy.estimated_effort}\n\n"
        
        if strategy.approach:
            explanation += "### Approach\n"
            for key, value in strategy.approach.items():
                explanation += f"- **{key}:** {value}\n"
            explanation += "\n"
        
        if strategy.required_changes:
            explanation += "### Required Changes\n"
            for change in strategy.required_changes:
                explanation += f"- {change.get('description', 'Change required')}\n"
            explanation += "\n"
        
        if strategy.benefits:
            explanation += "### Benefits\n"
            for benefit in strategy.benefits:
                explanation += f"- {benefit}\n"
            explanation += "\n"
        
        if strategy.risks:
            explanation += "### Risks\n"
            for risk in strategy.risks:
                explanation += f"- {risk}\n"
            explanation += "\n"
        
        if strategy.code_snippets:
            explanation += "### Example Code\n```python\n"
            explanation += "\n".join(strategy.code_snippets[:2])  # First 2 snippets
            explanation += "\n```\n"
        
        return explanation
    
    # Private methods for strategy generation
    
    def _generate_import_strategies(self, context: StrategyContext) -> List[ImplementationStrategy]:
        """Generate strategies for import errors."""
        strategies = []
        error_msg = context.failure_context.error_message
        
        # Extract missing module
        module_match = re.search(r"No module named '(\w+)'", error_msg)
        if module_match:
            missing_module = module_match.group(1)
            
            # Strategy 1: Install missing package
            strategies.append(ImplementationStrategy(
                strategy_id=f"install_{missing_module}",
                strategy_type=StrategyType.WORKAROUND,
                description=f"Install missing package: {missing_module}",
                confidence_score=0.9,
                approach={
                    'action': 'install_package',
                    'package': missing_module,
                    'method': 'pip'
                },
                code_snippets=[f"pip install {missing_module}"],
                required_changes=[{
                    'type': 'command',
                    'description': f"Run: pip install {missing_module}"
                }],
                new_dependencies=[missing_module],
                benefits=["Quick fix for missing dependency"],
                risks=["Package might not exist with this exact name"],
                estimated_effort="low"
            ))
            
            # Strategy 2: Find alternative library
            alternatives = self.library_alternatives.get(missing_module, [])
            for alt in alternatives[:2]:  # Top 2 alternatives
                strategies.append(ImplementationStrategy(
                    strategy_id=f"alternative_{alt['name']}",
                    strategy_type=StrategyType.ALTERNATIVE_LIB,
                    description=f"Use alternative library: {alt['name']}",
                    confidence_score=alt.get('confidence', 0.7),
                    approach={
                        'action': 'replace_library',
                        'original': missing_module,
                        'alternative': alt['name'],
                        'changes_required': alt.get('changes', [])
                    },
                    code_snippets=alt.get('examples', []),
                    required_changes=[{
                        'type': 'refactor',
                        'description': f"Replace {missing_module} with {alt['name']}"
                    }],
                    new_dependencies=[alt['name']],
                    removed_dependencies=[missing_module],
                    benefits=alt.get('benefits', ["Alternative solution available"]),
                    risks=alt.get('risks', ["API differences may require code changes"]),
                    estimated_effort=alt.get('effort', 'medium')
                ))
        
        # Strategy 3: Conditional import with fallback
        strategies.append(ImplementationStrategy(
            strategy_id="conditional_import",
            strategy_type=StrategyType.FALLBACK,
            description="Implement conditional import with fallback",
            confidence_score=0.6,
            approach={
                'action': 'conditional_import',
                'pattern': 'try_except'
            },
            code_snippets=[
                """try:
    import problematic_module
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False
    # Implement fallback functionality""",
                """if HAS_MODULE:
    # Use the module
    result = problematic_module.function()
else:
    # Use fallback implementation
    result = fallback_function()"""
            ],
            required_changes=[{
                'type': 'refactor',
                'description': "Wrap imports in try-except blocks"
            }],
            benefits=["Graceful degradation", "Works without the dependency"],
            risks=["Limited functionality without the module"],
            estimated_effort="medium"
        ))
        
        return strategies
    
    def _generate_syntax_strategies(self, context: StrategyContext) -> List[ImplementationStrategy]:
        """Generate strategies for syntax errors."""
        strategies = []
        
        # Strategy 1: Fix common syntax patterns
        if context.current_code:
            syntax_fixes = self._analyze_syntax_issues(context.current_code)
            for fix in syntax_fixes:
                strategies.append(ImplementationStrategy(
                    strategy_id=f"syntax_fix_{fix['type']}",
                    strategy_type=StrategyType.REFACTOR,
                    description=f"Fix {fix['description']}",
                    confidence_score=fix['confidence'],
                    approach={
                        'action': 'fix_syntax',
                        'issue_type': fix['type'],
                        'line': fix.get('line')
                    },
                    code_snippets=[fix['suggested_fix']],
                    required_changes=[{
                        'type': 'syntax_fix',
                        'description': fix['description'],
                        'location': fix.get('location')
                    }],
                    benefits=["Fixes syntax error", "Minimal changes required"],
                    risks=["May not address underlying logic issues"],
                    estimated_effort="low"
                ))
        
        # Strategy 2: Simplify complex expressions
        strategies.append(ImplementationStrategy(
            strategy_id="simplify_code",
            strategy_type=StrategyType.SIMPLIFIED,
            description="Simplify complex code structures",
            confidence_score=0.7,
            approach={
                'action': 'simplify',
                'techniques': ['break_complex_expressions', 'extract_variables', 'reduce_nesting']
            },
            code_snippets=[
                """# Instead of complex one-liner
result = [func(x) for x in data if condition(x) and other_condition(x)]

# Break it down
filtered_data = [x for x in data if condition(x)]
filtered_data = [x for x in filtered_data if other_condition(x)]
result = [func(x) for x in filtered_data]"""
            ],
            required_changes=[{
                'type': 'refactor',
                'description': "Break down complex expressions"
            }],
            benefits=["More readable code", "Easier to debug", "Less prone to syntax errors"],
            risks=["Might be slightly less performant"],
            estimated_effort="medium"
        ))
        
        return strategies
    
    def _generate_performance_strategies(self, context: StrategyContext) -> List[ImplementationStrategy]:
        """Generate strategies for timeout/performance issues."""
        strategies = []
        
        # Strategy 1: Implement chunking
        strategies.append(ImplementationStrategy(
            strategy_id="implement_chunking",
            strategy_type=StrategyType.INCREMENTAL,
            description="Process data in smaller chunks",
            confidence_score=0.8,
            approach={
                'action': 'chunk_processing',
                'chunk_size': 'adaptive',
                'parallel': False
            },
            code_snippets=[
                """def process_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        yield process_chunk(chunk)""",
                """# Process with progress tracking
total = len(data)
processed = 0
for result in process_in_chunks(data):
    processed += len(result)
    progress = processed / total * 100
    print(f"Progress: {progress:.1f}%")"""
            ],
            required_changes=[{
                'type': 'refactor',
                'description': "Implement chunked processing"
            }],
            benefits=["Handles large datasets", "Progress visibility", "Memory efficient"],
            risks=["Slightly more complex code"],
            estimated_effort="medium"
        ))
        
        # Strategy 2: Add caching
        strategies.append(ImplementationStrategy(
            strategy_id="add_caching",
            strategy_type=StrategyType.WORKAROUND,
            description="Implement caching to avoid redundant computations",
            confidence_score=0.75,
            approach={
                'action': 'implement_cache',
                'cache_type': 'lru_cache',
                'scope': 'function'
            },
            code_snippets=[
                """from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_function(param):
    # Expensive computation
    return result""",
                """# Manual caching
cache = {}
def cached_function(param):
    if param in cache:
        return cache[param]
    result = expensive_computation(param)
    cache[param] = result
    return result"""
            ],
            required_changes=[{
                'type': 'optimization',
                'description': "Add caching to expensive operations"
            }],
            benefits=["Significant performance improvement", "Reduces redundant work"],
            risks=["Memory usage increases", "Cache invalidation complexity"],
            estimated_effort="low"
        ))
        
        # Strategy 3: Parallel processing
        strategies.append(ImplementationStrategy(
            strategy_id="parallel_processing",
            strategy_type=StrategyType.PARALLEL,
            description="Use parallel processing for independent operations",
            confidence_score=0.7,
            approach={
                'action': 'parallelize',
                'method': 'multiprocessing',
                'workers': 'cpu_count'
            },
            code_snippets=[
                """from multiprocessing import Pool
import multiprocessing

def parallel_process(data):
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_item, data)
    return results""",
                """# Async approach
import asyncio

async def process_async(items):
    tasks = [process_item_async(item) for item in items]
    return await asyncio.gather(*tasks)"""
            ],
            required_changes=[{
                'type': 'refactor',
                'description': "Implement parallel processing"
            }],
            benefits=["Utilizes multiple cores", "Significant speedup possible"],
            risks=["Complexity increases", "Not suitable for all operations"],
            estimated_effort="high"
        ))
        
        return strategies
    
    def _generate_runtime_strategies(self, context: StrategyContext) -> List[ImplementationStrategy]:
        """Generate strategies for runtime errors."""
        strategies = []
        
        # Strategy 1: Add comprehensive error handling
        strategies.append(ImplementationStrategy(
            strategy_id="comprehensive_error_handling",
            strategy_type=StrategyType.REFACTOR,
            description="Add comprehensive error handling",
            confidence_score=0.8,
            approach={
                'action': 'add_error_handling',
                'scope': 'comprehensive',
                'logging': True
            },
            code_snippets=[
                """try:
    # Main operation
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Specific error occurred: {e}")
    # Handle specific case
    result = fallback_value
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Generic fallback
    raise  # or return safe default""",
                """# Context manager for resource handling
from contextlib import contextmanager

@contextmanager
def safe_resource():
    resource = None
    try:
        resource = acquire_resource()
        yield resource
    finally:
        if resource:
            release_resource(resource)"""
            ],
            required_changes=[{
                'type': 'safety',
                'description': "Wrap operations in try-except blocks"
            }],
            benefits=["Graceful error recovery", "Better debugging info"],
            risks=["May hide underlying issues if overused"],
            estimated_effort="medium"
        ))
        
        # Strategy 2: Input validation
        strategies.append(ImplementationStrategy(
            strategy_id="input_validation",
            strategy_type=StrategyType.REFACTOR,
            description="Add input validation and sanitization",
            confidence_score=0.75,
            approach={
                'action': 'validate_inputs',
                'validation_type': 'comprehensive',
                'early_return': True
            },
            code_snippets=[
                """def process_data(data):
    # Validate inputs
    if not data:
        logger.warning("Empty data provided")
        return []
    
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected list or tuple, got {type(data)}")
    
    # Sanitize data
    valid_items = [item for item in data if validate_item(item)]
    
    # Process valid items
    return [process_item(item) for item in valid_items]"""
            ],
            required_changes=[{
                'type': 'validation',
                'description': "Add input validation"
            }],
            benefits=["Prevents runtime errors", "Clear error messages"],
            risks=["Additional overhead"],
            estimated_effort="low"
        ))
        
        return strategies
    
    def _generate_generic_strategies(self, context: StrategyContext) -> List[ImplementationStrategy]:
        """Generate generic fallback strategies."""
        strategies = []
        
        # Strategy 1: Step-by-step approach
        strategies.append(ImplementationStrategy(
            strategy_id="step_by_step",
            strategy_type=StrategyType.INCREMENTAL,
            description="Break task into smaller, testable steps",
            confidence_score=0.6,
            approach={
                'action': 'decompose',
                'granularity': 'fine',
                'testing': 'after_each_step'
            },
            code_snippets=[
                """# Break complex task into steps
def complex_task(data):
    # Step 1: Validation
    validated_data = validate_data(data)
    print("✓ Validation complete")
    
    # Step 2: Preprocessing
    preprocessed = preprocess_data(validated_data)
    print("✓ Preprocessing complete")
    
    # Step 3: Main processing
    result = process_data(preprocessed)
    print("✓ Processing complete")
    
    # Step 4: Post-processing
    final_result = postprocess_result(result)
    print("✓ Post-processing complete")
    
    return final_result"""
            ],
            required_changes=[{
                'type': 'refactor',
                'description': "Decompose into smaller functions"
            }],
            benefits=["Easier to debug", "Clear progress tracking", "Partial results available"],
            risks=["May be over-engineered for simple tasks"],
            estimated_effort="medium"
        ))
        
        # Strategy 2: Minimal implementation
        strategies.append(ImplementationStrategy(
            strategy_id="minimal_implementation",
            strategy_type=StrategyType.SIMPLIFIED,
            description="Implement minimal viable solution first",
            confidence_score=0.5,
            approach={
                'action': 'minimize_scope',
                'features': 'core_only',
                'iterations': 'planned'
            },
            code_snippets=[
                """# Start with the simplest possible implementation
def minimal_solution(data):
    # Just the core functionality
    return [basic_transform(item) for item in data]
    
# Later iterations can add:
# - Error handling
# - Performance optimization
# - Additional features"""
            ],
            required_changes=[{
                'type': 'simplification',
                'description': "Remove non-essential features"
            }],
            benefits=["Quick to implement", "Easy to test", "Clear foundation"],
            risks=["May need significant expansion later"],
            estimated_effort="low"
        ))
        
        return strategies
    
    def _find_similar_successful_implementations(self, context: StrategyContext) -> List[ImplementationStrategy]:
        """Find strategies from similar successful implementations."""
        strategies = []
        
        # This would query a database of successful implementations
        # For now, return empty list
        return strategies
    
    def _analyze_syntax_issues(self, code: str) -> List[Dict[str, Any]]:
        """Analyze code for syntax issues and suggest fixes."""
        fixes = []
        
        try:
            # Try to parse the code
            ast.parse(code)
        except SyntaxError as e:
            # Common syntax fixes
            if "invalid syntax" in str(e):
                # Check for missing colons
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if any(keyword in line for keyword in ['if ', 'for ', 'while ', 'def ', 'class ']):
                        if not line.rstrip().endswith(':'):
                            fixes.append({
                                'type': 'missing_colon',
                                'description': 'Missing colon after statement',
                                'line': i + 1,
                                'confidence': 0.9,
                                'suggested_fix': line.rstrip() + ':',
                                'location': f"Line {i + 1}"
                            })
            
            # Check for unclosed brackets
            open_brackets = code.count('(') + code.count('[') + code.count('{')
            close_brackets = code.count(')') + code.count(']') + code.count('}')
            if open_brackets != close_brackets:
                fixes.append({
                    'type': 'unclosed_bracket',
                    'description': 'Unclosed brackets detected',
                    'confidence': 0.8,
                    'suggested_fix': 'Check and balance all brackets',
                    'location': 'Multiple locations'
                })
        
        return fixes
    
    def _learn_from_strategy_failure(self, strategy: ImplementationStrategy, error_info: Dict[str, Any]):
        """Learn from failed strategy execution."""
        # Record failure pattern
        # This would update the failure database with strategy-specific information
        logger.info(f"Learning from failed strategy: {strategy.strategy_id}")
    
    def _initialize_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize reusable strategy templates."""
        return {
            'error_handling': {
                'description': 'Add comprehensive error handling',
                'snippets': ['try-except blocks', 'logging', 'fallback values']
            },
            'performance': {
                'description': 'Optimize for performance',
                'snippets': ['caching', 'chunking', 'parallel processing']
            },
            'simplification': {
                'description': 'Simplify implementation',
                'snippets': ['reduce complexity', 'extract functions', 'clear naming']
            }
        }
    
    def _initialize_code_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize code pattern recognizers."""
        return {
            'import': re.compile(r'^import\s+(\S+)|^from\s+(\S+)\s+import'),
            'function': re.compile(r'^def\s+(\w+)\s*\('),
            'class': re.compile(r'^class\s+(\w+)'),
            'loop': re.compile(r'^\s*for\s+|^\s*while\s+'),
            'condition': re.compile(r'^\s*if\s+|^\s*elif\s+|^\s*else:')
        }
    
    def _initialize_library_alternatives(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize library alternatives mapping."""
        return {
            'requests': [{
                'name': 'urllib',
                'confidence': 0.8,
                'changes': ['Different API', 'Built-in library'],
                'examples': [
                    """import urllib.request
response = urllib.request.urlopen(url)
data = response.read()"""
                ],
                'benefits': ['No external dependency', 'Built into Python'],
                'risks': ['More verbose API'],
                'effort': 'medium'
            }, {
                'name': 'httpx',
                'confidence': 0.9,
                'changes': ['Similar API', 'Async support'],
                'examples': [
                    """import httpx
response = httpx.get(url)
data = response.text"""
                ],
                'benefits': ['Modern alternative', 'Async support'],
                'risks': ['External dependency'],
                'effort': 'low'
            }],
            'pandas': [{
                'name': 'csv',
                'confidence': 0.6,
                'changes': ['Basic CSV operations only'],
                'examples': [
                    """import csv
with open('file.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)"""
                ],
                'benefits': ['Built-in library', 'Lightweight'],
                'risks': ['Limited functionality'],
                'effort': 'high'
            }],
            'numpy': [{
                'name': 'array',
                'confidence': 0.5,
                'changes': ['Basic array operations only'],
                'examples': [
                    """import array
arr = array.array('d', [1.0, 2.0, 3.0])"""
                ],
                'benefits': ['Built-in module'],
                'risks': ['Very limited functionality'],
                'effort': 'high'
            }]
        }


# Singleton instance
_strategy_generator: Optional[AlternativeStrategyGenerator] = None


def get_strategy_generator() -> AlternativeStrategyGenerator:
    """Get or create singleton strategy generator."""
    global _strategy_generator
    if _strategy_generator is None:
        _strategy_generator = AlternativeStrategyGenerator()
    return _strategy_generator