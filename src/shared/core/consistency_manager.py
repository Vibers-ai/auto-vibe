"""Code Consistency Management System for VIBE."""

import ast
import re
import json
import logging
from typing import Dict, Any, List, Set, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from shared.utils.config import Config
from shared.utils.file_utils import read_text_file, write_json_file, read_json_file
from shared.core.schema import Task, TasksPlan

logger = logging.getLogger(__name__)


class ConsistencyCheckType(Enum):
    """Types of consistency checks."""
    NAMING_CONVENTION = "naming_convention"
    ARCHITECTURE_PATTERN = "architecture_pattern"
    CODING_STYLE = "coding_style"
    IMPORT_STRUCTURE = "import_structure"
    ERROR_HANDLING = "error_handling"
    DOCUMENTATION = "documentation"


@dataclass
class NamingPattern:
    """Represents a naming pattern found in the codebase."""
    pattern_type: str  # variable, function, class, constant, file
    pattern: str       # regex pattern
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.0
    frequency: int = 0


@dataclass
class ArchitecturePattern:
    """Represents an architectural pattern found in the codebase."""
    pattern_name: str
    pattern_type: str  # mvc, component, service, etc.
    structure: Dict[str, Any]
    examples: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class CodingStyleGuideline:
    """Represents a coding style guideline."""
    rule_name: str
    rule_type: str
    pattern: str
    description: str
    examples: Dict[str, str] = field(default_factory=dict)  # good/bad examples
    confidence: float = 0.0


@dataclass
class ConsistencyViolation:
    """Represents a consistency violation found in code."""
    violation_type: ConsistencyCheckType
    file_path: str
    line_number: int
    description: str
    suggested_fix: str
    confidence: float
    severity: str  # low, medium, high, critical


class CodeAnalyzer:
    """Analyzes code patterns and extracts style guidelines."""
    
    def __init__(self):
        self.naming_patterns = {}
        self.architecture_patterns = {}
        self.style_guidelines = {}
    
    def analyze_codebase(self, workspace_path: str) -> Dict[str, Any]:
        """Analyze existing codebase to extract patterns and guidelines."""
        
        workspace = Path(workspace_path)
        if not workspace.exists():
            logger.warning(f"Workspace {workspace_path} does not exist")
            return {}
        
        analysis_result = {
            'naming_patterns': {},
            'architecture_patterns': {},
            'style_guidelines': {},
            'file_structure': {},
            'import_patterns': {}
        }
        
        # Analyze different file types
        python_files = list(workspace.rglob("*.py"))
        js_files = list(workspace.rglob("*.js")) + list(workspace.rglob("*.jsx"))
        ts_files = list(workspace.rglob("*.ts")) + list(workspace.rglob("*.tsx"))
        
        if python_files:
            analysis_result.update(self._analyze_python_files(python_files))
        
        if js_files or ts_files:
            analysis_result.update(self._analyze_javascript_files(js_files + ts_files))
        
        # Analyze project structure
        analysis_result['file_structure'] = self._analyze_file_structure(workspace)
        
        return analysis_result
    
    def _analyze_python_files(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze Python files for patterns."""
        
        naming_patterns = {
            'variables': [],
            'functions': [],
            'classes': [],
            'constants': [],
            'modules': []
        }
        
        style_guidelines = []
        import_patterns = []
        
        for file_path in python_files:
            try:
                content = read_text_file(str(file_path))
                tree = ast.parse(content, filename=str(file_path))
                
                # Extract naming patterns
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        naming_patterns['functions'].append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        naming_patterns['classes'].append(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                if target.id.isupper():
                                    naming_patterns['constants'].append(target.id)
                                else:
                                    naming_patterns['variables'].append(target.id)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            import_patterns.append(f"import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            import_patterns.append(f"from {module} import {alias.name}")
                
                # Analyze style patterns
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    
                    # Check docstring patterns
                    if '"""' in line or "'''" in line:
                        style_guidelines.append({
                            'type': 'docstring',
                            'pattern': 'triple_quotes',
                            'file': str(file_path),
                            'line': i + 1
                        })
                    
                    # Check import grouping
                    if line.startswith('import ') or line.startswith('from '):
                        style_guidelines.append({
                            'type': 'import',
                            'pattern': line,
                            'file': str(file_path),
                            'line': i + 1
                        })
                
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
        
        # Extract patterns from collected data
        extracted_patterns = self._extract_naming_patterns(naming_patterns)
        
        return {
            'naming_patterns': extracted_patterns,
            'style_guidelines': style_guidelines,
            'import_patterns': import_patterns
        }
    
    def _analyze_javascript_files(self, js_files: List[Path]) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript files for patterns."""
        
        naming_patterns = {
            'variables': [],
            'functions': [],
            'classes': [],
            'constants': [],
            'components': []
        }
        
        style_guidelines = []
        
        for file_path in js_files:
            try:
                content = read_text_file(str(file_path))
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    
                    # Function declarations
                    func_match = re.search(r'function\s+(\w+)', line)
                    if func_match:
                        naming_patterns['functions'].append(func_match.group(1))
                    
                    # Arrow functions
                    arrow_match = re.search(r'const\s+(\w+)\s*=.*=>', line)
                    if arrow_match:
                        naming_patterns['functions'].append(arrow_match.group(1))
                    
                    # Class declarations
                    class_match = re.search(r'class\s+(\w+)', line)
                    if class_match:
                        naming_patterns['classes'].append(class_match.group(1))
                    
                    # React components
                    component_match = re.search(r'const\s+(\w+)\s*=.*React|export.*function\s+(\w+)', line)
                    if component_match:
                        component_name = component_match.group(1) or component_match.group(2)
                        if component_name and component_name[0].isupper():
                            naming_patterns['components'].append(component_name)
                    
                    # Variable declarations
                    var_match = re.search(r'(?:const|let|var)\s+(\w+)', line)
                    if var_match:
                        var_name = var_match.group(1)
                        if var_name.isupper():
                            naming_patterns['constants'].append(var_name)
                        else:
                            naming_patterns['variables'].append(var_name)
                    
                    # Style patterns
                    if '//' in line:
                        style_guidelines.append({
                            'type': 'comment',
                            'pattern': 'single_line',
                            'file': str(file_path),
                            'line': i + 1
                        })
                    
                    if '/* ' in line:
                        style_guidelines.append({
                            'type': 'comment',
                            'pattern': 'multi_line',
                            'file': str(file_path),
                            'line': i + 1
                        })
                
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
        
        extracted_patterns = self._extract_naming_patterns(naming_patterns)
        
        return {
            'naming_patterns': extracted_patterns,
            'style_guidelines': style_guidelines
        }
    
    def _extract_naming_patterns(self, raw_patterns: Dict[str, List[str]]) -> Dict[str, NamingPattern]:
        """Extract naming patterns from raw naming data."""
        
        patterns = {}
        
        for pattern_type, names in raw_patterns.items():
            if not names:
                continue
            
            # Analyze naming conventions
            snake_case_count = sum(1 for name in names if '_' in name and name.islower())
            camel_case_count = sum(1 for name in names if 
                                 re.match(r'^[a-z][a-zA-Z0-9]*$', name) and 
                                 any(c.isupper() for c in name))
            pascal_case_count = sum(1 for name in names if 
                                  re.match(r'^[A-Z][a-zA-Z0-9]*$', name))
            upper_case_count = sum(1 for name in names if name.isupper())
            
            total_names = len(names)
            
            # Determine dominant pattern
            dominant_pattern = None
            confidence = 0.0
            
            if snake_case_count / total_names > 0.6:
                dominant_pattern = "snake_case"
                confidence = snake_case_count / total_names
            elif camel_case_count / total_names > 0.6:
                dominant_pattern = "camelCase"
                confidence = camel_case_count / total_names
            elif pascal_case_count / total_names > 0.6:
                dominant_pattern = "PascalCase"
                confidence = pascal_case_count / total_names
            elif upper_case_count / total_names > 0.6:
                dominant_pattern = "UPPER_CASE"
                confidence = upper_case_count / total_names
            
            if dominant_pattern:
                patterns[pattern_type] = NamingPattern(
                    pattern_type=pattern_type,
                    pattern=dominant_pattern,
                    examples=names[:5],  # First 5 examples
                    confidence=confidence,
                    frequency=total_names
                )
        
        return patterns
    
    def _analyze_file_structure(self, workspace: Path) -> Dict[str, Any]:
        """Analyze project file structure patterns."""
        
        structure = {
            'directories': [],
            'file_naming': {},
            'organization_patterns': []
        }
        
        # Collect directory structure
        for item in workspace.rglob("*"):
            if item.is_dir():
                relative_path = item.relative_to(workspace)
                structure['directories'].append(str(relative_path))
        
        # Analyze file naming patterns
        file_extensions = {}
        for item in workspace.rglob("*"):
            if item.is_file():
                ext = item.suffix.lower()
                if ext:
                    if ext not in file_extensions:
                        file_extensions[ext] = []
                    file_extensions[ext].append(item.stem)
        
        structure['file_naming'] = file_extensions
        
        # Detect common patterns
        if any('component' in d.lower() for d in structure['directories']):
            structure['organization_patterns'].append('component_based')
        
        if any('service' in d.lower() for d in structure['directories']):
            structure['organization_patterns'].append('service_layer')
        
        if any('model' in d.lower() for d in structure['directories']):
            structure['organization_patterns'].append('mvc_pattern')
        
        return structure


class ConsistencyChecker:
    """Checks code consistency against established patterns."""
    
    def __init__(self, patterns: Dict[str, Any]):
        self.patterns = patterns
        self.violations = []
    
    def check_file_consistency(self, file_path: str, content: str) -> List[ConsistencyViolation]:
        """Check a file for consistency violations."""
        
        violations = []
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.py':
            violations.extend(self._check_python_consistency(file_path, content))
        elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
            violations.extend(self._check_javascript_consistency(file_path, content))
        
        return violations
    
    def _check_python_consistency(self, file_path: str, content: str) -> List[ConsistencyViolation]:
        """Check Python file consistency."""
        
        violations = []
        
        try:
            tree = ast.parse(content, filename=file_path)
            lines = content.split('\n')
            
            # Check naming conventions
            naming_patterns = self.patterns.get('naming_patterns', {})
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    violations.extend(self._check_naming_violation(
                        file_path, node.lineno, 'function', node.name, naming_patterns
                    ))
                elif isinstance(node, ast.ClassDef):
                    violations.extend(self._check_naming_violation(
                        file_path, node.lineno, 'class', node.name, naming_patterns
                    ))
            
            # Check style consistency
            violations.extend(self._check_python_style_consistency(file_path, lines))
            
        except Exception as e:
            logger.warning(f"Could not check consistency for {file_path}: {e}")
        
        return violations
    
    def _check_javascript_consistency(self, file_path: str, content: str) -> List[ConsistencyViolation]:
        """Check JavaScript/TypeScript file consistency."""
        
        violations = []
        lines = content.split('\n')
        naming_patterns = self.patterns.get('naming_patterns', {})
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check function naming
            func_match = re.search(r'function\s+(\w+)', line)
            if func_match:
                violations.extend(self._check_naming_violation(
                    file_path, i + 1, 'function', func_match.group(1), naming_patterns
                ))
            
            # Check class naming
            class_match = re.search(r'class\s+(\w+)', line)
            if class_match:
                violations.extend(self._check_naming_violation(
                    file_path, i + 1, 'class', class_match.group(1), naming_patterns
                ))
        
        return violations
    
    def _check_naming_violation(
        self, 
        file_path: str, 
        line_number: int, 
        name_type: str, 
        name: str, 
        patterns: Dict[str, NamingPattern]
    ) -> List[ConsistencyViolation]:
        """Check if a name violates established naming patterns."""
        
        violations = []
        
        # Map name types
        pattern_key = {
            'function': 'functions',
            'class': 'classes',
            'variable': 'variables',
            'constant': 'constants'
        }.get(name_type, name_type)
        
        if pattern_key not in patterns:
            return violations
        
        pattern = patterns[pattern_key]
        expected_pattern = pattern.pattern
        
        # Check if name follows pattern
        follows_pattern = False
        
        if expected_pattern == "snake_case":
            follows_pattern = re.match(r'^[a-z][a-z0-9_]*$', name) is not None
        elif expected_pattern == "camelCase":
            follows_pattern = re.match(r'^[a-z][a-zA-Z0-9]*$', name) is not None
        elif expected_pattern == "PascalCase":
            follows_pattern = re.match(r'^[A-Z][a-zA-Z0-9]*$', name) is not None
        elif expected_pattern == "UPPER_CASE":
            follows_pattern = re.match(r'^[A-Z][A-Z0-9_]*$', name) is not None
        
        if not follows_pattern and pattern.confidence > 0.7:
            suggested_name = self._suggest_name_fix(name, expected_pattern)
            
            violations.append(ConsistencyViolation(
                violation_type=ConsistencyCheckType.NAMING_CONVENTION,
                file_path=file_path,
                line_number=line_number,
                description=f"{name_type.capitalize()} '{name}' does not follow project naming convention ({expected_pattern})",
                suggested_fix=f"Consider renaming to '{suggested_name}'",
                confidence=pattern.confidence,
                severity="medium"
            ))
        
        return violations
    
    def _suggest_name_fix(self, name: str, expected_pattern: str) -> str:
        """Suggest a name fix based on expected pattern."""
        
        if expected_pattern == "snake_case":
            # Convert camelCase or PascalCase to snake_case
            return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        elif expected_pattern == "camelCase":
            # Convert snake_case to camelCase
            if '_' in name:
                parts = name.split('_')
                return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])
            return name
        elif expected_pattern == "PascalCase":
            # Convert to PascalCase
            if '_' in name:
                return ''.join(word.capitalize() for word in name.split('_'))
            return name.capitalize()
        elif expected_pattern == "UPPER_CASE":
            return name.upper().replace(' ', '_')
        
        return name
    
    def _check_python_style_consistency(self, file_path: str, lines: List[str]) -> List[ConsistencyViolation]:
        """Check Python style consistency."""
        
        violations = []
        style_guidelines = self.patterns.get('style_guidelines', [])
        
        # Check import organization
        import_lines = [(i, line) for i, line in enumerate(lines) if line.strip().startswith(('import ', 'from '))]
        
        if import_lines:
            # Check if imports are grouped (stdlib, third-party, local)
            # This is a simplified check
            has_blank_lines = any(
                i < len(lines) - 1 and not lines[i + 1].strip() 
                for i, _ in import_lines[:-1]
            )
            
            if not has_blank_lines and len(import_lines) > 3:
                violations.append(ConsistencyViolation(
                    violation_type=ConsistencyCheckType.IMPORT_STRUCTURE,
                    file_path=file_path,
                    line_number=import_lines[0][0] + 1,
                    description="Imports should be grouped with blank lines",
                    suggested_fix="Add blank lines between import groups (stdlib, third-party, local)",
                    confidence=0.8,
                    severity="low"
                ))
        
        return violations


class ConsistencyManager:
    """Main consistency management system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.analyzer = CodeAnalyzer()
        self.patterns = {}
        self.consistency_cache = {}
    
    def initialize_from_workspace(self, workspace_path: str) -> None:
        """Initialize consistency patterns from existing workspace."""
        
        logger.info(f"Analyzing workspace for consistency patterns: {workspace_path}")
        
        self.patterns = self.analyzer.analyze_codebase(workspace_path)
        
        # Save patterns for future use
        patterns_file = Path(workspace_path) / ".vibe" / "consistency_patterns.json"
        patterns_file.parent.mkdir(exist_ok=True)
        
        # Convert patterns to JSON-serializable format
        serializable_patterns = self._serialize_patterns(self.patterns)
        write_json_file(str(patterns_file), serializable_patterns)
        
        logger.info(f"Consistency patterns saved to {patterns_file}")
    
    def load_patterns(self, workspace_path: str) -> bool:
        """Load existing consistency patterns."""
        
        patterns_file = Path(workspace_path) / ".vibe" / "consistency_patterns.json"
        
        if patterns_file.exists():
            try:
                self.patterns = read_json_file(str(patterns_file))
                logger.info(f"Loaded consistency patterns from {patterns_file}")
                return True
            except Exception as e:
                logger.warning(f"Could not load patterns: {e}")
        
        return False
    
    def check_task_consistency(self, task: Task, workspace_path: str) -> Dict[str, Any]:
        """Check if a task's generated code maintains consistency."""
        
        if not self.patterns:
            if not self.load_patterns(workspace_path):
                self.initialize_from_workspace(workspace_path)
        
        checker = ConsistencyChecker(self.patterns)
        all_violations = []
        
        # Check each file that will be created/modified
        for file_path in task.files_to_create_or_modify:
            full_path = Path(workspace_path) / file_path
            
            if full_path.exists():
                try:
                    content = read_text_file(str(full_path))
                    violations = checker.check_file_consistency(str(full_path), content)
                    all_violations.extend(violations)
                except Exception as e:
                    logger.warning(f"Could not check consistency for {full_path}: {e}")
        
        # Generate consistency report
        report = self._generate_consistency_report(all_violations, task)
        
        return report
    
    def generate_consistency_prompt(self, task: Task, workspace_path: str) -> str:
        """Generate prompt instructions for maintaining consistency."""
        
        if not self.patterns:
            if not self.load_patterns(workspace_path):
                self.initialize_from_workspace(workspace_path)
        
        prompt_parts = []
        
        # Naming conventions
        naming_patterns = self.patterns.get('naming_patterns', {})
        if naming_patterns:
            prompt_parts.append("**NAMING CONVENTIONS:**")
            for name_type, pattern in naming_patterns.items():
                if hasattr(pattern, 'pattern'):
                    prompt_parts.append(f"- {name_type}: Use {pattern.pattern} (e.g., {', '.join(pattern.examples[:3])})")
                else:
                    prompt_parts.append(f"- {name_type}: Follow established patterns")
        
        # Architecture patterns
        file_structure = self.patterns.get('file_structure', {})
        if file_structure.get('organization_patterns'):
            prompt_parts.append("\n**ARCHITECTURE PATTERNS:**")
            for pattern in file_structure['organization_patterns']:
                prompt_parts.append(f"- Follow {pattern} organization")
        
        # Style guidelines
        style_guidelines = self.patterns.get('style_guidelines', [])
        if style_guidelines:
            prompt_parts.append("\n**STYLE GUIDELINES:**")
            
            # Extract common style patterns
            docstring_patterns = [g for g in style_guidelines if g.get('type') == 'docstring']
            if docstring_patterns:
                prompt_parts.append("- Use triple quotes for docstrings")
            
            comment_patterns = [g for g in style_guidelines if g.get('type') == 'comment']
            if comment_patterns:
                prompt_parts.append("- Follow established commenting patterns")
        
        # Import patterns
        import_patterns = self.patterns.get('import_patterns', [])
        if import_patterns:
            prompt_parts.append("\n**IMPORT ORGANIZATION:**")
            prompt_parts.append("- Group imports: stdlib, third-party, local")
            prompt_parts.append("- Use absolute imports when possible")
        
        if not prompt_parts:
            prompt_parts.append("**CONSISTENCY:** Follow general coding best practices")
        
        return "\n".join(prompt_parts)
    
    def _serialize_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Convert patterns to JSON-serializable format."""
        
        serializable = {}
        
        for key, value in patterns.items():
            if key == 'naming_patterns':
                serializable[key] = {}
                for pattern_key, pattern in value.items():
                    if hasattr(pattern, '__dict__'):
                        serializable[key][pattern_key] = pattern.__dict__
                    else:
                        serializable[key][pattern_key] = pattern
            else:
                serializable[key] = value
        
        return serializable
    
    def _generate_consistency_report(self, violations: List[ConsistencyViolation], task: Task) -> Dict[str, Any]:
        """Generate a comprehensive consistency report."""
        
        # Group violations by type and severity
        by_type = {}
        by_severity = {}
        
        for violation in violations:
            violation_type = violation.violation_type.value
            severity = violation.severity
            
            if violation_type not in by_type:
                by_type[violation_type] = []
            by_type[violation_type].append(violation)
            
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(violation)
        
        # Calculate overall consistency score
        total_violations = len(violations)
        critical_violations = len(by_severity.get('critical', []))
        high_violations = len(by_severity.get('high', []))
        
        # Score calculation (0-100)
        consistency_score = 100
        if total_violations > 0:
            penalty = critical_violations * 20 + high_violations * 10 + (total_violations - critical_violations - high_violations) * 5
            consistency_score = max(0, 100 - penalty)
        
        return {
            'task_id': task.id,
            'consistency_score': consistency_score,
            'total_violations': total_violations,
            'violations_by_type': {k: len(v) for k, v in by_type.items()},
            'violations_by_severity': {k: len(v) for k, v in by_severity.items()},
            'detailed_violations': [
                {
                    'type': v.violation_type.value,
                    'file': v.file_path,
                    'line': v.line_number,
                    'description': v.description,
                    'suggestion': v.suggested_fix,
                    'severity': v.severity,
                    'confidence': v.confidence
                }
                for v in violations
            ],
            'recommendations': self._generate_recommendations(violations)
        }
    
    def _generate_recommendations(self, violations: List[ConsistencyViolation]) -> List[str]:
        """Generate actionable recommendations based on violations."""
        
        recommendations = []
        
        # Group by type to generate specific recommendations
        naming_violations = [v for v in violations if v.violation_type == ConsistencyCheckType.NAMING_CONVENTION]
        if naming_violations:
            recommendations.append("Review and standardize naming conventions across the project")
        
        style_violations = [v for v in violations if v.violation_type == ConsistencyCheckType.CODING_STYLE]
        if style_violations:
            recommendations.append("Apply consistent coding style guidelines")
        
        import_violations = [v for v in violations if v.violation_type == ConsistencyCheckType.IMPORT_STRUCTURE]
        if import_violations:
            recommendations.append("Organize imports following established patterns")
        
        if len(violations) > 10:
            recommendations.append("Consider setting up automated linting and formatting tools")
        
        return recommendations