"""Real-time Syntax Validation System for VIBE

This module provides real-time syntax validation for generated code across
multiple programming languages, detecting errors before execution.
"""

import ast
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import tempfile

import esprima  # For JavaScript parsing
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JSX = "jsx"
    TSX = "tsx"
    JSON = "json"
    YAML = "yaml"
    SQL = "sql"


@dataclass
class SyntaxError:
    """Represents a syntax error found in code."""
    line: int
    column: int
    message: str
    level: ValidationLevel
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class ImportValidation:
    """Import statement validation result."""
    import_statement: str
    is_valid: bool
    module_exists: bool
    error_message: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result for a code file."""
    file_path: str
    language: LanguageType
    is_valid: bool
    errors: List[SyntaxError] = field(default_factory=list)
    warnings: List[SyntaxError] = field(default_factory=list)
    import_issues: List[ImportValidation] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)


class RealTimeSyntaxValidator:
    """Real-time syntax validation for multiple programming languages."""
    
    def __init__(self):
        self.validators = {
            LanguageType.PYTHON: self._validate_python,
            LanguageType.JAVASCRIPT: self._validate_javascript,
            LanguageType.JSX: self._validate_jsx,
            LanguageType.TYPESCRIPT: self._validate_typescript,
            LanguageType.TSX: self._validate_tsx,
            LanguageType.JSON: self._validate_json,
            LanguageType.YAML: self._validate_yaml,
            LanguageType.SQL: self._validate_sql
        }
        
        # Cache for module existence checks
        self.module_cache: Dict[str, bool] = {}
        
        # Common import patterns
        self.import_patterns = {
            'python': re.compile(r'^(?:from\s+(\S+)\s+)?import\s+(.+)$', re.MULTILINE),
            'javascript': re.compile(r'^(?:import\s+(?:{[^}]+}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"]([^\'"]+)[\'"]|require\s*\([\'"]([^\'"]+)[\'"]\))', re.MULTILINE),
            'typescript': re.compile(r'^(?:import\s+(?:{[^}]+}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"]([^\'"]+)[\'"])', re.MULTILINE)
        }
    
    def validate_code(self, code: str, language: LanguageType, file_path: str = "unknown") -> ValidationResult:
        """Validate code for syntax errors and issues."""
        if language not in self.validators:
            return ValidationResult(
                file_path=file_path,
                language=language,
                is_valid=True,
                warnings=[SyntaxError(
                    line=0,
                    column=0,
                    message=f"No validator available for {language.value}",
                    level=ValidationLevel.WARNING
                )]
            )
        
        try:
            return self.validators[language](code, file_path)
        except Exception as e:
            logger.error(f"Validation failed for {file_path}: {e}")
            return ValidationResult(
                file_path=file_path,
                language=language,
                is_valid=False,
                errors=[SyntaxError(
                    line=0,
                    column=0,
                    message=f"Validation error: {str(e)}",
                    level=ValidationLevel.ERROR
                )]
            )
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """Validate a file based on its extension."""
        path = Path(file_path)
        if not path.exists():
            return ValidationResult(
                file_path=file_path,
                language=LanguageType.PYTHON,
                is_valid=False,
                errors=[SyntaxError(
                    line=0,
                    column=0,
                    message=f"File not found: {file_path}",
                    level=ValidationLevel.ERROR
                )]
            )
        
        # Determine language from extension
        language = self._detect_language(path)
        if not language:
            return ValidationResult(
                file_path=file_path,
                language=LanguageType.PYTHON,
                is_valid=True,
                warnings=[SyntaxError(
                    line=0,
                    column=0,
                    message=f"Unknown file type: {path.suffix}",
                    level=ValidationLevel.WARNING
                )]
            )
        
        # Read and validate
        try:
            code = path.read_text(encoding='utf-8')
            return self.validate_code(code, language, file_path)
        except Exception as e:
            return ValidationResult(
                file_path=file_path,
                language=language,
                is_valid=False,
                errors=[SyntaxError(
                    line=0,
                    column=0,
                    message=f"Error reading file: {str(e)}",
                    level=ValidationLevel.ERROR
                )]
            )
    
    def _detect_language(self, path: Path) -> Optional[LanguageType]:
        """Detect language from file extension."""
        ext_map = {
            '.py': LanguageType.PYTHON,
            '.js': LanguageType.JAVASCRIPT,
            '.jsx': LanguageType.JSX,
            '.ts': LanguageType.TYPESCRIPT,
            '.tsx': LanguageType.TSX,
            '.json': LanguageType.JSON,
            '.yaml': LanguageType.YAML,
            '.yml': LanguageType.YAML,
            '.sql': LanguageType.SQL
        }
        return ext_map.get(path.suffix.lower())
    
    def _validate_python(self, code: str, file_path: str) -> ValidationResult:
        """Validate Python code."""
        result = ValidationResult(
            file_path=file_path,
            language=LanguageType.PYTHON,
            is_valid=True
        )
        
        # Parse AST
        try:
            tree = ast.parse(code, filename=file_path)
            result.metrics['ast_nodes'] = len(list(ast.walk(tree)))
            
            # Validate imports
            imports = self._extract_python_imports(tree)
            for imp in imports:
                validation = self._validate_python_import(imp)
                if not validation.is_valid:
                    result.import_issues.append(validation)
                    if not validation.module_exists:
                        result.warnings.append(SyntaxError(
                            line=imp.lineno,
                            column=imp.col_offset,
                            message=f"Module '{validation.import_statement}' may not be available",
                            level=ValidationLevel.WARNING,
                            suggestion=validation.suggestion
                        ))
            
            # Check for common issues
            self._check_python_common_issues(tree, code, result)
            
        except SyntaxError as e:
            result.is_valid = False
            result.errors.append(SyntaxError(
                line=e.lineno or 0,
                column=e.offset or 0,
                message=e.msg,
                level=ValidationLevel.ERROR,
                code_snippet=self._get_code_snippet(code, e.lineno) if e.lineno else None
            ))
        
        return result
    
    def _validate_javascript(self, code: str, file_path: str) -> ValidationResult:
        """Validate JavaScript code."""
        result = ValidationResult(
            file_path=file_path,
            language=LanguageType.JAVASCRIPT,
            is_valid=True
        )
        
        try:
            # Parse with esprima
            tree = esprima.parseScript(code, loc=True, tolerant=True)
            
            # Check for errors
            if hasattr(tree, 'errors') and tree.errors:
                result.is_valid = False
                for error in tree.errors:
                    result.errors.append(SyntaxError(
                        line=error.lineNumber if hasattr(error, 'lineNumber') else 0,
                        column=error.column if hasattr(error, 'column') else 0,
                        message=str(error),
                        level=ValidationLevel.ERROR
                    ))
            
            # Validate imports
            self._validate_js_imports(code, result)
            
            # Check common issues
            self._check_js_common_issues(code, result)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(SyntaxError(
                line=0,
                column=0,
                message=f"JavaScript parsing error: {str(e)}",
                level=ValidationLevel.ERROR
            ))
        
        return result
    
    def _validate_jsx(self, code: str, file_path: str) -> ValidationResult:
        """Validate JSX code."""
        result = ValidationResult(
            file_path=file_path,
            language=LanguageType.JSX,
            is_valid=True
        )
        
        try:
            # Parse with esprima (JSX support)
            tree = esprima.parseScript(code, jsx=True, loc=True, tolerant=True)
            
            if hasattr(tree, 'errors') and tree.errors:
                result.is_valid = False
                for error in tree.errors:
                    result.errors.append(SyntaxError(
                        line=error.lineNumber if hasattr(error, 'lineNumber') else 0,
                        column=error.column if hasattr(error, 'column') else 0,
                        message=str(error),
                        level=ValidationLevel.ERROR
                    ))
            
            # Validate React-specific patterns
            self._validate_react_patterns(code, result)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(SyntaxError(
                line=0,
                column=0,
                message=f"JSX parsing error: {str(e)}",
                level=ValidationLevel.ERROR
            ))
        
        return result
    
    def _validate_typescript(self, code: str, file_path: str) -> ValidationResult:
        """Validate TypeScript code using tsc."""
        result = ValidationResult(
            file_path=file_path,
            language=LanguageType.TYPESCRIPT,
            is_valid=True
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Run TypeScript compiler
            process = subprocess.run(
                ['npx', 'tsc', '--noEmit', '--skipLibCheck', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if process.returncode != 0:
                result.is_valid = False
                # Parse tsc output
                self._parse_tsc_errors(process.stdout + process.stderr, result, temp_path)
            
            # Validate imports
            self._validate_ts_imports(code, result)
            
        except subprocess.TimeoutExpired:
            result.warnings.append(SyntaxError(
                line=0,
                column=0,
                message="TypeScript validation timed out",
                level=ValidationLevel.WARNING
            ))
        except Exception as e:
            result.warnings.append(SyntaxError(
                line=0,
                column=0,
                message=f"TypeScript validation unavailable: {str(e)}",
                level=ValidationLevel.WARNING
            ))
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
        
        return result
    
    def _validate_tsx(self, code: str, file_path: str) -> ValidationResult:
        """Validate TSX code."""
        result = ValidationResult(
            file_path=file_path,
            language=LanguageType.TSX,
            is_valid=True
        )
        
        # Similar to TypeScript but with .tsx extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsx', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            process = subprocess.run(
                ['npx', 'tsc', '--noEmit', '--skipLibCheck', '--jsx', 'react', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if process.returncode != 0:
                result.is_valid = False
                self._parse_tsc_errors(process.stdout + process.stderr, result, temp_path)
            
            # Validate React patterns
            self._validate_react_patterns(code, result)
            
        except Exception as e:
            result.warnings.append(SyntaxError(
                line=0,
                column=0,
                message=f"TSX validation unavailable: {str(e)}",
                level=ValidationLevel.WARNING
            ))
        finally:
            Path(temp_path).unlink(missing_ok=True)
        
        return result
    
    def _validate_json(self, code: str, file_path: str) -> ValidationResult:
        """Validate JSON code."""
        result = ValidationResult(
            file_path=file_path,
            language=LanguageType.JSON,
            is_valid=True
        )
        
        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            result.is_valid = False
            result.errors.append(SyntaxError(
                line=e.lineno,
                column=e.colno,
                message=e.msg,
                level=ValidationLevel.ERROR,
                code_snippet=self._get_code_snippet(code, e.lineno)
            ))
        
        return result
    
    def _validate_yaml(self, code: str, file_path: str) -> ValidationResult:
        """Validate YAML code."""
        result = ValidationResult(
            file_path=file_path,
            language=LanguageType.YAML,
            is_valid=True
        )
        
        try:
            import yaml
            yaml.safe_load(code)
        except yaml.YAMLError as e:
            result.is_valid = False
            line = 0
            if hasattr(e, 'problem_mark'):
                line = e.problem_mark.line + 1
            
            result.errors.append(SyntaxError(
                line=line,
                column=0,
                message=str(e),
                level=ValidationLevel.ERROR
            ))
        except ImportError:
            result.warnings.append(SyntaxError(
                line=0,
                column=0,
                message="YAML validation unavailable (pyyaml not installed)",
                level=ValidationLevel.WARNING
            ))
        
        return result
    
    def _validate_sql(self, code: str, file_path: str) -> ValidationResult:
        """Basic SQL validation."""
        result = ValidationResult(
            file_path=file_path,
            language=LanguageType.SQL,
            is_valid=True
        )
        
        # Basic SQL keyword validation
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        code_upper = code.upper()
        
        has_sql_keyword = any(keyword in code_upper for keyword in sql_keywords)
        if not has_sql_keyword and len(code.strip()) > 0:
            result.warnings.append(SyntaxError(
                line=1,
                column=0,
                message="No SQL keywords found",
                level=ValidationLevel.WARNING
            ))
        
        # Check for common SQL syntax errors
        if code.count('(') != code.count(')'):
            result.errors.append(SyntaxError(
                line=0,
                column=0,
                message="Mismatched parentheses",
                level=ValidationLevel.ERROR
            ))
            result.is_valid = False
        
        if code.count("'") % 2 != 0:
            result.errors.append(SyntaxError(
                line=0,
                column=0,
                message="Unclosed string literal",
                level=ValidationLevel.ERROR
            ))
            result.is_valid = False
        
        return result
    
    # Helper methods
    
    def _extract_python_imports(self, tree: ast.AST) -> List[ast.Import]:
        """Extract import statements from Python AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
        return imports
    
    def _validate_python_import(self, node: ast.Import) -> ImportValidation:
        """Validate a Python import statement."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                return self._check_python_module_exists(module_name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            return self._check_python_module_exists(module_name)
        
        return ImportValidation(
            import_statement="",
            is_valid=True,
            module_exists=True
        )
    
    def _check_python_module_exists(self, module_name: str) -> ImportValidation:
        """Check if a Python module exists."""
        if not module_name:
            return ImportValidation(
                import_statement=module_name,
                is_valid=True,
                module_exists=True
            )
        
        # Check cache
        if module_name in self.module_cache:
            exists = self.module_cache[module_name]
            return ImportValidation(
                import_statement=module_name,
                is_valid=exists,
                module_exists=exists,
                error_message=None if exists else f"Module '{module_name}' not found"
            )
        
        # Common standard library modules
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'collections', 'itertools',
            'functools', 'pathlib', 'typing', 'logging', 're', 'math',
            'random', 'string', 'time', 'subprocess', 'asyncio', 'unittest'
        }
        
        if module_name in stdlib_modules or module_name.split('.')[0] in stdlib_modules:
            self.module_cache[module_name] = True
            return ImportValidation(
                import_statement=module_name,
                is_valid=True,
                module_exists=True
            )
        
        # For now, warn about non-stdlib modules
        return ImportValidation(
            import_statement=module_name,
            is_valid=True,
            module_exists=False,
            error_message=f"External module '{module_name}' may need to be installed",
            suggestion=f"Run: pip install {module_name.split('.')[0]}"
        )
    
    def _check_python_common_issues(self, tree: ast.AST, code: str, result: ValidationResult):
        """Check for common Python issues."""
        # Check for undefined variables (basic)
        defined_names = set()
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
        
        # Python builtins
        builtins = {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'bool', 'True', 'False', 'None'}
        undefined = used_names - defined_names - builtins
        
        # Filter out common false positives
        undefined = {name for name in undefined if not name.startswith('__') and name not in ['self', 'cls']}
        
        for name in undefined:
            result.warnings.append(SyntaxError(
                line=0,
                column=0,
                message=f"Possible undefined variable: '{name}'",
                level=ValidationLevel.WARNING
            ))
    
    def _validate_js_imports(self, code: str, result: ValidationResult):
        """Validate JavaScript imports."""
        import_pattern = self.import_patterns['javascript']
        
        for match in import_pattern.finditer(code):
            module = match.group(1) or match.group(2)
            if module:
                if module.startswith('.'):
                    # Relative import
                    continue
                elif module in ['react', 'react-dom', 'vue', 'angular']:
                    # Common frameworks
                    continue
                else:
                    result.warnings.append(SyntaxError(
                        line=code[:match.start()].count('\n') + 1,
                        column=0,
                        message=f"External module '{module}' may need to be installed",
                        level=ValidationLevel.WARNING,
                        suggestion=f"Run: npm install {module}"
                    ))
    
    def _validate_ts_imports(self, code: str, result: ValidationResult):
        """Validate TypeScript imports."""
        import_pattern = self.import_patterns['typescript']
        
        for match in import_pattern.finditer(code):
            module = match.group(1)
            if module and not module.startswith('.'):
                if module.startswith('@types/'):
                    result.warnings.append(SyntaxError(
                        line=code[:match.start()].count('\n') + 1,
                        column=0,
                        message=f"Type definition '{module}' may need to be installed",
                        level=ValidationLevel.WARNING,
                        suggestion=f"Run: npm install -D {module}"
                    ))
    
    def _check_js_common_issues(self, code: str, result: ValidationResult):
        """Check for common JavaScript issues."""
        # Check for console.log in production code
        if 'console.log' in code:
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if 'console.log' in line and not line.strip().startswith('//'):
                    result.warnings.append(SyntaxError(
                        line=i + 1,
                        column=line.index('console.log'),
                        message="console.log should be removed in production",
                        level=ValidationLevel.WARNING,
                        suggestion="Use a proper logging library or remove"
                    ))
        
        # Check for var usage
        if re.search(r'\bvar\s+\w+', code):
            result.warnings.append(SyntaxError(
                line=0,
                column=0,
                message="'var' keyword is deprecated, use 'let' or 'const'",
                level=ValidationLevel.WARNING
            ))
    
    def _validate_react_patterns(self, code: str, result: ValidationResult):
        """Validate React-specific patterns."""
        # Check for React import
        if 'React' in code and 'import React' not in code:
            result.warnings.append(SyntaxError(
                line=1,
                column=0,
                message="React is used but not imported",
                level=ValidationLevel.WARNING,
                suggestion="Add: import React from 'react'"
            ))
        
        # Check for hooks in class components
        if 'extends React.Component' in code or 'extends Component' in code:
            if re.search(r'\buse[A-Z]\w*\(', code):
                result.errors.append(SyntaxError(
                    line=0,
                    column=0,
                    message="React Hooks cannot be used in class components",
                    level=ValidationLevel.ERROR,
                    suggestion="Convert to functional component or use class methods"
                ))
    
    def _parse_tsc_errors(self, output: str, result: ValidationResult, temp_path: str):
        """Parse TypeScript compiler errors."""
        # TSC error format: file(line,col): error TS####: message
        error_pattern = re.compile(r'(.+?)\((\d+),(\d+)\):\s+error\s+TS\d+:\s+(.+)')
        
        for line in output.split('\n'):
            match = error_pattern.match(line)
            if match:
                file_path, line_num, col_num, message = match.groups()
                if temp_path in file_path:
                    result.errors.append(SyntaxError(
                        line=int(line_num),
                        column=int(col_num),
                        message=message,
                        level=ValidationLevel.ERROR
                    ))
    
    def _get_code_snippet(self, code: str, line_num: Optional[int], context: int = 2) -> Optional[str]:
        """Get code snippet around error line."""
        if not line_num:
            return None
        
        lines = code.split('\n')
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)
        
        snippet_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_num - 1 else "    "
            snippet_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")
        
        return '\n'.join(snippet_lines)


# Singleton instance
realtime_syntax_validator = RealTimeSyntaxValidator()