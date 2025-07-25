"""Technology Stack Consistency Enforcer."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .workspace_analyzer import TechnologyStack, workspace_analyzer
from .enhanced_logger import get_logger, LogCategory

logger = logging.getLogger(__name__)


class StackViolationType(Enum):
    """Types of technology stack violations."""
    FILE_EXTENSION_MISMATCH = "file_extension_mismatch"
    FRAMEWORK_CONFLICT = "framework_conflict"
    DEPENDENCY_INCONSISTENCY = "dependency_inconsistency"
    CONFIG_MISMATCH = "config_mismatch"
    CONVENTION_VIOLATION = "convention_violation"


@dataclass
class StackViolation:
    """Represents a technology stack consistency violation."""
    violation_type: StackViolationType
    file_path: str
    expected: str
    actual: str
    severity: str  # "error", "warning", "info"
    description: str
    fix_suggestion: str


class TechStackEnforcer:
    """Enforces technology stack consistency across the project."""
    
    # File extension rules per technology stack
    STACK_EXTENSIONS = {
        TechnologyStack.NEXTJS_TS: {
            'component_files': ['.tsx'],
            'utility_files': ['.ts'],
            'config_files': ['.ts', '.js', '.mjs'],
            'test_files': ['.test.ts', '.test.tsx', '.spec.ts', '.spec.tsx']
        },
        TechnologyStack.NEXTJS_JS: {
            'component_files': ['.jsx'],
            'utility_files': ['.js'],
            'config_files': ['.js', '.mjs'],
            'test_files': ['.test.js', '.test.jsx', '.spec.js', '.spec.jsx']
        },
        TechnologyStack.REACT_TS: {
            'component_files': ['.tsx'],
            'utility_files': ['.ts'],
            'config_files': ['.ts', '.js'],
            'test_files': ['.test.ts', '.test.tsx', '.spec.ts', '.spec.tsx']
        },
        TechnologyStack.REACT_JS: {
            'component_files': ['.jsx'],
            'utility_files': ['.js'],
            'config_files': ['.js'],
            'test_files': ['.test.js', '.test.jsx', '.spec.js', '.spec.jsx']
        },
        TechnologyStack.EXPRESS_TS: {
            'route_files': ['.ts'],
            'controller_files': ['.ts'],
            'model_files': ['.ts'],
            'utility_files': ['.ts'],
            'config_files': ['.ts', '.js'],
            'test_files': ['.test.ts', '.spec.ts']
        },
        TechnologyStack.EXPRESS_JS: {
            'route_files': ['.js'],
            'controller_files': ['.js'],
            'model_files': ['.js'],
            'utility_files': ['.js'],
            'config_files': ['.js'],
            'test_files': ['.test.js', '.spec.js']
        }
    }
    
    # Directory naming conventions per stack
    STACK_CONVENTIONS = {
        TechnologyStack.NEXTJS_TS: {
            'component_dirs': ['components', 'app', 'src/app', 'src/components'],
            'util_dirs': ['lib', 'utils', 'src/lib', 'src/utils'],
            'style_dirs': ['styles', 'src/styles'],
            'config_location': 'root'
        },
        TechnologyStack.NEXTJS_JS: {
            'component_dirs': ['components', 'app', 'src/app', 'src/components'],
            'util_dirs': ['lib', 'utils', 'src/lib', 'src/utils'],
            'style_dirs': ['styles', 'src/styles'],
            'config_location': 'root'
        },
        TechnologyStack.REACT_TS: {
            'component_dirs': ['src/components', 'components'],
            'util_dirs': ['src/utils', 'src/lib', 'utils', 'lib'],
            'hook_dirs': ['src/hooks', 'hooks'],
            'config_location': 'root'
        },
        TechnologyStack.REACT_JS: {
            'component_dirs': ['src/components', 'components'],
            'util_dirs': ['src/utils', 'src/lib', 'utils', 'lib'],
            'hook_dirs': ['src/hooks', 'hooks'],
            'config_location': 'root'
        }
    }
    
    def __init__(self):
        self.logger = get_logger("tech_stack_enforcer")
    
    def validate_file_against_stack(self, file_path: str, workspace_path: str, 
                                  primary_stack: TechnologyStack) -> List[StackViolation]:
        """Validate a single file against the primary technology stack."""
        violations = []
        
        # Get relative path from workspace
        try:
            rel_path = os.path.relpath(file_path, workspace_path)
        except ValueError:
            # File is outside workspace
            return violations
        
        # Determine file type and expected extensions
        file_type = self._classify_file(rel_path)
        if not file_type:
            return violations  # Skip unclassified files
        
        # Check file extension consistency
        expected_extensions = self.STACK_EXTENSIONS.get(primary_stack, {}).get(file_type, [])
        if expected_extensions:
            actual_extension = self._get_file_extension(file_path)
            
            if actual_extension not in expected_extensions:
                violations.append(StackViolation(
                    violation_type=StackViolationType.FILE_EXTENSION_MISMATCH,
                    file_path=rel_path,
                    expected=f"One of: {', '.join(expected_extensions)}",
                    actual=actual_extension,
                    severity="warning",
                    description=f"File extension '{actual_extension}' doesn't match {primary_stack.value} conventions",
                    fix_suggestion=f"Rename to use one of: {', '.join(expected_extensions)}"
                ))
        
        # Check directory placement
        directory_violations = self._check_directory_placement(rel_path, file_type, primary_stack)
        violations.extend(directory_violations)
        
        return violations
    
    def validate_project_consistency(self, workspace_path: str) -> Dict[str, Any]:
        """Validate entire project for technology stack consistency."""
        
        self.logger.info("Starting project-wide technology stack validation", 
                        LogCategory.STRUCTURE_VALIDATION)
        
        # Analyze workspace
        workspace_structure = workspace_analyzer.analyze_workspace(workspace_path)
        
        if not workspace_structure.technology_stacks:
            return {
                'primary_stack': None,
                'violations': [],
                'summary': 'No technology stack detected',
                'recommendations': ['Establish a clear technology stack for the project']
            }
        
        primary_stack = workspace_structure.technology_stacks[0].stack
        violations = []
        
        # Scan all relevant files
        for root, dirs, files in os.walk(workspace_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in [
                'node_modules', '.git', 'dist', 'build', '__pycache__', 
                '.next', 'coverage', '.vscode', '.idea'
            ]]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_violations = self.validate_file_against_stack(file_path, workspace_path, primary_stack)
                violations.extend(file_violations)
        
        # Check for framework conflicts
        framework_violations = self._check_framework_conflicts(workspace_path, workspace_structure)
        violations.extend(framework_violations)
        
        # Generate summary
        summary = self._generate_consistency_summary(violations, primary_stack)
        
        self.logger.info(f"Technology stack validation complete. Found {len(violations)} violations.",
                        LogCategory.STRUCTURE_VALIDATION,
                        metadata={'primary_stack': primary_stack.value, 'violation_count': len(violations)})
        
        return {
            'primary_stack': primary_stack.value,
            'violations': [self._violation_to_dict(v) for v in violations],
            'summary': summary,
            'recommendations': self._generate_recommendations(violations, primary_stack)
        }
    
    def _classify_file(self, file_path: str) -> Optional[str]:
        """Classify file type based on path and name."""
        path_lower = file_path.lower()
        file_name = os.path.basename(file_path).lower()
        
        # Component files
        if any(comp_dir in path_lower for comp_dir in ['component', 'page', 'layout']):
            return 'component_files'
        
        # Utility/lib files
        if any(util_dir in path_lower for util_dir in ['util', 'lib', 'helper']):
            return 'utility_files'
        
        # Route/controller files
        if any(route_dir in path_lower for route_dir in ['route', 'controller', 'api']):
            return 'route_files'
        
        # Model files
        if any(model_dir in path_lower for model_dir in ['model', 'schema', 'type']):
            return 'model_files'
        
        # Config files
        if any(config_name in file_name for config_name in [
            'config', 'next.config', 'vite.config', 'webpack.config',
            'tailwind.config', 'jest.config'
        ]):
            return 'config_files'
        
        # Test files
        if any(test_name in file_name for test_name in ['test', 'spec']):
            return 'test_files'
        
        # Default to utility for .ts/.js files in src
        if 'src' in path_lower and file_name.endswith(('.ts', '.js', '.tsx', '.jsx')):
            return 'utility_files'
        
        return None
    
    def _get_file_extension(self, file_path: str) -> str:
        """Get the full file extension (including compound extensions like .test.ts)."""
        file_name = os.path.basename(file_path)
        
        # Check for compound extensions first
        compound_extensions = ['.test.ts', '.test.tsx', '.test.js', '.test.jsx',
                             '.spec.ts', '.spec.tsx', '.spec.js', '.spec.jsx',
                             '.config.ts', '.config.js']
        
        for comp_ext in compound_extensions:
            if file_name.endswith(comp_ext):
                return comp_ext
        
        # Return simple extension
        return os.path.splitext(file_name)[1]
    
    def _check_directory_placement(self, file_path: str, file_type: str, 
                                 primary_stack: TechnologyStack) -> List[StackViolation]:
        """Check if file is placed in appropriate directory."""
        violations = []
        
        conventions = self.STACK_CONVENTIONS.get(primary_stack, {})
        expected_dirs = conventions.get(f"{file_type.replace('_files', '_dirs')}", [])
        
        if not expected_dirs:
            return violations
        
        # Check if file is in any of the expected directories
        file_dir = os.path.dirname(file_path)
        
        is_in_expected_dir = any(
            expected_dir in file_path or file_path.startswith(expected_dir + '/')
            for expected_dir in expected_dirs
        )
        
        if not is_in_expected_dir and file_type != 'config_files':  # Config files can be flexible
            violations.append(StackViolation(
                violation_type=StackViolationType.CONVENTION_VIOLATION,
                file_path=file_path,
                expected=f"One of: {', '.join(expected_dirs)}",
                actual=file_dir or 'root',
                severity="info",
                description=f"{file_type.replace('_', ' ').title()} should be in standard directories",
                fix_suggestion=f"Move to one of: {', '.join(expected_dirs)}"
            ))
        
        return violations
    
    def _check_framework_conflicts(self, workspace_path: str, workspace_structure) -> List[StackViolation]:
        """Check for conflicting frameworks or technologies."""
        violations = []
        
        # Check for Next.js routing conflicts (App Router vs Pages Router)
        has_app_router = os.path.exists(os.path.join(workspace_path, 'app')) or \
                        os.path.exists(os.path.join(workspace_path, 'src', 'app'))
        has_pages_router = os.path.exists(os.path.join(workspace_path, 'pages'))
        
        if has_app_router and has_pages_router:
            violations.append(StackViolation(
                violation_type=StackViolationType.FRAMEWORK_CONFLICT,
                file_path="app, pages",
                expected="Single routing approach",
                actual="Mixed App Router and Pages Router",
                severity="error",
                description="Next.js App Router and Pages Router are both present",
                fix_suggestion="Choose either App Router (recommended) or Pages Router and remove the other"
            ))
        
        # Check for TypeScript/JavaScript mixing
        ts_files = self._count_files_by_extension(workspace_path, ['.ts', '.tsx'])
        js_files = self._count_files_by_extension(workspace_path, ['.js', '.jsx'])
        
        if ts_files > 0 and js_files > 0:
            # Allow some JS files (like config files) but warn if significant mixing
            ratio = min(ts_files, js_files) / max(ts_files, js_files)
            if ratio > 0.3:  # More than 30% of the smaller type
                violations.append(StackViolation(
                    violation_type=StackViolationType.FRAMEWORK_CONFLICT,
                    file_path="multiple files",
                    expected="Consistent TypeScript or JavaScript",
                    actual=f"{ts_files} TS files, {js_files} JS files",
                    severity="warning",
                    description="Significant mixing of TypeScript and JavaScript files",
                    fix_suggestion="Convert to consistent TypeScript or JavaScript usage"
                ))
        
        return violations
    
    def _count_files_by_extension(self, workspace_path: str, extensions: List[str]) -> int:
        """Count files with specific extensions."""
        count = 0
        for root, dirs, files in os.walk(workspace_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in [
                'node_modules', '.git', 'dist', 'build', '__pycache__', '.next'
            ]]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    count += 1
        
        return count
    
    def _generate_consistency_summary(self, violations: List[StackViolation], 
                                    primary_stack: TechnologyStack) -> str:
        """Generate human-readable consistency summary."""
        if not violations:
            return f"✅ Project is fully consistent with {primary_stack.value} standards"
        
        error_count = len([v for v in violations if v.severity == "error"])
        warning_count = len([v for v in violations if v.severity == "warning"])
        info_count = len([v for v in violations if v.severity == "info"])
        
        summary_parts = [f"Technology stack validation for {primary_stack.value}:"]
        
        if error_count > 0:
            summary_parts.append(f"❌ {error_count} critical violations")
        if warning_count > 0:
            summary_parts.append(f"⚠️ {warning_count} warnings")
        if info_count > 0:
            summary_parts.append(f"ℹ️ {info_count} suggestions")
        
        return " | ".join(summary_parts)
    
    def _generate_recommendations(self, violations: List[StackViolation], 
                                primary_stack: TechnologyStack) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            if violation.violation_type not in violation_types:
                violation_types[violation.violation_type] = []
            violation_types[violation.violation_type].append(violation)
        
        # Generate recommendations per violation type
        if StackViolationType.FILE_EXTENSION_MISMATCH in violation_types:
            count = len(violation_types[StackViolationType.FILE_EXTENSION_MISMATCH])
            recommendations.append(f"Standardize file extensions for {count} files to match {primary_stack.value}")
        
        if StackViolationType.FRAMEWORK_CONFLICT in violation_types:
            recommendations.append("Resolve framework conflicts to maintain consistency")
        
        if StackViolationType.CONVENTION_VIOLATION in violation_types:
            count = len(violation_types[StackViolationType.CONVENTION_VIOLATION])
            recommendations.append(f"Reorganize {count} files to follow standard directory conventions")
        
        # Add general recommendations
        if violations:
            recommendations.append("Run automated code formatting tools (Prettier, ESLint)")
            recommendations.append("Set up pre-commit hooks to enforce consistency")
        
        return recommendations
    
    def _violation_to_dict(self, violation: StackViolation) -> Dict[str, Any]:
        """Convert violation to dictionary for serialization."""
        return {
            'type': violation.violation_type.value,
            'file_path': violation.file_path,
            'expected': violation.expected,
            'actual': violation.actual,
            'severity': violation.severity,
            'description': violation.description,
            'fix_suggestion': violation.fix_suggestion
        }
    
    def get_stack_guidelines_for_task(self, primary_stack: TechnologyStack, 
                                    task_type: str) -> List[str]:
        """Get specific guidelines for a task based on the technology stack."""
        guidelines = []
        
        # General stack guidelines
        if "typescript" in primary_stack.value:
            guidelines.extend([
                "Use TypeScript with proper type definitions",
                "Prefer .ts files for utilities and .tsx files for React components",
                "Include explicit return types for functions",
                "Use interface or type definitions for complex objects"
            ])
        elif "javascript" in primary_stack.value:
            guidelines.extend([
                "Use modern JavaScript (ES6+) syntax",
                "Prefer .js files for utilities and .jsx files for React components",
                "Use JSDoc comments for type information",
                "Follow consistent naming conventions"
            ])
        
        # Framework-specific guidelines
        if "nextjs" in primary_stack.value:
            guidelines.extend([
                "Use Next.js App Router structure (src/app/) for new features",
                "Follow Next.js file-based routing conventions",
                "Use Next.js specific components (Image, Link) when appropriate",
                "Place API routes in app/api/ directory"
            ])
        elif "react" in primary_stack.value:
            guidelines.extend([
                "Use functional components with React hooks",
                "Group related components in directories",
                "Use descriptive component names with PascalCase",
                "Implement proper prop validation"
            ])
        
        # Task-specific guidelines
        if task_type in ["component", "frontend"]:
            guidelines.extend([
                "Place components in designated component directories",
                "Use consistent styling approach (CSS modules, styled-components, etc.)",
                "Implement proper error boundaries for robustness"
            ])
        elif task_type in ["api", "backend"]:
            guidelines.extend([
                "Organize routes and controllers logically",
                "Implement proper error handling and validation",
                "Use consistent response formats"
            ])
        
        return guidelines


# Global tech stack enforcer instance
tech_stack_enforcer = TechStackEnforcer()