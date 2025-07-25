"""Workspace Structure Analyzer and Validator."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class TechnologyStack(Enum):
    """Supported technology stacks."""
    REACT_TS = "react_typescript"
    REACT_JS = "react_javascript"
    NEXTJS_TS = "nextjs_typescript"
    NEXTJS_JS = "nextjs_javascript"
    NODE_TS = "node_typescript"
    NODE_JS = "node_javascript"
    EXPRESS_TS = "express_typescript"
    EXPRESS_JS = "express_javascript"
    FASTAPI_PY = "fastapi_python"
    FLASK_PY = "flask_python"
    DJANGO_PY = "django_python"


class ProjectStructureIssue(Enum):
    """Types of project structure issues."""
    DUPLICATE_DIRECTORIES = "duplicate_directories"
    NESTED_SAME_NAME = "nested_same_name"  # frontend/frontend, backend/backend
    MIXED_CONVENTIONS = "mixed_conventions"  # App Router + Pages Router
    MISSING_CORE_FILES = "missing_core_files"
    INCONSISTENT_TECH_STACK = "inconsistent_tech_stack"
    INVALID_NESTING = "invalid_nesting"


@dataclass
class StructureIssue:
    """Represents a workspace structure issue."""
    issue_type: ProjectStructureIssue
    path: str
    description: str
    severity: str  # "critical", "warning", "info"
    suggested_fix: str
    conflicting_paths: List[str] = field(default_factory=list)


@dataclass
class TechnologyDetection:
    """Technology stack detection result."""
    stack: TechnologyStack
    confidence: float
    evidence: List[str]
    package_manager: Optional[str] = None  # npm, yarn, pip, etc.


@dataclass
class WorkspaceStructure:
    """Represents the analyzed workspace structure."""
    root_path: str
    technology_stacks: List[TechnologyDetection]
    project_type: str  # "monorepo", "fullstack", "frontend", "backend"
    main_directories: List[str]
    config_files: List[str]
    issues: List[StructureIssue]
    
    # Recommended structure
    recommended_structure: Dict[str, Any] = field(default_factory=dict)


class WorkspaceAnalyzer:
    """Analyzes and validates workspace structure."""
    
    # Technology detection patterns
    TECH_PATTERNS = {
        TechnologyStack.NEXTJS_TS: {
            'files': ['next.config.js', 'next.config.ts', 'next.config.mjs'],
            'dirs': ['app', 'pages'],
            'package_deps': ['next', '@types/react'],
            'file_extensions': ['.tsx', '.ts']
        },
        TechnologyStack.NEXTJS_JS: {
            'files': ['next.config.js', 'next.config.mjs'],
            'dirs': ['app', 'pages'],
            'package_deps': ['next'],
            'file_extensions': ['.jsx', '.js']
        },
        TechnologyStack.REACT_TS: {
            'files': ['vite.config.ts', 'webpack.config.js'],
            'dirs': ['src'],
            'package_deps': ['react', '@types/react'],
            'file_extensions': ['.tsx', '.ts']
        },
        TechnologyStack.REACT_JS: {
            'files': ['vite.config.js', 'webpack.config.js'],
            'dirs': ['src'],
            'package_deps': ['react'],
            'file_extensions': ['.jsx', '.js']
        },
        TechnologyStack.EXPRESS_TS: {
            'files': ['tsconfig.json'],
            'dirs': ['src', 'routes'],
            'package_deps': ['express', '@types/express'],
            'file_extensions': ['.ts']
        },
        TechnologyStack.EXPRESS_JS: {
            'files': ['package.json'],
            'dirs': ['routes', 'controllers'],
            'package_deps': ['express'],
            'file_extensions': ['.js']
        },
        TechnologyStack.FASTAPI_PY: {
            'files': ['requirements.txt', 'pyproject.toml'],
            'dirs': ['app', 'api'],
            'package_deps': ['fastapi', 'uvicorn'],
            'file_extensions': ['.py']
        }
    }
    
    # Common structure issues to detect
    ISSUE_PATTERNS = {
        ProjectStructureIssue.NESTED_SAME_NAME: [
            r'frontend[/\\]frontend',
            r'backend[/\\]backend',
            r'src[/\\]src',
            r'api[/\\]api',
            r'components[/\\]components'
        ],
        ProjectStructureIssue.DUPLICATE_DIRECTORIES: [
            ('components', 'components'),
            ('utils', 'utils'),
            ('services', 'services'),
            ('hooks', 'hooks')
        ]
    }
    
    def __init__(self):
        self.workspace_cache: Dict[str, WorkspaceStructure] = {}
    
    def analyze_workspace(self, workspace_path: str, force_refresh: bool = False) -> WorkspaceStructure:
        """Analyze workspace structure comprehensively."""
        workspace_path = os.path.abspath(workspace_path)
        
        # Check cache
        if not force_refresh and workspace_path in self.workspace_cache:
            return self.workspace_cache[workspace_path]
        
        logger.info(f"Analyzing workspace structure: {workspace_path}")
        
        structure = WorkspaceStructure(
            root_path=workspace_path,
            technology_stacks=[],
            project_type="unknown",
            main_directories=[],
            config_files=[],
            issues=[]
        )
        
        if not os.path.exists(workspace_path):
            structure.issues.append(StructureIssue(
                issue_type=ProjectStructureIssue.MISSING_CORE_FILES,
                path=workspace_path,
                description="Workspace directory does not exist",
                severity="critical",
                suggested_fix=f"Create workspace directory: {workspace_path}"
            ))
            return structure
        
        # Analyze directory structure
        structure.main_directories = self._get_main_directories(workspace_path)
        structure.config_files = self._find_config_files(workspace_path)
        
        # Detect technology stacks
        structure.technology_stacks = self._detect_technology_stacks(workspace_path)
        
        # Determine project type
        structure.project_type = self._determine_project_type(structure)
        
        # Find structure issues
        structure.issues = self._find_structure_issues(workspace_path, structure)
        
        # Generate recommendations
        structure.recommended_structure = self._generate_recommended_structure(structure)
        
        # Cache result
        self.workspace_cache[workspace_path] = structure
        
        logger.info(f"Workspace analysis complete. Found {len(structure.issues)} issues.")
        return structure
    
    def _get_main_directories(self, workspace_path: str) -> List[str]:
        """Get main directories in workspace."""
        main_dirs = []
        
        try:
            for item in os.listdir(workspace_path):
                item_path = os.path.join(workspace_path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    main_dirs.append(item)
        except OSError as e:
            logger.warning(f"Could not list directory {workspace_path}: {e}")
        
        return sorted(main_dirs)
    
    def _find_config_files(self, workspace_path: str) -> List[str]:
        """Find configuration files in workspace."""
        config_patterns = [
            'package.json', 'tsconfig.json', 'next.config.*',
            'vite.config.*', 'webpack.config.*', 'requirements.txt',
            'pyproject.toml', 'Dockerfile', 'docker-compose.*',
            '.env*', 'tailwind.config.*', 'jest.config.*'
        ]
        
        config_files = []
        
        for root, dirs, files in os.walk(workspace_path):
            # Limit depth to avoid deep scanning
            if root.count(os.sep) - workspace_path.count(os.sep) > 2:
                continue
                
            for file in files:
                for pattern in config_patterns:
                    if self._matches_pattern(file, pattern):
                        rel_path = os.path.relpath(os.path.join(root, file), workspace_path)
                        config_files.append(rel_path)
                        break
        
        return sorted(config_files)
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern (with basic glob support)."""
        if '*' in pattern:
            # Convert to regex
            regex_pattern = pattern.replace('.', r'\.').replace('*', '.*')
            return re.match(f"^{regex_pattern}$", filename) is not None
        return filename == pattern
    
    def _detect_technology_stacks(self, workspace_path: str) -> List[TechnologyDetection]:
        """Detect technology stacks used in the workspace."""
        detections = []
        
        for stack, patterns in self.TECH_PATTERNS.items():
            confidence = 0.0
            evidence = []
            
            # Check for config files
            for config_file in patterns.get('files', []):
                if self._file_exists_in_workspace(workspace_path, config_file):
                    confidence += 0.3
                    evidence.append(f"Found config file: {config_file}")
            
            # Check for directories
            for directory in patterns.get('dirs', []):
                if self._directory_exists_in_workspace(workspace_path, directory):
                    confidence += 0.2
                    evidence.append(f"Found directory: {directory}")
            
            # Check package dependencies
            package_deps = self._get_package_dependencies(workspace_path)
            for dep in patterns.get('package_deps', []):
                if dep in package_deps:
                    confidence += 0.3
                    evidence.append(f"Found dependency: {dep}")
            
            # Check file extensions
            file_extensions = self._get_file_extensions(workspace_path)
            for ext in patterns.get('file_extensions', []):
                if ext in file_extensions:
                    confidence += 0.2
                    evidence.append(f"Found file extension: {ext}")
            
            # Only include if confidence is reasonable
            if confidence >= 0.4:
                detections.append(TechnologyDetection(
                    stack=stack,
                    confidence=min(confidence, 1.0),
                    evidence=evidence
                ))
        
        return sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    def _file_exists_in_workspace(self, workspace_path: str, filename: str) -> bool:
        """Check if file exists in workspace (including subdirectories)."""
        for root, dirs, files in os.walk(workspace_path):
            if filename in files:
                return True
        return False
    
    def _directory_exists_in_workspace(self, workspace_path: str, dirname: str) -> bool:
        """Check if directory exists in workspace."""
        for root, dirs, files in os.walk(workspace_path):
            if dirname in dirs:
                return True
        return False
    
    def _get_package_dependencies(self, workspace_path: str) -> Set[str]:
        """Get package dependencies from package.json files."""
        dependencies = set()
        
        for root, dirs, files in os.walk(workspace_path):
            if 'package.json' in files:
                package_path = os.path.join(root, 'package.json')
                try:
                    with open(package_path, 'r', encoding='utf-8') as f:
                        package_data = json.load(f)
                        
                    # Get all dependencies
                    for dep_type in ['dependencies', 'devDependencies', 'peerDependencies']:
                        if dep_type in package_data:
                            dependencies.update(package_data[dep_type].keys())
                            
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Could not read package.json at {package_path}: {e}")
        
        return dependencies
    
    def _get_file_extensions(self, workspace_path: str) -> Set[str]:
        """Get all file extensions in workspace."""
        extensions = set()
        
        for root, dirs, files in os.walk(workspace_path):
            # Skip node_modules and other common ignored directories
            dirs[:] = [d for d in dirs if d not in [
                'node_modules', '.git', 'dist', 'build', '__pycache__', '.next'
            ]]
            
            for file in files:
                if '.' in file:
                    ext = os.path.splitext(file)[1]
                    if ext:
                        extensions.add(ext)
        
        return extensions
    
    def _determine_project_type(self, structure: WorkspaceStructure) -> str:
        """Determine the type of project based on structure."""
        dirs = set(structure.main_directories)
        
        # Check for monorepo patterns
        if 'packages' in dirs or 'apps' in dirs:
            return "monorepo"
        
        # Check for fullstack patterns
        if ('frontend' in dirs and 'backend' in dirs) or ('client' in dirs and 'server' in dirs):
            return "fullstack"
        
        # Check technology stacks
        if structure.technology_stacks:
            primary_stack = structure.technology_stacks[0].stack
            
            if primary_stack in [TechnologyStack.NEXTJS_TS, TechnologyStack.NEXTJS_JS, 
                               TechnologyStack.REACT_TS, TechnologyStack.REACT_JS]:
                return "frontend"
            elif primary_stack in [TechnologyStack.EXPRESS_TS, TechnologyStack.EXPRESS_JS,
                                 TechnologyStack.FASTAPI_PY, TechnologyStack.FLASK_PY]:
                return "backend"
        
        return "unknown"
    
    def _find_structure_issues(self, workspace_path: str, structure: WorkspaceStructure) -> List[StructureIssue]:
        """Find structure issues in the workspace."""
        issues = []
        
        # Check for nested same-name directories
        issues.extend(self._find_nested_same_name_issues(workspace_path))
        
        # Check for duplicate directories
        issues.extend(self._find_duplicate_directory_issues(workspace_path))
        
        # Check for mixed conventions (e.g., Next.js App Router + Pages Router)
        issues.extend(self._find_mixed_convention_issues(workspace_path, structure))
        
        # Check for inconsistent technology stacks
        issues.extend(self._find_tech_stack_inconsistencies(structure))
        
        return issues
    
    def _find_nested_same_name_issues(self, workspace_path: str) -> List[StructureIssue]:
        """Find nested same-name directory issues like frontend/frontend."""
        issues = []
        
        for root, dirs, files in os.walk(workspace_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                rel_path = os.path.relpath(dir_path, workspace_path)
                
                # Check if this creates a nested same-name pattern
                path_parts = rel_path.split(os.sep)
                if len(path_parts) >= 2 and path_parts[-1] == path_parts[-2]:
                    issues.append(StructureIssue(
                        issue_type=ProjectStructureIssue.NESTED_SAME_NAME,
                        path=rel_path,
                        description=f"Nested same-name directory: {rel_path}",
                        severity="critical",
                        suggested_fix=f"Move contents from {rel_path} to parent directory {'/'.join(path_parts[:-1])}"
                    ))
        
        return issues
    
    def _find_duplicate_directory_issues(self, workspace_path: str) -> List[StructureIssue]:
        """Find duplicate directories with similar purposes."""
        issues = []
        
        # Common directory duplicates to check
        similar_dirs = [
            (['components', 'component'], 'component organization'),
            (['utils', 'utilities', 'helpers'], 'utility functions'),
            (['services', 'service'], 'service layer'),
            (['hooks', 'hook'], 'React hooks'),
            (['types', 'type', 'interfaces'], 'type definitions'),
            (['styles', 'css', 'scss'], 'styling')
        ]
        
        all_dirs = []
        for root, dirs, files in os.walk(workspace_path):
            for dir_name in dirs:
                rel_path = os.path.relpath(os.path.join(root, dir_name), workspace_path)
                all_dirs.append((dir_name.lower(), rel_path))
        
        for similar_group, purpose in similar_dirs:
            found_dirs = []
            for dir_name_lower, full_path in all_dirs:
                if dir_name_lower in similar_group:
                    found_dirs.append(full_path)
            
            if len(found_dirs) > 1:
                issues.append(StructureIssue(
                    issue_type=ProjectStructureIssue.DUPLICATE_DIRECTORIES,
                    path=found_dirs[0],
                    description=f"Multiple directories for {purpose}: {', '.join(found_dirs)}",
                    severity="warning",
                    suggested_fix=f"Consolidate into single directory for {purpose}",
                    conflicting_paths=found_dirs
                ))
        
        return issues
    
    def _find_mixed_convention_issues(self, workspace_path: str, structure: WorkspaceStructure) -> List[StructureIssue]:
        """Find mixed convention issues (e.g., Next.js App Router + Pages Router)."""
        issues = []
        
        # Check for Next.js mixed routing conventions
        has_app_router = self._directory_exists_in_workspace(workspace_path, 'app')
        has_pages_router = self._directory_exists_in_workspace(workspace_path, 'pages')
        
        if has_app_router and has_pages_router:
            issues.append(StructureIssue(
                issue_type=ProjectStructureIssue.MIXED_CONVENTIONS,
                path="app, pages",
                description="Mixed Next.js routing conventions: both App Router (/app) and Pages Router (/pages) detected",
                severity="critical",
                suggested_fix="Choose either App Router (recommended for new projects) or Pages Router and remove the other",
                conflicting_paths=["app", "pages"]
            ))
        
        return issues
    
    def _find_tech_stack_inconsistencies(self, structure: WorkspaceStructure) -> List[StructureIssue]:
        """Find technology stack inconsistencies."""
        issues = []
        
        if len(structure.technology_stacks) > 2:
            # Too many different tech stacks might indicate confusion
            stack_names = [stack.stack.value for stack in structure.technology_stacks]
            issues.append(StructureIssue(
                issue_type=ProjectStructureIssue.INCONSISTENT_TECH_STACK,
                path="multiple",
                description=f"Multiple technology stacks detected: {', '.join(stack_names)}",
                severity="warning",
                suggested_fix="Consider consolidating to a primary technology stack"
            ))
        
        return issues
    
    def _generate_recommended_structure(self, structure: WorkspaceStructure) -> Dict[str, Any]:
        """Generate recommended structure based on analysis."""
        recommendations = {
            "primary_tech_stack": None,
            "recommended_directories": [],
            "files_to_create": [],
            "directories_to_consolidate": [],
            "directories_to_remove": []
        }
        
        # Set primary tech stack
        if structure.technology_stacks:
            recommendations["primary_tech_stack"] = structure.technology_stacks[0].stack.value
        
        # Based on project type, recommend structure
        if structure.project_type == "fullstack":
            recommendations["recommended_directories"] = [
                "frontend/src", "frontend/public", "frontend/components",
                "backend/src", "backend/routes", "backend/models",
                "shared", "docs"
            ]
        elif structure.project_type == "frontend":
            if structure.technology_stacks and "nextjs" in structure.technology_stacks[0].stack.value:
                recommendations["recommended_directories"] = [
                    "src/app", "src/components", "src/lib", "public"
                ]
            else:
                recommendations["recommended_directories"] = [
                    "src/components", "src/utils", "src/hooks", "public"
                ]
        elif structure.project_type == "backend":
            recommendations["recommended_directories"] = [
                "src/routes", "src/models", "src/services", "src/utils", "tests"
            ]
        
        # Add consolidation recommendations based on issues
        for issue in structure.issues:
            if issue.issue_type == ProjectStructureIssue.DUPLICATE_DIRECTORIES:
                recommendations["directories_to_consolidate"].extend(issue.conflicting_paths)
            elif issue.issue_type == ProjectStructureIssue.NESTED_SAME_NAME:
                recommendations["directories_to_remove"].append(issue.path)
        
        return recommendations
    
    def get_workspace_context_for_claude(self, workspace_path: str) -> str:
        """Generate workspace context string for Claude CLI."""
        structure = self.analyze_workspace(workspace_path)
        
        context_parts = [
            f"# Workspace Structure Analysis",
            f"**Root Path:** {structure.root_path}",
            f"**Project Type:** {structure.project_type}",
            "",
        ]
        
        # Technology stack information
        if structure.technology_stacks:
            context_parts.append("## Technology Stack")
            for i, tech in enumerate(structure.technology_stacks):
                confidence_str = f"({tech.confidence:.1%} confidence)"
                context_parts.append(f"{i+1}. **{tech.stack.value}** {confidence_str}")
                for evidence in tech.evidence[:3]:  # Top 3 evidence
                    context_parts.append(f"   - {evidence}")
            context_parts.append("")
        
        # Current directory structure
        context_parts.append("## Current Directory Structure")
        for dir_name in structure.main_directories:
            context_parts.append(f"- {dir_name}/")
        context_parts.append("")
        
        # Critical issues
        critical_issues = [issue for issue in structure.issues if issue.severity == "critical"]
        if critical_issues:
            context_parts.append("## ⚠️ Critical Structure Issues")
            for issue in critical_issues:
                context_parts.append(f"- **{issue.description}**")
                context_parts.append(f"  - Path: {issue.path}")
                context_parts.append(f"  - Fix: {issue.suggested_fix}")
            context_parts.append("")
        
        # Configuration files
        if structure.config_files:
            context_parts.append("## Configuration Files")
            for config_file in structure.config_files:
                context_parts.append(f"- {config_file}")
            context_parts.append("")
        
        # Recommendations
        if structure.recommended_structure.get("primary_tech_stack"):
            context_parts.append("## Recommendations")
            context_parts.append(f"- **Primary Tech Stack:** {structure.recommended_structure['primary_tech_stack']}")
            
            if structure.recommended_structure.get("directories_to_remove"):
                context_parts.append("- **Remove these nested directories:**")
                for dir_path in structure.recommended_structure["directories_to_remove"]:
                    context_parts.append(f"  - {dir_path}")
            
            if structure.recommended_structure.get("directories_to_consolidate"):
                context_parts.append("- **Consolidate these duplicate directories:**")
                for dir_path in structure.recommended_structure["directories_to_consolidate"]:
                    context_parts.append(f"  - {dir_path}")
        
        return "\n".join(context_parts)


# Global analyzer instance
workspace_analyzer = WorkspaceAnalyzer()