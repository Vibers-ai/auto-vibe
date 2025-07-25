"""Master Planner Agent

This module implements the Master Planner Agent that analyzes the ProjectBrief.md
and generates a comprehensive technical architecture and tasks.json file.
"""


import sys
from pathlib import Path
# src 디렉토리를 Python path에 추가
src_dir = Path(__file__).parent.parent if 'src' in str(Path(__file__).parent) else Path(__file__).parent
while src_dir.name != 'src' and src_dir.parent != src_dir:
    src_dir = src_dir.parent
if src_dir.name == 'src':
    sys.path.insert(0, str(src_dir))

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from shared.utils.config import Config
from shared.utils.file_utils import read_text_file, write_json_file, write_text_file
from shared.agents.architecture_patterns import architecture_library, ProjectType
from shared.agents.task_optimizer import task_optimizer

logger = logging.getLogger(__name__)
console = Console()


MASTER_PLANNER_PROMPT = """You are an expert software architect and project planner. Your task is to analyze the provided project documentation and create a comprehensive technical plan.

Given the ProjectBrief below, you must:

1. **Define the High-Level Architecture**
   - Choose the technology stack (e.g., Backend: Python/FastAPI, Frontend: React/TypeScript)
   - Define the system architecture (monolithic, microservices, etc.)
   - Explain your technology choices
   - **IMPORTANT**: Accurately detect if TypeScript is being used (Next.js, React with TypeScript, etc.)

2. **Design the Database Schema**
   - Define all tables, columns, data types, relationships, and constraints
   - Output as SQL CREATE TABLE statements
   - Consider scalability and normalization

3. **Design API Endpoints**
   - Define all necessary REST API endpoints
   - Include HTTP methods, URL paths, request/response schemas
   - Follow RESTful conventions and best practices
   - Output in OpenAPI 3.0 format

4. **Design Frontend Components**
   - Break down the UI into a component hierarchy
   - Define component purposes, state, and props
   - Consider reusability and maintainability
   - **IMPORTANT**: Use correct file extensions based on technology stack:
     - TypeScript React: .tsx for components, .ts for utilities
     - JavaScript React: .jsx for components, .js for utilities
     - Next.js with TypeScript: .tsx/.ts files with proper src/ structure

5. **Create Detailed Tasks** (MONOLITHIC APPROACH - SMALL INCREMENTS)
   - Break down the implementation into VERY SMALL, atomic tasks
   - Each task should be completable in 30 minutes to 2 hours MAX
   - **MONOLITHIC STRATEGY**: Build incrementally in the same codebase
   - Start with the simplest foundation and add features one by one
   - Each task should add ONE clear piece of functionality
   - Include clear dependencies - later tasks build on earlier ones
   - Define specific, testable acceptance criteria for each task
   - **CRITICAL**: Ensure file paths match the actual project structure:
     - For Next.js: Use src/app/, src/components/, etc.
     - For standard React: Use src/components/, src/utils/, etc.
     - Match file extensions to the chosen technology stack

## ProjectBrief:

{project_brief}

## Required Output Format:

Please structure your response in the following sections:

### 1. ARCHITECTURE
```
Technology Stack:
- Backend: [technology choices]
- Frontend: [technology choices]
- Database: [technology choices]
- Additional Tools: [any other technologies]

Architecture Pattern: [pattern choice and justification]
```

### 2. DATABASE_SCHEMA
```sql
-- SQL CREATE TABLE statements
```

### 3. API_DESIGN
```yaml
# OpenAPI 3.0 specification
```

### 4. FRONTEND_COMPONENTS
```
Component Hierarchy:
- ComponentName
  - purpose: [description]
  - state: [state variables]
  - props: [component props]
  - children: [child components]
```

### 5. TASKS_JSON
```json
{
  "project_id": "generated-project-id",
  "created_at": "timestamp",
  "total_tasks": number,
  "tasks": [
    {
      "id": "unique-task-id",
      "description": "Clear description of what needs to be done",
      "type": "setup|backend|frontend|database|testing|deployment",
      "dependencies": ["task-id-1", "task-id-2"],
      "project_area": "backend|frontend|shared",
      "files_to_create_or_modify": ["file/path1.py", "file/path2.js"],
      "acceptance_criteria": {
        "tests": [
          {
            "type": "unit|integration|e2e",
            "file": "tests/test_file.py",
            "function": "test_function_name"
          }
        ],
        "linting": {
          "command": "ruff check file/path.py",
          "files": ["file/path.py"]
        },
        "manual_checks": ["Check that X works", "Verify Y is displayed"]
      },
      "estimated_hours": 1.5,
      "technical_details": {
        "framework_specific": "Any framework-specific implementation notes",
        "dependencies_to_install": ["package1==1.0.0", "package2>=2.0"],
        "environment_variables": ["API_KEY", "DATABASE_URL"]
      }
    }
  ]
}
```

Remember (MONOLITHIC SMALL-TASK APPROACH):
- Tasks should be VERY SMALL and focused on ONE thing
- Each task adds a single feature/component/function
- Build incrementally - start simple, add complexity gradually
- Dependencies should form a clear progression
- Example good task size: "Create User model with basic fields"
- Example bad task size: "Implement entire user management system"
- Include specific file paths and function names
- Each task should have 1-3 clear acceptance criteria maximum
- Use descriptive IDs like "create-user-model", "add-user-api-endpoint"

Think step by step and be comprehensive in your analysis."""


class MasterPlannerAgent:
    """Agent responsible for creating the master plan and tasks from ProjectBrief."""
    
    def __init__(self, config: Config):
        self.config = config
        self.gemini_model = None
        self.architecture_library = architecture_library
        self.task_optimizer = task_optimizer
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Initialize Gemini model."""
        genai.configure(api_key=self.config.gemini_api_key)
        self.gemini_model = genai.GenerativeModel(self.config.gemini_model)
    
    def create_master_plan(self, project_brief_path: str = "ProjectBrief.md") -> str:
        """Create the master plan and tasks.json from ProjectBrief.md
        
        Args:
            project_brief_path: Path to the ProjectBrief.md file
            
        Returns:
            Path to the generated tasks.json file
        """
        console.print("[blue]Creating master plan from ProjectBrief...[/blue]")
        
        # Read ProjectBrief
        try:
            project_brief = read_text_file(project_brief_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"ProjectBrief not found at: {project_brief_path}")
        
        # Generate plan using Gemini
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing project and creating master plan...", total=None)
            
            try:
                response = self._generate_plan(project_brief)
                progress.update(task, completed=True)
            except Exception as e:
                console.print(f"[red]Error generating plan: {e}[/red]")
                raise
        
        # Parse response and extract components
        parsed_response = self._parse_response(response)
        
        # Save architecture document
        self._save_architecture_doc(parsed_response)
        
        # Extract and validate tasks.json
        tasks_json = parsed_response.get('tasks_json', {})
        
        # Apply task optimization
        if tasks_json.get('tasks'):
            console.print("[blue]Optimizing task dependencies and scheduling...[/blue]")
            optimized_tasks = self.task_optimizer.optimize_task_dependencies(tasks_json['tasks'])
            tasks_json['tasks'] = optimized_tasks
            
            # Log optimization results
            critical_tasks = [t for t in optimized_tasks if t.get('technical_details', {}).get('critical_path')]
            console.print(f"[green]✓ Identified {len(critical_tasks)} critical path tasks[/green]")
        
        validated_tasks = self._validate_and_enhance_tasks(tasks_json)
        
        # Save tasks.json
        tasks_path = "tasks.json"
        write_json_file(tasks_path, validated_tasks)
        
        console.print(f"[green]✓ Master plan created successfully![/green]")
        console.print(f"[green]✓ Architecture document saved to: architecture.md[/green]")
        console.print(f"[green]✓ Tasks saved to: {tasks_path}[/green]")
        console.print(f"[green]  Total tasks: {len(validated_tasks.get('tasks', []))}[/green]")
        
        return tasks_path
    
    def _generate_plan(self, project_brief: str) -> str:
        """Generate the master plan using Gemini."""
        console.print(f"[cyan]📋 Starting _generate_plan function[/cyan]")
        console.print(f"[cyan]📄 Project brief length: {len(project_brief)} characters[/cyan]")
        
        # Analyze project brief to determine project type and recommend architecture
        project_analysis = self._analyze_project_brief(project_brief)
        architecture_recommendations = self._get_architecture_recommendations(project_analysis)
        
        # Enhanced prompt with architecture recommendations
        enhanced_prompt = self._create_enhanced_prompt(project_brief, architecture_recommendations)
        
        try:
            prompt = enhanced_prompt
            console.print(f"[cyan]🎯 Prompt template formatted successfully with architecture recommendations[/cyan]")
        except Exception as e:
            console.print(f"[red]❌ Error formatting prompt: {e}[/red]")
            raise
        
        try:
            console.print(f"[yellow]🔍 Calling Gemini API with model: {self.config.gemini_model}[/yellow]")
            console.print(f"[yellow]🔑 API key configured: {'Yes' if self.config.gemini_api_key else 'No'}[/yellow]")
            console.print(f"[yellow]📝 Prompt length: {len(prompt)} characters[/yellow]")
            
            logger.info(f"Calling Gemini API with model: {self.config.gemini_model}")
            logger.info(f"API key configured: {'Yes' if self.config.gemini_api_key else 'No'}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=30000,
                )
            )
            
            console.print(f"[green]✅ Gemini API response received, length: {len(response.text)} characters[/green]")
            logger.info(f"Gemini API response received, length: {len(response.text)} characters")
            return response.text
            
        except Exception as e:
            console.print(f"[red]❌ Error calling Gemini API: {e}[/red]")
            logger.error(f"Error calling Gemini API: {e}")
            raise
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured response from Gemini."""
        sections = {
            'architecture': '',
            'database_schema': '',
            'api_design': '',
            'frontend_components': '',
            'tasks_json': {}
        }
        
        # Split response into sections
        current_section = None
        current_content = []
        in_code_block = False
        code_block_content = []
        
        for line in response.split('\n'):
            # Check for section headers
            if line.startswith('### 1. ARCHITECTURE'):
                current_section = 'architecture'
                current_content = []
            elif line.startswith('### 2. DATABASE_SCHEMA'):
                current_section = 'database_schema'
                current_content = []
            elif line.startswith('### 3. API_DESIGN'):
                current_section = 'api_design'
                current_content = []
            elif line.startswith('### 4. FRONTEND_COMPONENTS'):
                current_section = 'frontend_components'
                current_content = []
            elif line.startswith('### 5. TASKS_JSON'):
                current_section = 'tasks_json'
                current_content = []
            elif current_section:
                # Handle code blocks
                if line.strip().startswith('```'):
                    if in_code_block:
                        # End of code block
                        if current_section == 'tasks_json':
                            # Parse JSON
                            try:
                                json_content = '\n'.join(code_block_content).strip()
                                # Clean up common JSON formatting issues
                                if not json_content.startswith('{'):
                                    # Find the first { and start from there
                                    start_idx = json_content.find('{')
                                    if start_idx != -1:
                                        json_content = json_content[start_idx:]
                                if not json_content.endswith('}'):
                                    # Find the last } and end there
                                    end_idx = json_content.rfind('}')
                                    if end_idx != -1:
                                        json_content = json_content[:end_idx+1]
                                
                                sections['tasks_json'] = json.loads(json_content)
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing tasks JSON: {e}")
                                logger.error(f"JSON content: {json_content[:500]}...")
                                sections['tasks_json'] = {}
                        else:
                            current_content.extend(code_block_content)
                        
                        in_code_block = False
                        code_block_content = []
                    else:
                        # Start of code block
                        in_code_block = True
                elif in_code_block:
                    code_block_content.append(line)
                else:
                    current_content.append(line)
                
                # Update section content
                if current_section != 'tasks_json' and not in_code_block:
                    sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _validate_and_enhance_tasks(self, tasks_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance the tasks JSON with additional metadata."""
        # Ensure required fields
        if not tasks_json:
            tasks_json = {}
        
        # Add metadata if missing
        if 'project_id' not in tasks_json:
            tasks_json['project_id'] = f"vibe-project-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if 'created_at' not in tasks_json:
            tasks_json['created_at'] = datetime.now().isoformat()
        
        # Detect technology stack from tasks
        tech_stack = self._detect_technology_stack(tasks_json)
        tasks_json['technology_stack'] = tech_stack
        
        # Validate tasks
        tasks = tasks_json.get('tasks', [])
        validated_tasks = []
        
        for idx, task in enumerate(tasks):
            # Ensure required fields
            if 'id' not in task:
                task['id'] = f"task-{idx + 1:04d}"
            
            if 'type' not in task:
                task['type'] = self._infer_task_type(task.get('description', ''))
            
            if 'dependencies' not in task:
                task['dependencies'] = []
            
            if 'status' not in task:
                task['status'] = 'pending'
            
            if 'project_area' not in task:
                task['project_area'] = self._infer_project_area(task)
            
            if 'files_to_create_or_modify' not in task:
                task['files_to_create_or_modify'] = []
            else:
                # Fix file extensions based on detected technology stack
                task['files_to_create_or_modify'] = self._fix_file_extensions(
                    task['files_to_create_or_modify'], 
                    tech_stack
                )
            
            # Ensure acceptance_criteria has all required fields
            if 'acceptance_criteria' not in task:
                task['acceptance_criteria'] = {}
            
            acceptance_criteria = task['acceptance_criteria']
            if 'tests' not in acceptance_criteria:
                acceptance_criteria['tests'] = []
            if 'linting' not in acceptance_criteria:
                acceptance_criteria['linting'] = {}
            if 'manual_checks' not in acceptance_criteria:
                acceptance_criteria['manual_checks'] = []
            
            # MONOLITHIC TASK SIZE VALIDATION
            task_complexity_score = self._calculate_task_complexity(task)
            if task_complexity_score > 3:
                console.print(f"[yellow]⚠️  Task '{task['id']}' may be too complex (score: {task_complexity_score})[/yellow]")
                console.print(f"   Description: {task.get('description', '')[:80]}...")
                console.print("   Consider breaking it into smaller subtasks")
            
            # Add estimated complexity to task metadata
            task['estimated_complexity'] = task_complexity_score
            
            validated_tasks.append(task)
        
        tasks_json['tasks'] = validated_tasks
        tasks_json['total_tasks'] = len(validated_tasks)
        
        return tasks_json
    
    def _calculate_task_complexity(self, task: Dict[str, Any]) -> int:
        """Calculate task complexity score (1-5 scale, lower is better for monolithic approach)."""
        complexity_score = 1
        
        # Check number of files to modify
        files_count = len(task.get('files_to_create_or_modify', []))
        if files_count > 5:
            complexity_score += 2
        elif files_count > 3:
            complexity_score += 1
        
        # Check description length and keywords
        description = task.get('description', '')
        if len(description) > 200:
            complexity_score += 1
        
        # Keywords that indicate complex tasks
        complex_keywords = [
            'entire', 'complete', 'full', 'system', 'all', 
            'implement everything', 'whole', 'comprehensive'
        ]
        if any(keyword in description.lower() for keyword in complex_keywords):
            complexity_score += 1
        
        # Check acceptance criteria
        criteria = task.get('acceptance_criteria', {})
        tests = criteria.get('tests', [])
        manual_checks = criteria.get('manual_checks', [])
        
        if len(tests) + len(manual_checks) > 5:
            complexity_score += 1
        
        # Multiple areas indicate task is too broad
        if ' and ' in description:
            complexity_score += 0.5
        
        return min(complexity_score, 5)  # Cap at 5
    
    def _infer_task_type(self, description: str) -> str:
        """Infer task type from description."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['setup', 'initialize', 'create project']):
            return 'setup'
        elif any(word in description_lower for word in ['database', 'model', 'schema', 'migration']):
            return 'database'
        elif any(word in description_lower for word in ['api', 'endpoint', 'route', 'backend']):
            return 'backend'
        elif any(word in description_lower for word in ['component', 'ui', 'frontend', 'react', 'vue']):
            return 'frontend'
        elif any(word in description_lower for word in ['test', 'testing', 'spec']):
            return 'testing'
        elif any(word in description_lower for word in ['deploy', 'docker', 'ci/cd']):
            return 'deployment'
        else:
            return 'general'
    
    def _infer_project_area(self, task: Dict[str, Any]) -> str:
        """Infer project area from task details."""
        task_type = task.get('type', '')
        description = task.get('description', '').lower()
        files = task.get('files_to_create_or_modify', [])
        
        # Check files first
        for file in files:
            if 'backend' in file or 'api' in file or 'server' in file:
                return 'backend'
            elif 'frontend' in file or 'client' in file or 'src/components' in file:
                return 'frontend'
        
        # Check task type
        if task_type in ['backend', 'database']:
            return 'backend'
        elif task_type == 'frontend':
            return 'frontend'
        
        # Check description
        if any(word in description for word in ['backend', 'api', 'server', 'database']):
            return 'backend'
        elif any(word in description for word in ['frontend', 'ui', 'component', 'client']):
            return 'frontend'
        
        return 'shared'
    
    def _save_architecture_doc(self, parsed_response: Dict[str, Any]) -> None:
        """Save the architecture documentation."""
        doc_content = f"""# System Architecture Document
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## High-Level Architecture

{parsed_response.get('architecture', 'No architecture information provided.')}

## Database Schema

{parsed_response.get('database_schema', 'No database schema provided.')}

## API Design

{parsed_response.get('api_design', 'No API design provided.')}

## Frontend Components

{parsed_response.get('frontend_components', 'No frontend component design provided.')}

---

This document provides the technical architecture for the project implementation.
Refer to tasks.json for the detailed implementation plan.
"""
        
        write_text_file("architecture.md", doc_content)
    
    def _detect_technology_stack(self, tasks_json: Dict[str, Any]) -> Dict[str, str]:
        """Detect technology stack from tasks and architecture info."""
        tech_stack = {
            'frontend_framework': 'unknown',
            'frontend_language': 'javascript',
            'backend_framework': 'unknown',
            'backend_language': 'unknown',
            'uses_typescript': False,
            'project_structure': 'standard'
        }
        
        # Check all tasks for technology hints
        all_files = []
        all_descriptions = []
        
        for task in tasks_json.get('tasks', []):
            all_files.extend(task.get('files_to_create_or_modify', []))
            all_descriptions.append(task.get('description', '').lower())
            
            # Check technical details for package dependencies
            tech_details = task.get('technical_details', {})
            deps = tech_details.get('dependencies_to_install', [])
            for dep in deps:
                dep_lower = dep.lower()
                if 'typescript' in dep_lower or '@types/' in dep_lower:
                    tech_stack['uses_typescript'] = True
                if 'next' in dep_lower:
                    tech_stack['frontend_framework'] = 'nextjs'
                    tech_stack['uses_typescript'] = True  # Next.js defaults to TypeScript
                elif 'react' in dep_lower:
                    tech_stack['frontend_framework'] = 'react'
                elif 'vue' in dep_lower:
                    tech_stack['frontend_framework'] = 'vue'
                elif 'fastapi' in dep_lower:
                    tech_stack['backend_framework'] = 'fastapi'
                    tech_stack['backend_language'] = 'python'
                elif 'express' in dep_lower:
                    tech_stack['backend_framework'] = 'express'
                    tech_stack['backend_language'] = 'javascript'
        
        # Analyze file patterns
        for file in all_files:
            file_lower = file.lower()
            # Frontend detection
            if 'next.config' in file_lower:
                tech_stack['frontend_framework'] = 'nextjs'
                tech_stack['uses_typescript'] = True
                tech_stack['project_structure'] = 'nextjs'
            elif '.tsx' in file_lower or '.ts' in file_lower:
                tech_stack['uses_typescript'] = True
            elif 'src/app/' in file_lower:
                tech_stack['project_structure'] = 'nextjs-app-router'
            elif 'pages/' in file_lower and 'nextjs' in tech_stack['frontend_framework']:
                tech_stack['project_structure'] = 'nextjs-pages-router'
            
            # Backend detection
            if '.py' in file_lower:
                tech_stack['backend_language'] = 'python'
            elif 'server.js' in file_lower or 'index.js' in file_lower:
                tech_stack['backend_language'] = 'javascript'
        
        # Analyze descriptions
        desc_text = ' '.join(all_descriptions)
        if 'typescript' in desc_text:
            tech_stack['uses_typescript'] = True
        if 'next.js' in desc_text or 'nextjs' in desc_text:
            tech_stack['frontend_framework'] = 'nextjs'
            tech_stack['uses_typescript'] = True
        
        # Set frontend language based on TypeScript detection
        if tech_stack['uses_typescript']:
            tech_stack['frontend_language'] = 'typescript'
            
        logger.info(f"Detected technology stack: {tech_stack}")
        return tech_stack
    
    def _fix_file_extensions(self, files: List[str], tech_stack: Dict[str, str]) -> List[str]:
        """Fix file extensions based on detected technology stack."""
        if not files:
            return []
            
        fixed_files = []
        
        for file in files:
            fixed_file = file
            
            # Frontend files
            if any(pattern in file for pattern in ['frontend/', 'client/', 'src/components/', 'src/app/']):
                if tech_stack['uses_typescript']:
                    # Convert .js/.jsx to .ts/.tsx
                    if file.endswith('.js'):
                        # Check if it's likely a component file
                        if any(pattern in file.lower() for pattern in ['component', 'page', 'layout', 'app']):
                            fixed_file = file[:-3] + '.tsx'
                        else:
                            fixed_file = file[:-3] + '.ts'
                    elif file.endswith('.jsx'):
                        fixed_file = file[:-4] + '.tsx'
                else:
                    # Convert .ts/.tsx to .js/.jsx if not using TypeScript
                    if file.endswith('.ts') and not file.endswith('.d.ts'):
                        fixed_file = file[:-3] + '.js'
                    elif file.endswith('.tsx'):
                        fixed_file = file[:-4] + '.jsx'
            
            # Fix Next.js specific paths
            if tech_stack['frontend_framework'] == 'nextjs':
                # Ensure proper src/ structure for Next.js
                if 'frontend/components/' in fixed_file and 'frontend/src/components/' not in fixed_file:
                    fixed_file = fixed_file.replace('frontend/components/', 'frontend/src/components/')
                elif 'frontend/app/' in fixed_file and 'frontend/src/app/' not in fixed_file:
                    fixed_file = fixed_file.replace('frontend/app/', 'frontend/src/app/')
            
            fixed_files.append(fixed_file)
            
            if fixed_file != file:
                logger.info(f"Fixed file path: {file} -> {fixed_file}")
        
        return fixed_files
    
    def _analyze_project_brief(self, project_brief: str) -> Dict[str, Any]:
        """Analyze the project brief to extract key information."""
        analysis = {
            "project_type": None,
            "requirements": [],
            "constraints": [],
            "technologies_mentioned": [],
            "features": []
        }
        
        brief_lower = project_brief.lower()
        
        # Detect project type
        project_types = {
            "e-commerce": ["e-commerce", "shopping", "cart", "checkout", "payment", "product catalog"],
            "social_platform": ["social", "chat", "messaging", "timeline", "friends", "posts"],
            "cms": ["content management", "cms", "blog", "articles", "publishing"],
            "api_service": ["api", "restful", "microservice", "backend service"],
            "web_app": ["web application", "dashboard", "portal", "platform"],
            "ml_system": ["machine learning", "ml", "ai", "prediction", "model"],
            "realtime": ["real-time", "realtime", "websocket", "collaborative", "live"]
        }
        
        for ptype, keywords in project_types.items():
            if any(keyword in brief_lower for keyword in keywords):
                analysis["project_type"] = ptype
                break
        
        # Extract requirements (simple pattern matching)
        requirement_patterns = [
            r"must\s+(?:have|support|provide|include)\s+([^.]+)",
            r"should\s+(?:have|support|provide|include)\s+([^.]+)",
            r"required\s+(?:features?|functionality):\s*([^.]+)",
            r"the\s+system\s+(?:must|should)\s+([^.]+)"
        ]
        
        for pattern in requirement_patterns:
            import re
            matches = re.findall(pattern, brief_lower, re.MULTILINE)
            analysis["requirements"].extend(matches)
        
        # Extract technology mentions
        tech_keywords = [
            "react", "angular", "vue", "next.js", "nextjs",
            "node.js", "nodejs", "express", "fastapi", "django",
            "postgresql", "mysql", "mongodb", "redis",
            "docker", "kubernetes", "aws", "azure"
        ]
        
        for tech in tech_keywords:
            if tech in brief_lower:
                analysis["technologies_mentioned"].append(tech)
        
        return analysis
    
    def _get_architecture_recommendations(self, project_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get architecture recommendations based on project analysis."""
        recommendations = {
            "suggested_templates": [],
            "tech_stack": None,
            "patterns": [],
            "considerations": []
        }
        
        # Get templates based on project type
        if project_analysis["project_type"]:
            templates = self.architecture_library.get_templates_for_project_type(
                project_analysis["project_type"]
            )
            recommendations["suggested_templates"] = [t.name for t in templates]
            
            # Use the first template as primary recommendation
            if templates:
                primary_template = templates[0]
                recommendations["tech_stack"] = primary_template.tech_stack
                recommendations["patterns"].append(primary_template.pattern.value)
                recommendations["considerations"] = primary_template.considerations
        
        # If no specific template found, use requirement-based recommendations
        if not recommendations["tech_stack"]:
            recommendations["tech_stack"] = self.architecture_library.get_tech_stack_recommendations(
                project_analysis["requirements"]
            )
        
        return recommendations
    
    def _create_enhanced_prompt(self, project_brief: str, architecture_recommendations: Dict[str, Any]) -> str:
        """Create an enhanced prompt with architecture recommendations."""
        # Build recommendation section
        recommendation_text = ""
        
        if architecture_recommendations["suggested_templates"]:
            recommendation_text += f"\n## Recommended Architecture Templates:\n"
            for template in architecture_recommendations["suggested_templates"]:
                recommendation_text += f"- {template}\n"
        
        if architecture_recommendations["tech_stack"]:
            tech_stack = architecture_recommendations["tech_stack"]
            recommendation_text += f"\n## Recommended Technology Stack:\n"
            recommendation_text += f"- Frontend: {', '.join(tech_stack.frontend)}\n"
            recommendation_text += f"- Backend: {', '.join(tech_stack.backend)}\n"
            recommendation_text += f"- Database: {', '.join(tech_stack.database)}\n"
            if tech_stack.cache:
                recommendation_text += f"- Cache: {', '.join(tech_stack.cache)}\n"
            if tech_stack.devops:
                recommendation_text += f"- DevOps: {', '.join(tech_stack.devops)}\n"
        
        if architecture_recommendations["considerations"]:
            recommendation_text += f"\n## Key Architectural Considerations:\n"
            for consideration in architecture_recommendations["considerations"]:
                recommendation_text += f"- {consideration}\n"
        
        # Create enhanced prompt
        enhanced_prompt = MASTER_PLANNER_PROMPT.replace("{project_brief}", project_brief)
        
        # Insert recommendations after the project brief section
        if recommendation_text:
            insert_point = enhanced_prompt.find("## Required Output Format:")
            if insert_point > 0:
                enhanced_prompt = (
                    enhanced_prompt[:insert_point] + 
                    recommendation_text + 
                    "\n" + 
                    enhanced_prompt[insert_point:]
                )
        
        return enhanced_prompt
    
    def _generate_smart_tasks(self, project_type: str, tech_stack: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate smart tasks using the task optimizer."""
        # Use task optimizer to generate optimized tasks
        optimized_tasks = self.task_optimizer.generate_optimized_tasks(project_type, tech_stack)
        
        # Add project-specific tasks based on project type
        additional_tasks = []
        
        if project_type == "e-commerce":
            additional_tasks.extend([
                {
                    "id": "implement-product-catalog",
                    "description": "Create product catalog with search and filtering capabilities",
                    "type": "backend",
                    "dependencies": ["create-api-routes"],
                    "project_area": "backend",
                    "files_to_create_or_modify": ["backend/src/modules/products/"],
                    "estimated_hours": 4.0
                },
                {
                    "id": "implement-shopping-cart",
                    "description": "Implement shopping cart functionality with session management",
                    "type": "backend",
                    "dependencies": ["implement-auth-system"],
                    "project_area": "backend",
                    "files_to_create_or_modify": ["backend/src/modules/cart/"],
                    "estimated_hours": 3.0
                },
                {
                    "id": "create-product-ui",
                    "description": "Build product listing and detail pages",
                    "type": "frontend",
                    "dependencies": ["setup-ui-components", "implement-product-catalog"],
                    "project_area": "frontend",
                    "files_to_create_or_modify": ["frontend/src/app/products/"],
                    "estimated_hours": 4.0
                }
            ])
        elif project_type == "realtime":
            additional_tasks.extend([
                {
                    "id": "setup-websocket-server",
                    "description": "Configure WebSocket server with Socket.io",
                    "type": "backend",
                    "dependencies": ["setup-backend-server"],
                    "project_area": "backend",
                    "files_to_create_or_modify": ["backend/src/websocket/"],
                    "estimated_hours": 3.0
                },
                {
                    "id": "implement-realtime-features",
                    "description": "Implement real-time event handling and broadcasting",
                    "type": "backend",
                    "dependencies": ["setup-websocket-server"],
                    "project_area": "backend",
                    "files_to_create_or_modify": ["backend/src/events/"],
                    "estimated_hours": 4.0
                }
            ])
        
        # Optimize all tasks together
        all_tasks = optimized_tasks + additional_tasks
        return self.task_optimizer.optimize_task_dependencies(all_tasks)