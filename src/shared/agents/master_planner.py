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

logger = logging.getLogger(__name__)
console = Console()


MASTER_PLANNER_PROMPT = """You are an expert software architect and project planner. Your task is to analyze the provided project documentation and create a comprehensive technical plan.

Given the ProjectBrief below, you must:

1. **Define the High-Level Architecture**
   - Choose the technology stack (e.g., Backend: Python/FastAPI, Frontend: React/TypeScript)
   - Define the system architecture (monolithic, microservices, etc.)
   - Explain your technology choices

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

5. **Create Detailed Tasks**
   - Break down the implementation into atomic, verifiable tasks
   - Each task should be completable in 1-4 hours
   - Include clear dependencies between tasks
   - Define acceptance criteria for each task

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

Remember:
- Tasks should be atomic and independently verifiable
- Include all necessary technical details for implementation
- Consider proper build order and dependencies
- Each task should have clear acceptance criteria
- Use descriptive, unique IDs for tasks

Think step by step and be comprehensive in your analysis."""


class MasterPlannerAgent:
    """Agent responsible for creating the master plan and tasks from ProjectBrief."""
    
    def __init__(self, config: Config):
        self.config = config
        self.gemini_model = None
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
        
        try:
            prompt = MASTER_PLANNER_PROMPT.replace("{project_brief}", project_brief)
            console.print(f"[cyan]🎯 Prompt template formatted successfully[/cyan]")
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
            
            validated_tasks.append(task)
        
        tasks_json['tasks'] = validated_tasks
        tasks_json['total_tasks'] = len(validated_tasks)
        
        return tasks_json
    
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