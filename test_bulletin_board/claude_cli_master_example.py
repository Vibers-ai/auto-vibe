"""
Claude CLI as Master - Example Implementation
"""
import json
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    description: str
    files_to_create_or_modify: list
    type: str
    project_area: str

class ClaudeCliMaster:
    """Claude CLI acting as Master supervisor"""
    
    def __init__(self, claude_cli_path: str = "/usr/bin/claude"):
        self.claude_cli_path = claude_cli_path
        self.planning_system_prompt = """You are a master software architect and planner.
Your role is to analyze tasks, create execution plans, and evaluate results.
Always respond with structured JSON when requested."""
        
        self.evaluation_system_prompt = """You are a code quality evaluator.
Assess implementations for correctness, completeness, and best practices.
Provide constructive feedback and success determination."""
    
    async def analyze_and_plan(self, task: Task, workspace_context: str) -> Dict[str, Any]:
        """Use Claude CLI to analyze task and create execution plan"""
        
        prompt = f"""
{self.planning_system_prompt}

Analyze this task and create a detailed execution plan.

TASK DETAILS:
- ID: {task.id}
- Description: {task.description}
- Files to create/modify: {', '.join(task.files_to_create_or_modify)}
- Type: {task.type}
- Area: {task.project_area}

WORKSPACE CONTEXT:
{workspace_context}

Create a JSON execution plan with:
1. situation_analysis - Current state assessment
2. execution_strategy - How to approach the task
3. step_by_step_plan - Detailed implementation steps
4. success_criteria - How to verify completion
5. potential_challenges - What might go wrong

Respond with valid JSON only.
"""
        
        # Execute Claude CLI
        result = await self._execute_claude_cli(prompt)
        
        # Parse JSON response
        try:
            plan = json.loads(result)
            return plan
        except json.JSONDecodeError:
            # Extract JSON from response if wrapped in markdown
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            raise
    
    async def evaluate_result(self, task: Task, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Use Claude CLI to evaluate execution results"""
        
        prompt = f"""
{self.evaluation_system_prompt}

Evaluate the execution result for this task.

TASK: {task.id} - {task.description}

EXECUTION RESULT:
{json.dumps(execution_result, indent=2)}

Evaluate based on:
1. Were all required files created?
2. Is the implementation complete and functional?
3. Does it follow best practices?
4. Are there any issues or improvements needed?

Provide JSON evaluation with:
- success: boolean
- completeness_score: 0-100
- quality_score: 0-100  
- feedback: string with specific observations
- improvements: list of suggested improvements
- ready_for_production: boolean

Respond with valid JSON only.
"""
        
        result = await self._execute_claude_cli(prompt)
        
        try:
            evaluation = json.loads(result)
            return evaluation
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            raise
    
    async def _execute_claude_cli(self, prompt: str) -> str:
        """Execute Claude CLI with prompt"""
        import tempfile
        import subprocess
        
        # Write prompt to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name
        
        try:
            # Execute Claude CLI
            cmd = f"cat {prompt_file} | {self.claude_cli_path} --print"
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Claude CLI failed: {stderr.decode()}")
            
            return stdout.decode()
            
        finally:
            # Clean up temp file
            import os
            os.unlink(prompt_file)


# Example usage
async def main():
    # Create task
    task = Task(
        id="create-api-endpoint",
        description="Create REST API endpoint for user management",
        files_to_create_or_modify=["api/users.py", "api/schemas.py"],
        type="backend",
        project_area="api"
    )
    
    # Initialize Claude CLI Master
    master = ClaudeCliMaster()
    
    # Phase 1: Planning
    print("Phase 1: Planning with Claude CLI Master...")
    plan = await master.analyze_and_plan(task, "Current workspace has FastAPI setup")
    print(f"Plan created: {json.dumps(plan, indent=2)}")
    
    # Phase 2: Execution (would be done by worker)
    execution_result = {
        "success": True,
        "files_created": ["api/users.py", "api/schemas.py"],
        "summary": "Created user management endpoints with CRUD operations"
    }
    
    # Phase 3: Evaluation
    print("\nPhase 3: Evaluation with Claude CLI Master...")
    evaluation = await master.evaluate_result(task, execution_result)
    print(f"Evaluation: {json.dumps(evaluation, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())