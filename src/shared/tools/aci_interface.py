"""Agent-Computer Interface tools for VIBE."""

import os
import subprocess
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

from shared.core.process_manager import create_managed_subprocess_shell

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""
    return_code: int
    stdout: str
    stderr: str
    success: bool
    
    @classmethod
    def from_process(cls, process_result: subprocess.CompletedProcess) -> 'CommandResult':
        return cls(
            return_code=process_result.returncode,
            stdout=process_result.stdout.decode('utf-8') if process_result.stdout else '',
            stderr=process_result.stderr.decode('utf-8') if process_result.stderr else '',
            success=process_result.returncode == 0
        )


@dataclass
class TestResult:
    """Result of a test execution."""
    success: bool
    output: str
    failed_tests: List[str]
    passed_tests: List[str]
    total_tests: int


class ACIInterface:
    """Agent-Computer Interface for code execution environments."""
    
    def __init__(self, timeout: int = 300):
        self.timeout = timeout
    
    # File Operations
    async def read_file(self, file_path: str) -> str:
        """Read contents of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")
    
    async def write_file(self, file_path: str, content: str, create_dirs: bool = True) -> bool:
        """Write content to a file."""
        try:
            if create_dirs:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return False
    
    async def append_to_file(self, file_path: str, content: str) -> bool:
        """Append content to a file."""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error appending to file {file_path}: {e}")
            return False
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file."""
        try:
            os.remove(file_path)
            return True
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(file_path)
    
    async def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """List files in a directory."""
        try:
            path = Path(directory)
            if pattern == "*":
                files = [str(f) for f in path.rglob('*') if f.is_file()]
            else:
                files = [str(f) for f in path.rglob(pattern) if f.is_file()]
            return files
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {e}")
            return []
    
    async def get_directory_structure(self, directory: str, max_depth: int = 3) -> str:
        """Get a tree-like directory structure."""
        try:
            result = await self.run_command(directory, f"tree -L {max_depth} || find . -maxdepth {max_depth} -type d")
            return result.stdout if result.success else "Unable to get directory structure"
        except Exception as e:
            logger.error(f"Error getting directory structure: {e}")
            return "Error getting directory structure"
    
    # Command Execution
    async def run_command(self, working_dir: str, command: str, 
                         capture_output: bool = True) -> CommandResult:
        """Run a shell command in the specified directory."""
        try:
            logger.info(f"Running command in {working_dir}: {command}")
            
            process = await create_managed_subprocess_shell(
                command,
                creator="aci_interface.run_command",
                timeout=self.timeout,
                cwd=working_dir,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
            
            return CommandResult(
                return_code=process.returncode,
                stdout=stdout.decode('utf-8') if stdout else '',
                stderr=stderr.decode('utf-8') if stderr else '',
                success=process.returncode == 0
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Command timed out: {command}")
            return CommandResult(
                return_code=-1,
                stdout='',
                stderr=f'Command timed out after {self.timeout} seconds',
                success=False
            )
        except Exception as e:
            logger.error(f"Error running command {command}: {e}")
            return CommandResult(
                return_code=-1,
                stdout='',
                stderr=str(e),
                success=False
            )
    
    # Package Management
    async def install_python_package(self, working_dir: str, package: str) -> CommandResult:
        """Install a Python package using pip."""
        return await self.run_command(working_dir, f"pip install {package}")
    
    async def install_npm_package(self, working_dir: str, package: str, 
                                 dev: bool = False) -> CommandResult:
        """Install an npm package."""
        flag = "--save-dev" if dev else "--save"
        return await self.run_command(working_dir, f"npm install {flag} {package}")
    
    # Testing
    async def run_test(self, working_dir: str, test_file: str, 
                      test_function: Optional[str] = None) -> TestResult:
        """Run tests using pytest or npm test."""
        try:
            # Determine test runner based on file extension
            if test_file.endswith('.py'):
                return await self._run_pytest(working_dir, test_file, test_function)
            elif test_file.endswith(('.js', '.ts', '.jsx', '.tsx')):
                return await self._run_npm_test(working_dir, test_file)
            else:
                raise ValueError(f"Unsupported test file type: {test_file}")
                
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return TestResult(
                success=False,
                output=str(e),
                failed_tests=[],
                passed_tests=[],
                total_tests=0
            )
    
    async def _run_pytest(self, working_dir: str, test_file: str, 
                         test_function: Optional[str] = None) -> TestResult:
        """Run pytest for Python tests."""
        command = f"python -m pytest {test_file}"
        if test_function:
            command += f"::{test_function}"
        command += " -v --tb=short"
        
        result = await self.run_command(working_dir, command)
        
        # Parse pytest output
        failed_tests = []
        passed_tests = []
        total_tests = 0
        
        if result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if '::' in line and 'PASSED' in line:
                    passed_tests.append(line.split('::')[1].split()[0])
                    total_tests += 1
                elif '::' in line and 'FAILED' in line:
                    failed_tests.append(line.split('::')[1].split()[0])
                    total_tests += 1
        
        return TestResult(
            success=result.success and len(failed_tests) == 0,
            output=result.stdout,
            failed_tests=failed_tests,
            passed_tests=passed_tests,
            total_tests=total_tests
        )
    
    async def _run_npm_test(self, working_dir: str, test_file: str) -> TestResult:
        """Run npm test for JavaScript/TypeScript tests."""
        # Try to run specific test file
        command = f"npm test -- {test_file}"
        result = await self.run_command(working_dir, command)
        
        # If that fails, try running all tests
        if not result.success:
            result = await self.run_command(working_dir, "npm test")
        
        # Parse npm test output (basic implementation)
        failed_tests = []
        passed_tests = []
        total_tests = 0
        
        # This is a simplified parser - real implementation would need more sophisticated parsing
        if result.stdout:
            if 'Tests:' in result.stdout:
                # Extract test summary
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'passed' in line.lower():
                        try:
                            passed_count = int([s for s in line.split() if s.isdigit()][0])
                            total_tests += passed_count
                        except (IndexError, ValueError):
                            pass
                    elif 'failed' in line.lower():
                        try:
                            failed_count = int([s for s in line.split() if s.isdigit()][0])
                            total_tests += failed_count
                        except (IndexError, ValueError):
                            pass
        
        return TestResult(
            success=result.success,
            output=result.stdout,
            failed_tests=failed_tests,
            passed_tests=passed_tests,
            total_tests=total_tests
        )
    
    # Code Quality
    async def run_linter(self, working_dir: str, files: List[str], 
                        linter: str = "auto") -> CommandResult:
        """Run code linting."""
        if linter == "auto":
            # Auto-detect linter based on files
            if any(f.endswith('.py') for f in files):
                linter = "ruff"
            elif any(f.endswith(('.js', '.ts', '.jsx', '.tsx')) for f in files):
                linter = "eslint"
        
        if linter == "ruff":
            command = f"ruff check {' '.join(files)}"
        elif linter == "eslint":
            command = f"npx eslint {' '.join(files)}"
        elif linter == "pylint":
            command = f"pylint {' '.join(files)}"
        else:
            raise ValueError(f"Unsupported linter: {linter}")
        
        return await self.run_command(working_dir, command)
    
    async def format_code(self, working_dir: str, files: List[str], 
                         formatter: str = "auto") -> CommandResult:
        """Format code using appropriate formatter."""
        if formatter == "auto":
            # Auto-detect formatter based on files
            if any(f.endswith('.py') for f in files):
                formatter = "black"
            elif any(f.endswith(('.js', '.ts', '.jsx', '.tsx')) for f in files):
                formatter = "prettier"
        
        if formatter == "black":
            command = f"black {' '.join(files)}"
        elif formatter == "prettier":
            command = f"npx prettier --write {' '.join(files)}"
        elif formatter == "ruff":
            command = f"ruff format {' '.join(files)}"
        else:
            raise ValueError(f"Unsupported formatter: {formatter}")
        
        return await self.run_command(working_dir, command)
    
    # Project Setup
    async def initialize_python_project(self, working_dir: str, 
                                       project_name: str) -> CommandResult:
        """Initialize a Python project structure."""
        commands = [
            f"mkdir -p {project_name}",
            f"cd {project_name} && python -m venv venv",
            f"cd {project_name} && echo '{project_name}' > README.md",
            f"cd {project_name} && touch requirements.txt",
            f"cd {project_name} && mkdir -p src tests"
        ]
        
        results = []
        for cmd in commands:
            result = await self.run_command(working_dir, cmd)
            results.append(result)
            if not result.success:
                break
        
        # Return the last result or combined result
        return results[-1] if results else CommandResult(1, '', 'No commands executed', False)
    
    async def initialize_node_project(self, working_dir: str, 
                                     project_name: str) -> CommandResult:
        """Initialize a Node.js project structure."""
        commands = [
            f"mkdir -p {project_name}",
            f"cd {project_name} && npm init -y",
            f"cd {project_name} && echo '# {project_name}' > README.md",
            f"cd {project_name} && mkdir -p src tests"
        ]
        
        results = []
        for cmd in commands:
            result = await self.run_command(working_dir, cmd)
            results.append(result)
            if not result.success:
                break
        
        return results[-1] if results else CommandResult(1, '', 'No commands executed', False)
    
    # Git Operations
    async def git_init(self, working_dir: str) -> CommandResult:
        """Initialize a git repository."""
        return await self.run_command(working_dir, "git init")
    
    async def git_add(self, working_dir: str, files: str = ".") -> CommandResult:
        """Add files to git staging."""
        return await self.run_command(working_dir, f"git add {files}")
    
    async def git_commit(self, working_dir: str, message: str) -> CommandResult:
        """Create a git commit."""
        return await self.run_command(working_dir, f'git commit -m "{message}"')
    
    async def git_status(self, working_dir: str) -> CommandResult:
        """Get git status."""
        return await self.run_command(working_dir, "git status")
    
    # Environment Information
    async def get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        info = {}
        
        try:
            # Python version
            result = await self.run_command(".", "python --version")
            info['python_version'] = result.stdout.strip()
            
            # Node version
            result = await self.run_command(".", "node --version")
            info['node_version'] = result.stdout.strip()
            
            # npm version
            result = await self.run_command(".", "npm --version")
            info['npm_version'] = result.stdout.strip()
            
            # Git version
            result = await self.run_command(".", "git --version")
            info['git_version'] = result.stdout.strip()
            
            # OS info
            result = await self.run_command(".", "uname -a")
            info['os_info'] = result.stdout.strip()
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
        
        return info