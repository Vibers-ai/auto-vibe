"""Configuration management for VIBE."""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    """Central configuration for the VIBE system."""
    
    # API Keys
    gemini_api_key: str
    anthropic_api_key: str
    
    # Model Configuration
    gemini_model: str = "gemini-1.5-flash"
    claude_model: str = "claude-3-5-sonnet-20241022"
    
    # Execution Configuration
    max_retries: int = 3
    task_timeout: int = 1800  # seconds (30 minutes)
    parallel_tasks: int = 1  # Single task for debugging
    
    # Docker Configuration
    docker_image_name: str = "vibe-sandbox"
    docker_container_prefix: str = "vibe-task-"
    
    # Paths
    docs_path: str = "docs"
    output_path: str = "output"
    sandbox_path: str = "sandbox"
    
    # Claude CLI Configuration
    claude_cli_skip_permissions: bool = True  # Skip permissions for automation
    claude_cli_path: Optional[str] = None  # Custom CLI path if needed
    claude_cli_use_npx: bool = False  # Use npx claude if true
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "vibe.log"
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'Config':
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Required API keys
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        return cls(
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            claude_model=os.getenv("CLAUDE_MODEL", cls.claude_model),
            max_retries=int(os.getenv("MAX_RETRIES", cls.max_retries)),
            task_timeout=int(os.getenv("TASK_TIMEOUT", cls.task_timeout)),
            parallel_tasks=int(os.getenv("PARALLEL_TASKS", cls.parallel_tasks)),
            docker_image_name=os.getenv("DOCKER_IMAGE_NAME", cls.docker_image_name),
            docker_container_prefix=os.getenv("DOCKER_CONTAINER_PREFIX", cls.docker_container_prefix),
            claude_cli_skip_permissions=os.getenv("CLAUDE_CLI_SKIP_PERMISSIONS", "true").lower() == "true",
            claude_cli_path=os.getenv("CLAUDE_CLI_PATH"),
            claude_cli_use_npx=os.getenv("CLAUDE_CLI_USE_NPX", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            log_file=os.getenv("LOG_FILE", cls.log_file),
        )
    
    def validate(self) -> None:
        """Validate the configuration."""
        # Check paths exist
        for path_attr in ['docs_path', 'output_path', 'sandbox_path']:
            path = Path(getattr(self, path_attr))
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)