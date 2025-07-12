"""File utility functions for VIBE."""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def ensure_directory(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Read a text file and return its contents."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def write_text_file(file_path: str, content: str, encoding: str = 'utf-8') -> None:
    """Write content to a text file."""
    # Ensure parent directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)


def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read a JSON file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """Write data to a JSON file."""
    # Ensure parent directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def copy_file(src: str, dst: str) -> None:
    """Copy a file from source to destination."""
    # Ensure destination directory exists
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_directory(src: str, dst: str) -> None:
    """Copy a directory and all its contents."""
    shutil.copytree(src, dst, dirs_exist_ok=True)


def delete_file(file_path: str) -> bool:
    """Delete a file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


def delete_directory(dir_path: str) -> bool:
    """Delete a directory and all its contents."""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting directory {dir_path}: {e}")
        return False


def list_files(directory: str, pattern: Optional[str] = None, recursive: bool = True) -> List[Path]:
    """List files in a directory, optionally filtered by pattern."""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    if recursive:
        if pattern:
            return list(dir_path.rglob(pattern))
        else:
            return [p for p in dir_path.rglob('*') if p.is_file()]
    else:
        if pattern:
            return list(dir_path.glob(pattern))
        else:
            return [p for p in dir_path.iterdir() if p.is_file()]


def get_file_size(file_path: str) -> int:
    """Get the size of a file in bytes."""
    return os.path.getsize(file_path)


def get_file_extension(file_path: str) -> str:
    """Get the file extension (lowercase, without dot)."""
    return Path(file_path).suffix.lower().lstrip('.')


def create_temp_directory(prefix: str = 'vibe_temp_') -> Path:
    """Create a temporary directory."""
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return Path(temp_dir)