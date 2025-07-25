"""Shared state management for task execution across CLI and SDK implementations."""

import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional


class ExecutorState:
    """Persistent state management for the task executor."""
    
    def __init__(self, db_path: str = "executor_state.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for state persistence."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create execution sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_sessions (
                    session_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL DEFAULT 'in_progress',
                    tasks_file_path TEXT NOT NULL,
                    total_tasks INTEGER NOT NULL,
                    completed_tasks INTEGER DEFAULT 0,
                    failed_tasks INTEGER DEFAULT 0
                )
            """)
            
            # Create task executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_executions (
                    session_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    attempts INTEGER DEFAULT 0,
                    last_error TEXT,
                    output_log TEXT,
                    PRIMARY KEY (session_id, task_id),
                    FOREIGN KEY (session_id) REFERENCES execution_sessions(session_id)
                )
            """)
            
            conn.commit()
    
    def create_session(self, project_id: str, tasks_file_path: str, total_tasks: int) -> str:
        """Create a new execution session."""
        session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO execution_sessions 
                (session_id, project_id, started_at, tasks_file_path, total_tasks)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, project_id, datetime.now().isoformat(), tasks_file_path, total_tasks))
            conn.commit()
        
        return session_id
    
    def update_session_status(self, session_id: str, status: str, completed_tasks: int = None, failed_tasks: int = None):
        """Update session status and progress."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if status == 'completed':
                cursor.execute("""
                    UPDATE execution_sessions 
                    SET status = ?, completed_at = ?, completed_tasks = ?, failed_tasks = ?
                    WHERE session_id = ?
                """, (status, datetime.now().isoformat(), completed_tasks, failed_tasks, session_id))
            else:
                cursor.execute("""
                    UPDATE execution_sessions 
                    SET status = ?, completed_tasks = ?, failed_tasks = ?
                    WHERE session_id = ?
                """, (status, completed_tasks, failed_tasks, session_id))
            
            conn.commit()
    
    def update_task_status(self, session_id: str, task_id: str, status: str, 
                          error: str = None, output: str = None):
        """Update task execution status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if status == 'in_progress':
                cursor.execute("""
                    INSERT OR REPLACE INTO task_executions 
                    (session_id, task_id, status, started_at, attempts)
                    VALUES (?, ?, ?, ?, COALESCE((SELECT attempts FROM task_executions WHERE session_id = ? AND task_id = ?), 0) + 1)
                """, (session_id, task_id, status, datetime.now().isoformat(), session_id, task_id))
            elif status in ['completed', 'failed']:
                cursor.execute("""
                    UPDATE task_executions 
                    SET status = ?, completed_at = ?, last_error = ?, output_log = ?
                    WHERE session_id = ? AND task_id = ?
                """, (status, datetime.now().isoformat(), error, output, session_id, task_id))
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO task_executions 
                    (session_id, task_id, status)
                    VALUES (?, ?, ?)
                """, (session_id, task_id, status))
            
            conn.commit()
    
    def get_task_status(self, session_id: str, task_id: str) -> Optional[str]:
        """Get current status of a task."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status FROM task_executions 
                WHERE session_id = ? AND task_id = ?
            """, (session_id, task_id))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """Get execution progress for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute("""
                SELECT project_id, started_at, status, total_tasks, completed_tasks, failed_tasks
                FROM execution_sessions WHERE session_id = ?
            """, (session_id,))
            
            session_result = cursor.fetchone()
            if not session_result:
                return {}
            
            # Get task statuses
            cursor.execute("""
                SELECT task_id, status, attempts, last_error
                FROM task_executions WHERE session_id = ?
            """, (session_id,))
            
            task_results = cursor.fetchall()
            
            return {
                'session_id': session_id,
                'project_id': session_result[0],
                'started_at': session_result[1],
                'status': session_result[2],
                'total_tasks': session_result[3],
                'completed_tasks': session_result[4] or 0,
                'failed_tasks': session_result[5] or 0,
                'tasks': {
                    task_id: {
                        'status': status,
                        'attempts': attempts,
                        'last_error': last_error
                    }
                    for task_id, status, attempts, last_error in task_results
                }
            }