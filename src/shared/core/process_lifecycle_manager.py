"""Process Lifecycle Manager - Handles process completion signals and idle timeouts."""

import os
import time
import signal
import psutil
import logging
from typing import Dict, Optional, Set, Any
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class ProcessLifecycleManager:
    """Manages process lifecycles with completion signals and idle timeouts."""
    
    def __init__(self, idle_timeout_seconds: int = 300):
        """
        Initialize process lifecycle manager.
        
        Args:
            idle_timeout_seconds: Time before considering a process idle (default 5 minutes)
        """
        self.idle_timeout_seconds = idle_timeout_seconds
        self.processes: Dict[int, Dict[str, Any]] = {}
        self.completion_signals: Set[str] = {
            "TASK IMPLEMENTATION COMPLETE",
            "TASK FINISHED SUCCESSFULLY",
            "WORK COMPLETED",
            "IMPLEMENTATION COMPLETE",
            "ALL DONE",
            "TASK COMPLETE",
            "PROCESS COMPLETE"
        }
        self._monitor_thread = None
        self._running = False
        self._lock = threading.Lock()
    
    def register_process(self, pid: int, name: str = "", parent_pid: Optional[int] = None):
        """Register a process for lifecycle management."""
        with self._lock:
            self.processes[pid] = {
                'name': name,
                'parent_pid': parent_pid,
                'start_time': time.time(),
                'last_activity': time.time(),
                'output_lines': [],
                'completed': False,
                'completion_time': None,
                'termination_reason': None
            }
            logger.info(f"Registered process {pid} ({name}) for lifecycle management")
    
    def update_activity(self, pid: int, output_line: Optional[str] = None):
        """Update process activity timestamp and check for completion signals."""
        with self._lock:
            if pid not in self.processes:
                return
            
            self.processes[pid]['last_activity'] = time.time()
            
            if output_line:
                # Store recent output (keep last 100 lines)
                self.processes[pid]['output_lines'].append(output_line)
                if len(self.processes[pid]['output_lines']) > 100:
                    self.processes[pid]['output_lines'].pop(0)
                
                # Check for completion signals
                output_upper = output_line.upper().strip()
                for signal in self.completion_signals:
                    if signal in output_upper:
                        logger.info(f"Process {pid} sent completion signal: {signal}")
                        self._mark_process_completed(pid, f"Completion signal: {signal}")
                        # Schedule termination after brief delay
                        threading.Timer(2.0, self._terminate_process, args=[pid, "completion"]).start()
                        return
    
    def _mark_process_completed(self, pid: int, reason: str):
        """Mark a process as completed."""
        if pid in self.processes:
            self.processes[pid]['completed'] = True
            self.processes[pid]['completion_time'] = time.time()
            self.processes[pid]['termination_reason'] = reason
    
    def _terminate_process(self, pid: int, reason: str = "unknown"):
        """Terminate a process and its children."""
        try:
            process = psutil.Process(pid)
            
            # Get all children recursively
            children = process.children(recursive=True)
            
            # Terminate children first
            for child in children:
                try:
                    logger.info(f"Terminating child process {child.pid} of {pid}")
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Give children time to terminate
            time.sleep(1)
            
            # Force kill any remaining children
            for child in children:
                try:
                    if child.is_running():
                        child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Finally terminate the parent
            try:
                logger.info(f"Terminating process {pid} (reason: {reason})")
                process.terminate()
                time.sleep(1)
                if process.is_running():
                    process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            # Remove from tracking
            with self._lock:
                if pid in self.processes:
                    del self.processes[pid]
                    
        except psutil.NoSuchProcess:
            logger.info(f"Process {pid} already terminated")
            with self._lock:
                if pid in self.processes:
                    del self.processes[pid]
        except Exception as e:
            logger.error(f"Error terminating process {pid}: {e}")
    
    def check_idle_processes(self):
        """Check for idle processes and terminate them."""
        current_time = time.time()
        processes_to_terminate = []
        
        with self._lock:
            for pid, info in self.processes.items():
                if info['completed']:
                    continue
                    
                idle_time = current_time - info['last_activity']
                if idle_time > self.idle_timeout_seconds:
                    processes_to_terminate.append((pid, f"Idle for {idle_time:.1f} seconds"))
        
        # Terminate idle processes outside of lock
        for pid, reason in processes_to_terminate:
            logger.warning(f"Process {pid} is idle, terminating: {reason}")
            self._mark_process_completed(pid, reason)
            self._terminate_process(pid, "idle_timeout")
    
    def start_monitoring(self):
        """Start the background monitoring thread."""
        if self._running:
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Process lifecycle monitoring started")
    
    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Process lifecycle monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.check_idle_processes()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def get_process_status(self) -> Dict[int, Dict[str, Any]]:
        """Get status of all tracked processes."""
        with self._lock:
            status = {}
            current_time = time.time()
            
            for pid, info in self.processes.items():
                idle_time = current_time - info['last_activity']
                runtime = current_time - info['start_time']
                
                status[pid] = {
                    'name': info['name'],
                    'runtime_seconds': runtime,
                    'idle_seconds': idle_time,
                    'completed': info['completed'],
                    'termination_reason': info['termination_reason'],
                    'is_idle': idle_time > self.idle_timeout_seconds
                }
                
                # Check if process is still alive
                try:
                    psutil.Process(pid)
                    status[pid]['alive'] = True
                except psutil.NoSuchProcess:
                    status[pid]['alive'] = False
            
            return status
    
    def cleanup_completed_processes(self):
        """Clean up records of completed processes."""
        with self._lock:
            pids_to_remove = []
            for pid, info in self.processes.items():
                if info['completed'] and info['completion_time']:
                    # Keep completed processes for 1 minute for debugging
                    if time.time() - info['completion_time'] > 60:
                        pids_to_remove.append(pid)
            
            for pid in pids_to_remove:
                del self.processes[pid]
                logger.debug(f"Cleaned up completed process record: {pid}")
    
    def terminate_all_processes(self, reason: str = "shutdown"):
        """Terminate all tracked processes."""
        with self._lock:
            pids = list(self.processes.keys())
        
        for pid in pids:
            self._terminate_process(pid, reason)
    
    def add_completion_signal(self, signal: str):
        """Add a new completion signal pattern."""
        self.completion_signals.add(signal.upper())
        logger.info(f"Added completion signal: {signal}")
    
    def set_idle_timeout(self, seconds: int):
        """Update the idle timeout duration."""
        self.idle_timeout_seconds = seconds
        logger.info(f"Updated idle timeout to {seconds} seconds")