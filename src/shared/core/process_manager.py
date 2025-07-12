"""Process Manager - 중앙 집중식 child process 관리 시스템"""

import asyncio
import logging
import signal
import psutil
import os
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import weakref
import atexit

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """프로세스 상태"""
    RUNNING = "running"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"


@dataclass
class ManagedProcess:
    """관리되는 프로세스 정보"""
    process: asyncio.subprocess.Process
    pid: int
    command: str
    created_at: datetime
    timeout: Optional[float]
    state: ProcessState
    creator: str  # 프로세스를 생성한 모듈/함수명
    
    @property
    def age_seconds(self) -> float:
        """프로세스 생성 후 경과 시간"""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def is_alive(self) -> bool:
        """프로세스가 살아있는지 확인"""
        return self.process.returncode is None


class ProcessManager:
    """중앙 집중식 프로세스 관리자"""
    
    def __init__(self, max_processes: int = 50, cleanup_interval: int = 30):
        self.max_processes = max_processes
        self.cleanup_interval = cleanup_interval
        
        # 프로세스 추적
        self.managed_processes: Dict[int, ManagedProcess] = {}
        self.process_history: List[Dict] = []
        
        # 통계
        self.stats = {
            'total_created': 0,
            'total_terminated': 0,
            'total_failed': 0,
            'total_timeouts': 0,
            'max_concurrent': 0
        }
        
        # 모니터링
        self.monitoring_active = False
        self.monitoring_task = None
        self.shutdown_event = asyncio.Event()
        
        # 종료 핸들러 등록
        atexit.register(self._emergency_cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    async def create_subprocess_exec(self, *args, creator: str = "unknown", 
                                   timeout: Optional[float] = None, **kwargs) -> asyncio.subprocess.Process:
        """안전한 subprocess 생성"""
        # 최대 프로세스 수 체크
        if len(self.managed_processes) >= self.max_processes:
            await self._cleanup_terminated_processes()
            if len(self.managed_processes) >= self.max_processes:
                raise RuntimeError(f"Maximum process limit ({self.max_processes}) reached")
        
        # 프로세스 생성
        try:
            process = await asyncio.create_subprocess_exec(*args, **kwargs)
            
            # 관리 대상에 추가
            managed_proc = ManagedProcess(
                process=process,
                pid=process.pid,
                command=' '.join(str(arg) for arg in args),
                created_at=datetime.now(),
                timeout=timeout,
                state=ProcessState.RUNNING,
                creator=creator
            )
            
            self.managed_processes[process.pid] = managed_proc
            self.stats['total_created'] += 1
            self.stats['max_concurrent'] = max(
                self.stats['max_concurrent'], 
                len(self.managed_processes)
            )
            
            logger.info(f"Created subprocess PID {process.pid} by {creator}")
            
            # 타임아웃 설정
            if timeout:
                asyncio.create_task(self._monitor_timeout(process.pid, timeout))
            
            return process
            
        except Exception as e:
            self.stats['total_failed'] += 1
            logger.error(f"Failed to create subprocess: {e}")
            raise
    
    async def create_subprocess_shell(self, command: str, creator: str = "unknown",
                                    timeout: Optional[float] = None, **kwargs) -> asyncio.subprocess.Process:
        """안전한 shell subprocess 생성"""
        # 최대 프로세스 수 체크
        if len(self.managed_processes) >= self.max_processes:
            await self._cleanup_terminated_processes()
            if len(self.managed_processes) >= self.max_processes:
                raise RuntimeError(f"Maximum process limit ({self.max_processes}) reached")
        
        try:
            process = await asyncio.create_subprocess_shell(command, **kwargs)
            
            # 관리 대상에 추가
            managed_proc = ManagedProcess(
                process=process,
                pid=process.pid,
                command=command,
                created_at=datetime.now(),
                timeout=timeout,
                state=ProcessState.RUNNING,
                creator=creator
            )
            
            self.managed_processes[process.pid] = managed_proc
            self.stats['total_created'] += 1
            self.stats['max_concurrent'] = max(
                self.stats['max_concurrent'], 
                len(self.managed_processes)
            )
            
            logger.info(f"Created shell subprocess PID {process.pid} by {creator}")
            
            # 타임아웃 설정
            if timeout:
                asyncio.create_task(self._monitor_timeout(process.pid, timeout))
            
            return process
            
        except Exception as e:
            self.stats['total_failed'] += 1
            logger.error(f"Failed to create shell subprocess: {e}")
            raise
    
    async def _monitor_timeout(self, pid: int, timeout: float):
        """프로세스 타임아웃 모니터링"""
        try:
            await asyncio.sleep(timeout)
            
            # 타임아웃 시 프로세스 강제 종료
            if pid in self.managed_processes:
                managed_proc = self.managed_processes[pid]
                if managed_proc.is_alive:
                    logger.warning(f"Process PID {pid} timed out after {timeout}s, terminating")
                    await self.terminate_process(pid, force=True)
                    self.stats['total_timeouts'] += 1
                    
        except asyncio.CancelledError:
            pass  # 프로세스가 정상 종료된 경우
    
    async def terminate_process(self, pid: int, force: bool = False, 
                              grace_period: float = 5.0) -> bool:
        """안전한 프로세스 종료"""
        if pid not in self.managed_processes:
            logger.warning(f"Process PID {pid} not found in managed processes")
            return False
        
        managed_proc = self.managed_processes[pid]
        process = managed_proc.process
        
        if not managed_proc.is_alive:
            logger.debug(f"Process PID {pid} already terminated")
            await self._mark_terminated(pid)
            return True
        
        try:
            managed_proc.state = ProcessState.TERMINATING
            
            if force:
                # 강제 종료
                process.kill()
                logger.info(f"Force killed process PID {pid}")
            else:
                # 우아한 종료 시도
                process.terminate()
                logger.info(f"Sent SIGTERM to process PID {pid}")
                
                try:
                    # grace period 대기
                    await asyncio.wait_for(process.wait(), timeout=grace_period)
                except asyncio.TimeoutError:
                    # grace period 초과 시 강제 종료
                    logger.warning(f"Process PID {pid} did not terminate gracefully, force killing")
                    process.kill()
            
            # 종료 대기
            await process.wait()
            await self._mark_terminated(pid)
            
            logger.info(f"Successfully terminated process PID {pid}")
            return True
            
        except Exception as e:
            logger.error(f"Error terminating process PID {pid}: {e}")
            managed_proc.state = ProcessState.FAILED
            return False
    
    async def _mark_terminated(self, pid: int):
        """프로세스를 종료된 것으로 표시"""
        if pid in self.managed_processes:
            managed_proc = self.managed_processes[pid]
            managed_proc.state = ProcessState.TERMINATED
            
            # 히스토리에 추가
            self.process_history.append({
                'pid': pid,
                'command': managed_proc.command,
                'creator': managed_proc.creator,
                'created_at': managed_proc.created_at.isoformat(),
                'terminated_at': datetime.now().isoformat(),
                'age_seconds': managed_proc.age_seconds,
                'state': managed_proc.state.value
            })
            
            # 관리 목록에서 제거
            del self.managed_processes[pid]
            self.stats['total_terminated'] += 1
    
    async def terminate_all_processes(self, grace_period: float = 5.0):
        """모든 관리되는 프로세스 종료"""
        logger.info(f"Terminating all {len(self.managed_processes)} managed processes")
        
        # 모든 프로세스에 SIGTERM 전송
        termination_tasks = []
        for pid in list(self.managed_processes.keys()):
            task = asyncio.create_task(self.terminate_process(pid, grace_period=grace_period))
            termination_tasks.append(task)
        
        # 모든 종료 작업 완료 대기
        if termination_tasks:
            await asyncio.gather(*termination_tasks, return_exceptions=True)
        
        logger.info("All managed processes terminated")
    
    async def cleanup_orphaned_processes(self):
        """고아 프로세스 정리"""
        orphaned_count = 0
        
        for pid in list(self.managed_processes.keys()):
            managed_proc = self.managed_processes[pid]
            
            # 프로세스가 실제로 존재하는지 확인
            try:
                psutil_proc = psutil.Process(pid)
                if not psutil_proc.is_running():
                    await self._mark_terminated(pid)
                    orphaned_count += 1
            except psutil.NoSuchProcess:
                # 프로세스가 이미 종료됨
                await self._mark_terminated(pid)
                orphaned_count += 1
        
        if orphaned_count > 0:
            logger.info(f"Cleaned up {orphaned_count} orphaned processes")
        
        return orphaned_count
    
    async def _cleanup_terminated_processes(self):
        """종료된 프로세스들 정리"""
        cleanup_count = 0
        
        for pid in list(self.managed_processes.keys()):
            managed_proc = self.managed_processes[pid]
            
            if not managed_proc.is_alive:
                await self._mark_terminated(pid)
                cleanup_count += 1
        
        if cleanup_count > 0:
            logger.debug(f"Cleaned up {cleanup_count} terminated processes")
    
    def get_process_stats(self) -> Dict:
        """프로세스 통계 반환"""
        running_processes = [p for p in self.managed_processes.values() if p.is_alive]
        
        # 생성자별 통계
        creator_stats = {}
        for proc in running_processes:
            creator_stats[proc.creator] = creator_stats.get(proc.creator, 0) + 1
        
        return {
            'current_processes': len(self.managed_processes),
            'running_processes': len(running_processes),
            'creator_breakdown': creator_stats,
            'stats': self.stats.copy(),
            'max_processes': self.max_processes,
            'oldest_process_age': max([p.age_seconds for p in running_processes], default=0)
        }
    
    def get_process_list(self) -> List[Dict]:
        """현재 프로세스 목록 반환"""
        return [
            {
                'pid': proc.pid,
                'command': proc.command[:100] + '...' if len(proc.command) > 100 else proc.command,
                'creator': proc.creator,
                'age_seconds': proc.age_seconds,
                'state': proc.state.value,
                'is_alive': proc.is_alive
            }
            for proc in self.managed_processes.values()
        ]
    
    async def start_monitoring(self):
        """프로세스 모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Process monitoring started")
    
    async def stop_monitoring(self):
        """프로세스 모니터링 중지"""
        self.monitoring_active = False
        self.shutdown_event.set()
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Process monitoring stopped")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # 종료된 프로세스 정리
                await self._cleanup_terminated_processes()
                
                # 고아 프로세스 정리
                await self.cleanup_orphaned_processes()
                
                # 오래된 프로세스 경고
                for proc in self.managed_processes.values():
                    if proc.age_seconds > 3600:  # 1시간 이상
                        logger.warning(f"Long-running process detected: PID {proc.pid} "
                                     f"({proc.age_seconds:.0f}s old) by {proc.creator}")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        logger.info(f"Received signal {signum}, initiating process cleanup")
        asyncio.create_task(self.terminate_all_processes())
    
    def _emergency_cleanup(self):
        """긴급 정리 (atexit 핸들러)"""
        logger.critical("Emergency process cleanup initiated")
        
        # 모든 관리되는 프로세스 강제 종료
        for proc in self.managed_processes.values():
            try:
                if proc.is_alive:
                    proc.process.kill()
                    logger.warning(f"Emergency killed process PID {proc.pid}")
            except Exception as e:
                logger.error(f"Error in emergency cleanup for PID {proc.pid}: {e}")


# 전역 프로세스 매니저
global_process_manager = ProcessManager()


async def create_managed_subprocess_exec(*args, creator: str = "unknown", 
                                        timeout: Optional[float] = None, **kwargs):
    """전역 프로세스 매니저를 사용한 subprocess 생성"""
    return await global_process_manager.create_subprocess_exec(
        *args, creator=creator, timeout=timeout, **kwargs
    )


async def create_managed_subprocess_shell(command: str, creator: str = "unknown",
                                        timeout: Optional[float] = None, **kwargs):
    """전역 프로세스 매니저를 사용한 shell subprocess 생성"""
    return await global_process_manager.create_subprocess_shell(
        command, creator=creator, timeout=timeout, **kwargs
    )


async def terminate_managed_process(pid: int, force: bool = False):
    """관리되는 프로세스 안전 종료"""
    return await global_process_manager.terminate_process(pid, force=force)


async def initialize_process_manager():
    """프로세스 매니저 초기화"""
    await global_process_manager.start_monitoring()


async def shutdown_process_manager():
    """프로세스 매니저 종료"""
    await global_process_manager.terminate_all_processes()
    await global_process_manager.stop_monitoring()