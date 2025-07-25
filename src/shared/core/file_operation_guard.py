"""File Operation Guard - 파일 동시성 제어 및 충돌 방지 시스템"""

import os
import fcntl
import time
import asyncio
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Dict, Set, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


class ResourceBusyError(Exception):
    """리소스가 다른 작업에 의해 사용 중일 때 발생하는 예외"""
    pass


class FileLockTimeoutError(Exception):
    """파일 락 획득 타임아웃 시 발생하는 예외"""
    pass


@dataclass
class FileLockInfo:
    """파일 락 정보"""
    file_path: str
    task_id: str
    worker_id: str
    locked_at: datetime
    lock_type: str  # 'read', 'write', 'exclusive'


class FileOperationGuard:
    """파일 작업의 동시성을 제어하고 충돌을 방지하는 가드 시스템"""
    
    def __init__(self, lock_timeout: int = 60):
        self.lock_timeout = lock_timeout
        self.active_locks: Dict[str, FileLockInfo] = {}
        self.lock_queue: Dict[str, list] = {}
        self.cleanup_interval = 300  # 5분마다 정리
        self.last_cleanup = time.time()
        
        # 읽기/쓰기 락 관리
        self.read_locks: Dict[str, Set[str]] = {}  # file_path -> set of lock_ids
        self.write_locks: Dict[str, str] = {}  # file_path -> lock_id
        
        # 통계 정보
        self.stats = {
            'total_locks_acquired': 0,
            'total_locks_failed': 0,
            'total_timeouts': 0,
            'current_active_locks': 0
        }
    
    def _normalize_path(self, file_path: str) -> str:
        """파일 경로를 정규화"""
        return os.path.abspath(file_path)
    
    def _generate_lock_id(self, task_id: str, worker_id: str) -> str:
        """고유한 락 ID 생성"""
        timestamp = str(time.time())
        content = f"{task_id}-{worker_id}-{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def _cleanup_expired_locks(self):
        """만료된 락들을 정리"""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        expired_locks = []
        cutoff_time = datetime.now() - timedelta(seconds=self.lock_timeout * 2)
        
        for lock_key, lock_info in self.active_locks.items():
            if lock_info.locked_at < cutoff_time:
                expired_locks.append(lock_key)
        
        for lock_key in expired_locks:
            lock_info = self.active_locks.pop(lock_key, None)
            if lock_info:
                logger.warning(f"Cleaned up expired lock: {lock_info.file_path} from {lock_info.task_id}")
                
                # 물리적 락 파일도 제거
                lock_file_path = f"{lock_info.file_path}.vibe_lock"
                try:
                    if os.path.exists(lock_file_path):
                        os.remove(lock_file_path)
                except OSError as e:
                    logger.error(f"Failed to remove lock file {lock_file_path}: {e}")
        
        self.last_cleanup = current_time
        self.stats['current_active_locks'] = len(self.active_locks)
    
    @asynccontextmanager
    async def exclusive_file_access(self, file_path: str, task_id: str, worker_id: str):
        """파일에 대한 배타적 접근 제어"""
        normalized_path = self._normalize_path(file_path)
        lock_id = self._generate_lock_id(task_id, worker_id)
        lock_key = f"{normalized_path}:{lock_id}"
        
        # 만료된 락 정리
        await self._cleanup_expired_locks()
        
        # 이미 락이 걸려있는지 확인
        conflicting_locks = [
            lock_info for key, lock_info in self.active_locks.items()
            if lock_info.file_path == normalized_path
        ]
        
        if conflicting_locks:
            conflict_info = conflicting_locks[0]
            raise ResourceBusyError(
                f"File {file_path} is being modified by task {conflict_info.task_id} "
                f"(worker: {conflict_info.worker_id}) since {conflict_info.locked_at}"
            )
        
        # 물리적 파일 락 시도
        lock_file_path = f"{normalized_path}.vibe_lock"
        lock_file = None
        
        try:
            # 락 디렉토리 생성
            lock_dir = os.path.dirname(lock_file_path)
            os.makedirs(lock_dir, exist_ok=True)
            
            # 논블로킹 락 시도
            lock_file = open(lock_file_path, 'w')
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                lock_file.close()
                raise ResourceBusyError(f"Cannot acquire physical lock for {file_path}")
            
            # 락 정보 기록
            lock_info = FileLockInfo(
                file_path=normalized_path,
                task_id=task_id,
                worker_id=worker_id,
                locked_at=datetime.now(),
                lock_type='exclusive'
            )
            
            self.active_locks[lock_key] = lock_info
            self.stats['total_locks_acquired'] += 1
            self.stats['current_active_locks'] = len(self.active_locks)
            
            logger.info(f"Acquired exclusive lock: {file_path} for task {task_id}")
            
            yield lock_info
            
        except Exception as e:
            self.stats['total_locks_failed'] += 1
            logger.error(f"Failed to acquire exclusive lock for {file_path}: {e}")
            raise
            
        finally:
            # 락 해제
            if lock_key in self.active_locks:
                del self.active_locks[lock_key]
                self.stats['current_active_locks'] = len(self.active_locks)
            
            # 물리적 락 파일 해제
            if lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                except:
                    pass
            
            # 락 파일 삭제
            try:
                if os.path.exists(lock_file_path):
                    os.remove(lock_file_path)
            except OSError as e:
                logger.warning(f"Failed to remove lock file {lock_file_path}: {e}")
            
            logger.info(f"Released exclusive lock: {file_path} for task {task_id}")
    
    @asynccontextmanager
    async def shared_file_access(self, file_path: str, task_id: str, worker_id: str):
        """파일에 대한 공유 읽기 접근 제어"""
        normalized_path = self._normalize_path(file_path)
        lock_id = self._generate_lock_id(task_id, worker_id)
        
        # 쓰기 락이 있는지 확인
        write_lock = self.write_locks.get(normalized_path)
        if write_lock:
            raise ResourceBusyError(f"File {file_path} is being written by another task")
        
        try:
            # 읽기 락 추가
            if normalized_path not in self.read_locks:
                self.read_locks[normalized_path] = set()
            self.read_locks[normalized_path].add(lock_id)
            
            logger.debug(f"Acquired shared lock: {file_path} for task {task_id}")
            yield
            
        finally:
            # 읽기 락 해제
            if normalized_path in self.read_locks:
                self.read_locks[normalized_path].discard(lock_id)
                if not self.read_locks[normalized_path]:
                    del self.read_locks[normalized_path]
            
            logger.debug(f"Released shared lock: {file_path} for task {task_id}")
    
    async def check_file_availability(self, file_path: str) -> Dict[str, any]:
        """파일의 가용성 상태 확인"""
        normalized_path = self._normalize_path(file_path)
        
        # 활성 락 확인
        active_locks = [
            lock_info for lock_info in self.active_locks.values()
            if lock_info.file_path == normalized_path
        ]
        
        # 읽기/쓰기 락 확인
        read_count = len(self.read_locks.get(normalized_path, set()))
        has_write_lock = normalized_path in self.write_locks
        
        return {
            'available': len(active_locks) == 0 and not has_write_lock,
            'active_exclusive_locks': len(active_locks),
            'active_read_locks': read_count,
            'has_write_lock': has_write_lock,
            'lock_details': [
                {
                    'task_id': lock.task_id,
                    'worker_id': lock.worker_id,
                    'locked_since': lock.locked_at.isoformat(),
                    'lock_type': lock.lock_type
                }
                for lock in active_locks
            ]
        }
    
    def get_stats(self) -> Dict[str, any]:
        """락 시스템 통계 정보 반환"""
        return {
            **self.stats,
            'active_files': len(set(lock.file_path for lock in self.active_locks.values())),
            'longest_lock_duration': self._get_longest_lock_duration(),
            'read_locks_count': sum(len(locks) for locks in self.read_locks.values()),
            'write_locks_count': len(self.write_locks)
        }
    
    def _get_longest_lock_duration(self) -> float:
        """가장 오래된 락의 지속 시간 반환 (초)"""
        if not self.active_locks:
            return 0.0
        
        oldest_lock = min(self.active_locks.values(), key=lambda x: x.locked_at)
        return (datetime.now() - oldest_lock.locked_at).total_seconds()
    
    async def force_release_locks(self, task_id: str = None, file_path: str = None):
        """강제로 락 해제 (긴급 상황용)"""
        released_count = 0
        
        locks_to_remove = []
        for lock_key, lock_info in self.active_locks.items():
            should_release = False
            
            if task_id and lock_info.task_id == task_id:
                should_release = True
            elif file_path and lock_info.file_path == self._normalize_path(file_path):
                should_release = True
            elif not task_id and not file_path:  # 모든 락 해제
                should_release = True
            
            if should_release:
                locks_to_remove.append(lock_key)
        
        for lock_key in locks_to_remove:
            lock_info = self.active_locks.pop(lock_key, None)
            if lock_info:
                # 물리적 락 파일 제거
                lock_file_path = f"{lock_info.file_path}.vibe_lock"
                try:
                    if os.path.exists(lock_file_path):
                        os.remove(lock_file_path)
                except OSError:
                    pass
                
                released_count += 1
                logger.warning(f"Force released lock: {lock_info.file_path} from {lock_info.task_id}")
        
        self.stats['current_active_locks'] = len(self.active_locks)
        return released_count


# 전역 파일 가드 인스턴스
global_file_guard = FileOperationGuard()


async def safe_file_operation(file_path: str, task_id: str, worker_id: str, operation_type: str = 'exclusive'):
    """안전한 파일 작업을 위한 컨텍스트 매니저 팩토리"""
    if operation_type == 'exclusive':
        return global_file_guard.exclusive_file_access(file_path, task_id, worker_id)
    elif operation_type == 'shared':
        return global_file_guard.shared_file_access(file_path, task_id, worker_id)
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")