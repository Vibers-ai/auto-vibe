"""File Activity Monitor - 파일 생성/수정 활동을 모니터링하여 데드락 감지"""

import asyncio
import subprocess
import logging
import time
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class FileActivityMonitor:
    """Linux 명령어를 사용한 파일 활동 모니터링"""
    
    def __init__(self, workspace_path: str, inactivity_threshold: int = 30):
        self.workspace_path = Path(workspace_path).resolve()
        self.inactivity_threshold = inactivity_threshold  # 초 단위 (기본 30초로 단축)
        
        # 모니터링 상태
        self.monitoring_active = False
        self.monitoring_task = None
        self.last_activity_time = time.time()
        self.last_file_count = 0
        self.last_total_size = 0
        
        # 활동 기록
        self.activity_log: List[Dict] = []
        self.created_files: Set[str] = set()
        self.modified_files: Set[str] = set()
        
        # 초기 상태 캡처
        self._capture_initial_state()
    
    def _capture_initial_state(self):
        """초기 파일 시스템 상태 캡처"""
        try:
            # 파일 개수 확인
            result = subprocess.run(
                ['find', str(self.workspace_path), '-type', 'f', '-name', '*', '!', '-path', '*/.*'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.last_file_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            # 전체 크기 확인
            result = subprocess.run(
                ['du', '-sb', str(self.workspace_path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout:
                self.last_total_size = int(result.stdout.split('\t')[0])
                
            logger.info(f"Initial state captured: {self.last_file_count} files, {self.last_total_size} bytes")
            
        except Exception as e:
            logger.error(f"Error capturing initial state: {e}")
    
    async def start_monitoring(self):
        """파일 활동 모니터링 시작"""
        if self.monitoring_active:
            logger.warning("File activity monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"File activity monitoring started for {self.workspace_path}")
    
    async def stop_monitoring(self):
        """파일 활동 모니터링 중지"""
        self.monitoring_active = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.monitoring_task.cancel()
        
        logger.info("File activity monitoring stopped")
    
    async def _monitoring_loop(self):
        """메인 모니터링 루프"""
        check_interval = 3  # 3초마다 체크 (더 빠른 감지)
        
        while self.monitoring_active:
            try:
                # 파일 활동 체크
                activity_detected = await self._check_file_activity()
                
                if activity_detected:
                    self.last_activity_time = time.time()
                    logger.debug("File activity detected")
                else:
                    # 비활동 시간 체크
                    inactive_time = time.time() - self.last_activity_time
                    if inactive_time > self.inactivity_threshold:
                        logger.warning(f"No file activity for {inactive_time:.0f} seconds")
                        await self._handle_inactivity(inactive_time)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"File monitoring error: {e}")
                await asyncio.sleep(check_interval)
    
    async def _check_file_activity(self) -> bool:
        """파일 활동 체크 (Linux 명령어 사용)"""
        activity_detected = False
        
        try:
            # 1. 최근 수정된 파일 찾기 (최근 15초)
            result = await self._run_command([
                'find', str(self.workspace_path), 
                '-type', 'f',
                '-mmin', '-0.25',  # 15초 이내
                '!', '-path', '*/.*',  # 숨김 파일 제외
                '!', '-path', '*/node_modules/*',  # node_modules 제외
                '!', '-path', '*/venv/*',  # venv 제외
                '!', '-path', '*/__pycache__/*'  # pycache 제외
            ])
            
            if result and result.strip():
                recent_files = result.strip().split('\n')
                for file_path in recent_files:
                    if file_path and file_path not in self.modified_files:
                        self.modified_files.add(file_path)
                        activity_detected = True
                        logger.info(f"Modified file detected: {file_path}")
            
            # 2. 파일 개수 변화 체크
            result = await self._run_command([
                'find', str(self.workspace_path), 
                '-type', 'f',
                '!', '-path', '*/.*',
                '!', '-path', '*/node_modules/*',
                '!', '-path', '*/venv/*',
                '!', '-path', '*/__pycache__/*',
                '-printf', '.'  # 각 파일마다 점 하나씩 출력 (카운트용)
            ])
            
            if result is not None:
                current_file_count = len(result)
                if current_file_count != self.last_file_count:
                    activity_detected = True
                    file_diff = current_file_count - self.last_file_count
                    logger.info(f"File count changed: {self.last_file_count} → {current_file_count} ({file_diff:+d})")
                    self.last_file_count = current_file_count
            
            # 3. 디렉토리 크기 변화 체크
            result = await self._run_command(['du', '-sb', str(self.workspace_path)])
            if result and '\t' in result:
                current_size = int(result.split('\t')[0])
                if current_size != self.last_total_size:
                    activity_detected = True
                    size_diff = current_size - self.last_total_size
                    logger.info(f"Total size changed: {size_diff:+d} bytes")
                    self.last_total_size = current_size
            
            # 4. 활동 로그 기록
            if activity_detected:
                self.activity_log.append({
                    'timestamp': datetime.now(),
                    'file_count': self.last_file_count,
                    'total_size': self.last_total_size,
                    'recent_files': len(recent_files) if 'recent_files' in locals() else 0
                })
            
            return activity_detected
            
        except Exception as e:
            logger.error(f"Error checking file activity: {e}")
            return False
    
    async def _run_command(self, cmd: List[str], timeout: float = 5.0) -> Optional[str]:
        """비동기 명령 실행"""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            
            if proc.returncode == 0:
                return stdout.decode('utf-8')
            else:
                if stderr:
                    logger.debug(f"Command error: {stderr.decode('utf-8')}")
                return None
                
        except asyncio.TimeoutError:
            logger.warning(f"Command timed out: {' '.join(cmd)}")
            if 'proc' in locals():
                proc.terminate()
            return None
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return None
    
    async def _handle_inactivity(self, inactive_time: float):
        """비활동 상태 처리"""
        # 프로세스 상태 체크
        process_info = await self._check_claude_processes()
        
        # 상태 보고
        status = {
            'inactive_seconds': inactive_time,
            'last_file_count': self.last_file_count,
            'files_created': len(self.created_files),
            'files_modified': len(self.modified_files),
            'claude_processes': process_info
        }
        
        logger.warning(f"Inactivity detected: {status}")
        
        # 데드락 가능성 판단
        if inactive_time > self.inactivity_threshold:  # 30초 이상
            logger.warning(f"Possible stuck process: No file activity for {inactive_time:.0f} seconds")
            
            # Claude 프로세스 강제 종료
            if inactive_time > self.inactivity_threshold * 2:  # 60초 이상
                logger.critical(f"Deadlock detected: No file activity for {inactive_time:.0f} seconds - killing Claude processes")
                
                try:
                    # Claude 프로세스 종료
                    kill_result = await self._run_command([
                        'pkill', '-f', 'claude.*--print'
                    ])
                    
                    if kill_result is not None:
                        logger.info("Successfully sent kill signal to Claude processes")
                    
                    # 데드락 감지기에 알림
                    from shared.core.deadlock_detector import global_deadlock_detector
                    
                    # 현재 실행 중인 작업들의 last_activity 업데이트
                    for task_id, task_node in global_deadlock_detector.task_nodes.items():
                        if task_node.status in ["running", "waiting"]:
                            # 파일 활동이 없으므로 오래된 것으로 표시
                            task_node.last_activity = datetime.now() - timedelta(seconds=inactive_time)
                    
                    # 강제 데드락 체크 트리거
                    await global_deadlock_detector._check_timeouts()
                    
                except Exception as e:
                    logger.error(f"Failed to handle deadlock: {e}")
    
    async def _check_claude_processes(self) -> Dict:
        """Claude 프로세스 상태 확인"""
        try:
            # Claude 관련 프로세스 찾기
            result = await self._run_command([
                'ps', 'aux'
            ])
            
            if result:
                claude_processes = []
                for line in result.split('\n'):
                    if 'claude' in line.lower() and 'grep' not in line:
                        parts = line.split()
                        if len(parts) >= 11:
                            claude_processes.append({
                                'pid': parts[1],
                                'cpu': parts[2],
                                'mem': parts[3],
                                'time': parts[9],
                                'cmd': ' '.join(parts[10:])[:100]
                            })
                
                return {
                    'count': len(claude_processes),
                    'processes': claude_processes[:5]  # 최대 5개만
                }
            
        except Exception as e:
            logger.error(f"Error checking processes: {e}")
        
        return {'count': 0, 'processes': []}
    
    def get_activity_report(self) -> Dict:
        """활동 보고서 생성"""
        current_time = time.time()
        inactive_time = current_time - self.last_activity_time
        
        # 최근 활동 분석
        recent_activities = [
            log for log in self.activity_log 
            if log['timestamp'] > datetime.now() - timedelta(minutes=5)
        ]
        
        return {
            'monitoring_active': self.monitoring_active,
            'workspace': str(self.workspace_path),
            'inactive_seconds': round(inactive_time, 1),
            'is_active': inactive_time < 30,  # 30초 이내 활동이면 활성
            'potentially_stuck': inactive_time > self.inactivity_threshold,
            'last_activity': datetime.fromtimestamp(self.last_activity_time).isoformat(),
            'file_stats': {
                'total_files': self.last_file_count,
                'created': len(self.created_files),
                'modified': len(self.modified_files),
                'total_size_mb': round(self.last_total_size / (1024 * 1024), 2)
            },
            'recent_activity_count': len(recent_activities),
            'inactivity_threshold': self.inactivity_threshold
        }
    
    async def force_activity_check(self) -> bool:
        """강제 활동 체크 (외부에서 호출 가능)"""
        return await self._check_file_activity()


# 전역 파일 활동 모니터 인스턴스 (필요시 생성)
_global_file_monitor: Optional[FileActivityMonitor] = None

def get_file_activity_monitor(workspace_path: str) -> FileActivityMonitor:
    """파일 활동 모니터 인스턴스 가져오기"""
    global _global_file_monitor
    
    if _global_file_monitor is None or str(_global_file_monitor.workspace_path) != workspace_path:
        _global_file_monitor = FileActivityMonitor(workspace_path)
    
    return _global_file_monitor