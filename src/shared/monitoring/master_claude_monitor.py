"""Master Claude Progress Monitoring System."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import websockets
import threading
from dataclasses import dataclass, asdict
from enum import Enum

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


class MasterClaudeState(Enum):
    """Master Claude의 현재 상태."""
    INITIALIZING = "initializing"
    ANALYZING_CONTEXT = "analyzing_context" 
    PLANNING_TASK = "planning_task"
    SUPERVISING_EXECUTION = "supervising_execution"
    EVALUATING_RESULT = "evaluating_result"
    COMPRESSING_CONTEXT = "compressing_context"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class ContextStats:
    """컨텍스트 사용량 통계."""
    total_tokens: int
    available_tokens: int
    utilization: float
    context_windows: int
    summaries: int
    compression_ratio: float
    last_compression: Optional[str] = None


@dataclass
class TaskProgress:
    """개별 작업 진행 상황."""
    task_id: str
    description: str
    status: str
    current_iteration: int
    max_iterations: int
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class MasterClaudeStatus:
    """Master Claude 전체 상태."""
    state: MasterClaudeState
    current_task: Optional[TaskProgress]
    context_stats: ContextStats
    insights_learned: Dict[str, int]
    session_duration: timedelta
    total_tasks_completed: int
    total_tasks_failed: int
    current_message: str
    last_update: datetime


class MasterClaudeMonitor:
    """Master Claude 진행 상황 모니터링 시스템."""
    
    def __init__(self, master_supervisor):
        self.master_supervisor = master_supervisor
        self.status = MasterClaudeStatus(
            state=MasterClaudeState.INITIALIZING,
            current_task=None,
            context_stats=ContextStats(0, 0, 0.0, 0, 0, 1.0),
            insights_learned={"patterns": 0, "conventions": 0, "decisions": 0},
            session_duration=timedelta(),
            total_tasks_completed=0,
            total_tasks_failed=0,
            current_message="Master Claude 초기화 중...",
            last_update=datetime.now()
        )
        
        self.session_start = datetime.now()
        self.subscribers: List[asyncio.Queue] = []
        self.is_monitoring = False
        
        # 실시간 대시보드
        self.live_display = None
        self.progress_bars = {}
        
    def start_monitoring(self):
        """모니터링 시작."""
        self.is_monitoring = True
        logger.info("Master Claude 모니터링 시작")
        
    def stop_monitoring(self):
        """모니터링 중지."""
        self.is_monitoring = False
        if self.live_display:
            self.live_display.stop()
        logger.info("Master Claude 모니터링 종료")
        
    async def subscribe_to_updates(self) -> asyncio.Queue:
        """실시간 업데이트 구독."""
        queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue
        
    async def _broadcast_update(self):
        """모든 구독자에게 업데이트 전송."""
        if not self.subscribers:
            return
            
        update = {
            "timestamp": datetime.now().isoformat(),
            "status": asdict(self.status)
        }
        
        for queue in self.subscribers[:]:  # 복사본으로 반복
            try:
                await queue.put(update)
            except:
                # 연결이 끊어진 구독자 제거
                self.subscribers.remove(queue)
    
    def update_state(self, state: MasterClaudeState, message: str = ""):
        """Master Claude 상태 업데이트."""
        if not self.is_monitoring:
            return
            
        self.status.state = state
        self.status.current_message = message or self._get_default_message(state)
        self.status.last_update = datetime.now()
        self.status.session_duration = datetime.now() - self.session_start
        
        # 컨텍스트 통계 업데이트
        self._update_context_stats()
        
        # 인사이트 통계 업데이트
        self._update_insights_stats()
        
        # 실시간 업데이트 전송
        asyncio.create_task(self._broadcast_update())
        
        logger.info(f"Master Claude 상태: {state.value} - {self.status.current_message}")
    
    def start_task(self, task_id: str, description: str, max_iterations: int = 3):
        """새 작업 시작 모니터링."""
        if not self.is_monitoring:
            return
            
        self.status.current_task = TaskProgress(
            task_id=task_id,
            description=description,
            status="starting",
            current_iteration=0,
            max_iterations=max_iterations,
            started_at=datetime.now()
        )
        
        self.update_state(MasterClaudeState.PLANNING_TASK, f"작업 '{task_id}' 계획 수립 중")
    
    def update_task_iteration(self, iteration: int, status: str = "running"):
        """작업 반복 상태 업데이트."""
        if not self.is_monitoring or not self.status.current_task:
            return
            
        self.status.current_task.current_iteration = iteration
        self.status.current_task.status = status
        
        # 완료 시간 추정
        if iteration > 0:
            elapsed = datetime.now() - self.status.current_task.started_at
            avg_per_iteration = elapsed / iteration
            remaining_iterations = self.status.current_task.max_iterations - iteration
            self.status.current_task.estimated_completion = datetime.now() + (avg_per_iteration * remaining_iterations)
        
        self.update_state(
            MasterClaudeState.SUPERVISING_EXECUTION,
            f"반복 {iteration}/{self.status.current_task.max_iterations} 실행 중"
        )
    
    def complete_task(self, success: bool, error_message: str = None):
        """작업 완료 처리."""
        if not self.is_monitoring or not self.status.current_task:
            return
            
        if success:
            self.status.total_tasks_completed += 1
            self.status.current_task.status = "completed"
            message = f"작업 '{self.status.current_task.task_id}' 완료"
        else:
            self.status.total_tasks_failed += 1
            self.status.current_task.status = "failed"
            self.status.current_task.error_message = error_message
            message = f"작업 '{self.status.current_task.task_id}' 실패: {error_message}"
        
        self.update_state(MasterClaudeState.WAITING, message)
        
        # 잠시 후 현재 작업 클리어
        asyncio.create_task(self._clear_current_task_after_delay(3))
    
    async def _clear_current_task_after_delay(self, delay_seconds: int):
        """지연 후 현재 작업 정보 클리어."""
        await asyncio.sleep(delay_seconds)
        self.status.current_task = None
        await self._broadcast_update()
    
    def report_context_compression(self, tokens_saved: int, compression_ratio: float):
        """컨텍스트 압축 보고."""
        if not self.is_monitoring:
            return
            
        self.update_state(
            MasterClaudeState.COMPRESSING_CONTEXT,
            f"컨텍스트 압축: {tokens_saved:,} 토큰 절약 ({compression_ratio:.2f}x)"
        )
        
        self.status.context_stats.last_compression = datetime.now().strftime("%H:%M:%S")
        
    def report_error(self, error_message: str):
        """오류 상황 보고."""
        if not self.is_monitoring:
            return
            
        self.update_state(MasterClaudeState.ERROR, f"오류 발생: {error_message}")
        
        if self.status.current_task:
            self.status.current_task.error_message = error_message
            self.status.current_task.status = "error"
    
    def _update_context_stats(self):
        """컨텍스트 통계 업데이트."""
        try:
            stats = self.master_supervisor.context_manager.get_context_stats()
            self.status.context_stats = ContextStats(
                total_tokens=stats['total_tokens'],
                available_tokens=stats['available_tokens'],
                utilization=stats['utilization'],
                context_windows=stats['context_windows'],
                summaries=stats['summaries'],
                compression_ratio=stats['compression_ratio'],
                last_compression=self.status.context_stats.last_compression
            )
        except Exception as e:
            logger.warning(f"컨텍스트 통계 업데이트 실패: {e}")
    
    def _update_insights_stats(self):
        """인사이트 통계 업데이트."""
        try:
            insights = self.master_supervisor.claude_insights
            self.status.insights_learned = {
                "patterns": len(insights.get('established_patterns', [])),
                "conventions": len(insights.get('coding_conventions', {})),
                "decisions": len(insights.get('architectural_decisions', [])),
                "cli_patterns": len(insights.get('cli_specific_patterns', []))
            }
        except Exception as e:
            logger.warning(f"인사이트 통계 업데이트 실패: {e}")
    
    def _get_default_message(self, state: MasterClaudeState) -> str:
        """상태별 기본 메시지."""
        messages = {
            MasterClaudeState.INITIALIZING: "시스템 초기화 중...",
            MasterClaudeState.ANALYZING_CONTEXT: "프로젝트 컨텍스트 분석 중...",
            MasterClaudeState.PLANNING_TASK: "작업 실행 계획 수립 중...",
            MasterClaudeState.SUPERVISING_EXECUTION: "Code Claude 실행 감독 중...",
            MasterClaudeState.EVALUATING_RESULT: "실행 결과 평가 중...",
            MasterClaudeState.COMPRESSING_CONTEXT: "컨텍스트 압축 진행 중...",
            MasterClaudeState.WAITING: "다음 작업 대기 중...",
            MasterClaudeState.ERROR: "오류 상황 처리 중..."
        }
        return messages.get(state, "알 수 없는 상태")
    
    def create_status_display(self) -> Layout:
        """실시간 상태 표시 레이아웃 생성."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # 헤더
        layout["header"].update(Panel(
            Text("🧠 Master Claude 실시간 모니터링", style="bold blue"),
            style="blue"
        ))
        
        # 메인 상태 정보
        layout["left"].update(self._create_main_status_panel())
        layout["right"].update(self._create_context_panel())
        
        # 푸터
        layout["footer"].update(self._create_progress_panel())
        
        return layout
    
    def _create_main_status_panel(self) -> Panel:
        """메인 상태 패널 생성."""
        status_text = Text()
        
        # 현재 상태
        state_color = self._get_state_color(self.status.state)
        status_text.append(f"상태: ", style="bold")
        status_text.append(f"{self.status.state.value}\n", style=state_color)
        
        # 현재 메시지
        status_text.append(f"진행상황: {self.status.current_message}\n")
        
        # 세션 정보
        status_text.append(f"세션 시간: {self._format_duration(self.status.session_duration)}\n")
        status_text.append(f"완료된 작업: {self.status.total_tasks_completed}\n")
        status_text.append(f"실패한 작업: {self.status.total_tasks_failed}\n")
        
        # 현재 작업 정보
        if self.status.current_task:
            task = self.status.current_task
            status_text.append(f"\n현재 작업: {task.task_id}\n", style="bold yellow")
            status_text.append(f"설명: {task.description}\n")
            status_text.append(f"반복: {task.current_iteration}/{task.max_iterations}\n")
            
            if task.estimated_completion:
                eta = task.estimated_completion - datetime.now()
                if eta.total_seconds() > 0:
                    status_text.append(f"예상 완료: {self._format_duration(eta)}\n")
        
        return Panel(status_text, title="Master Claude 상태", border_style="green")
    
    def _create_context_panel(self) -> Panel:
        """컨텍스트 상태 패널 생성."""
        ctx = self.status.context_stats
        
        context_text = Text()
        
        # 토큰 사용량
        utilization_color = "red" if ctx.utilization > 0.8 else "yellow" if ctx.utilization > 0.6 else "green"
        context_text.append(f"토큰 사용: {ctx.total_tokens:,}/{ctx.available_tokens:,}\n")
        context_text.append(f"사용률: ", style="bold")
        context_text.append(f"{ctx.utilization:.1%}\n", style=utilization_color)
        
        # 컨텍스트 구조
        context_text.append(f"컨텍스트 윈도우: {ctx.context_windows}\n")
        context_text.append(f"요약본: {ctx.summaries}\n")
        context_text.append(f"압축비: {ctx.compression_ratio:.2f}x\n")
        
        if ctx.last_compression:
            context_text.append(f"최근 압축: {ctx.last_compression}\n")
        
        # 학습된 인사이트
        insights = self.status.insights_learned
        context_text.append(f"\n학습된 인사이트:\n", style="bold cyan")
        context_text.append(f"• 패턴: {insights.get('patterns', 0)}\n")
        context_text.append(f"• 규칙: {insights.get('conventions', 0)}\n")
        context_text.append(f"• 결정: {insights.get('decisions', 0)}\n")
        
        if 'cli_patterns' in insights:
            context_text.append(f"• CLI 패턴: {insights['cli_patterns']}\n")
        
        return Panel(context_text, title="컨텍스트 & 학습", border_style="cyan")
    
    def _create_progress_panel(self) -> Panel:
        """진행 상황 패널 생성."""
        if not self.status.current_task:
            return Panel("현재 진행 중인 작업이 없습니다.", title="작업 진행률")
        
        task = self.status.current_task
        progress_text = Text()
        
        # 진행률 바 텍스트로 표현
        progress_ratio = task.current_iteration / task.max_iterations if task.max_iterations > 0 else 0
        bar_length = 30
        filled_length = int(bar_length * progress_ratio)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        progress_text.append(f"작업: {task.task_id}\n")
        progress_text.append(f"진행률: [{bar}] {progress_ratio:.1%}\n")
        progress_text.append(f"반복: {task.current_iteration}/{task.max_iterations}\n")
        
        elapsed = datetime.now() - task.started_at
        progress_text.append(f"경과 시간: {self._format_duration(elapsed)}\n")
        
        return Panel(progress_text, title="작업 진행률", border_style="yellow")
    
    def _get_state_color(self, state: MasterClaudeState) -> str:
        """상태별 색상 반환."""
        colors = {
            MasterClaudeState.INITIALIZING: "blue",
            MasterClaudeState.ANALYZING_CONTEXT: "cyan",
            MasterClaudeState.PLANNING_TASK: "yellow",
            MasterClaudeState.SUPERVISING_EXECUTION: "green",
            MasterClaudeState.EVALUATING_RESULT: "magenta",
            MasterClaudeState.COMPRESSING_CONTEXT: "orange",
            MasterClaudeState.WAITING: "dim",
            MasterClaudeState.ERROR: "red"
        }
        return colors.get(state, "white")
    
    def _format_duration(self, duration: timedelta) -> str:
        """시간 간격을 읽기 쉬운 형태로 포맷."""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}시간 {minutes}분 {seconds}초"
        elif minutes > 0:
            return f"{minutes}분 {seconds}초"
        else:
            return f"{seconds}초"
    
    def start_live_dashboard(self):
        """실시간 대시보드 시작."""
        if self.live_display:
            return
            
        layout = self.create_status_display()
        
        self.live_display = Live(
            layout,
            refresh_per_second=2,
            screen=True
        )
        
        # 별도 스레드에서 실행
        def run_dashboard():
            with self.live_display:
                while self.is_monitoring:
                    # 레이아웃 업데이트
                    layout["left"].update(self._create_main_status_panel())
                    layout["right"].update(self._create_context_panel())
                    layout["footer"].update(self._create_progress_panel())
                    
                    import time
                    time.sleep(0.5)
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
    
    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태를 딕셔너리로 반환."""
        return asdict(self.status)
    
    async def export_session_log(self, file_path: str):
        """세션 로그를 파일로 내보내기."""
        session_data = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "final_status": asdict(self.status),
            "total_duration": str(self.status.session_duration),
            "summary": {
                "tasks_completed": self.status.total_tasks_completed,
                "tasks_failed": self.status.total_tasks_failed,
                "success_rate": self.status.total_tasks_completed / (self.status.total_tasks_completed + self.status.total_tasks_failed) if (self.status.total_tasks_completed + self.status.total_tasks_failed) > 0 else 0,
                "context_efficiency": {
                    "final_utilization": self.status.context_stats.utilization,
                    "compression_ratio": self.status.context_stats.compression_ratio,
                    "summaries_created": self.status.context_stats.summaries
                },
                "insights_learned": self.status.insights_learned
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"세션 로그 내보내기 완료: {file_path}")