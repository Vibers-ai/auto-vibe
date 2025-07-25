"""Web-based Dashboard for Master Claude Monitoring."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from .master_claude_monitor import MasterClaudeMonitor

logger = logging.getLogger(__name__)


class WebDashboard:
    """Master Claude 웹 기반 대시보드."""
    
    def __init__(self, monitor: MasterClaudeMonitor, host="localhost", port=8000):
        self.monitor = monitor
        self.host = host
        self.port = port
        self.app = FastAPI(title="Master Claude Dashboard")
        
        # 활성 WebSocket 연결들
        self.active_connections: list[WebSocket] = []
        
        # 라우트 설정
        self._setup_routes()
        
        # 정적 파일 및 템플릿 설정
        self._setup_static_files()
    
    def _setup_routes(self):
        """라우트 설정."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """메인 대시보드 페이지."""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "title": "Master Claude Dashboard"
            })
        
        @self.app.get("/api/status")
        async def get_status():
            """현재 상태 API."""
            return self.monitor.get_current_status()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """실시간 업데이트를 위한 WebSocket."""
            await self._handle_websocket(websocket)
        
        @self.app.get("/api/export")
        async def export_session():
            """세션 데이터 내보내기."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"master_claude_session_{timestamp}.json"
            await self.monitor.export_session_log(filename)
            return {"message": f"Session exported to {filename}"}
    
    def _setup_static_files(self):
        """정적 파일 및 템플릿 설정."""
        # 정적 파일 디렉토리 생성
        static_dir = Path(__file__).parent / "static"
        templates_dir = Path(__file__).parent / "templates"
        
        static_dir.mkdir(exist_ok=True)
        templates_dir.mkdir(exist_ok=True)
        
        # 정적 파일 마운트
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # 템플릿 설정
        self.templates = Jinja2Templates(directory=str(templates_dir))
        
        # 기본 파일들 생성
        self._create_dashboard_files(static_dir, templates_dir)
    
    def _create_dashboard_files(self, static_dir: Path, templates_dir: Path):
        """대시보드 HTML, CSS, JS 파일 생성."""
        
        # HTML 템플릿
        html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>🧠 Master Claude 실시간 모니터링</h1>
            <div class="connection-status" id="connectionStatus">연결 중...</div>
        </header>
        
        <div class="dashboard-grid">
            <!-- 메인 상태 -->
            <div class="card main-status">
                <h2>Master Claude 상태</h2>
                <div class="status-info">
                    <div class="status-item">
                        <span class="label">현재 상태:</span>
                        <span class="value" id="currentState">초기화 중</span>
                    </div>
                    <div class="status-item">
                        <span class="label">진행 상황:</span>
                        <span class="value" id="currentMessage">시스템 시작 중...</span>
                    </div>
                    <div class="status-item">
                        <span class="label">세션 시간:</span>
                        <span class="value" id="sessionDuration">0초</span>
                    </div>
                    <div class="status-item">
                        <span class="label">완료된 작업:</span>
                        <span class="value" id="completedTasks">0</span>
                    </div>
                    <div class="status-item">
                        <span class="label">실패한 작업:</span>
                        <span class="value" id="failedTasks">0</span>
                    </div>
                </div>
            </div>
            
            <!-- 현재 작업 -->
            <div class="card current-task">
                <h2>현재 작업</h2>
                <div id="taskInfo" class="task-info">
                    <div class="no-task">진행 중인 작업이 없습니다</div>
                </div>
                <div class="progress-container" id="progressContainer" style="display: none;">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <div class="progress-text" id="progressText">0%</div>
                </div>
            </div>
            
            <!-- 컨텍스트 상태 -->
            <div class="card context-status">
                <h2>컨텍스트 & 학습</h2>
                <div class="context-info">
                    <div class="context-item">
                        <span class="label">토큰 사용률:</span>
                        <span class="value" id="tokenUtilization">0%</span>
                    </div>
                    <div class="context-item">
                        <span class="label">컨텍스트 윈도우:</span>
                        <span class="value" id="contextWindows">0</span>
                    </div>
                    <div class="context-item">
                        <span class="label">요약본:</span>
                        <span class="value" id="summaries">0</span>
                    </div>
                    <div class="context-item">
                        <span class="label">압축비:</span>
                        <span class="value" id="compressionRatio">1.0x</span>
                    </div>
                </div>
                
                <div class="insights-info">
                    <h3>학습된 인사이트</h3>
                    <div class="insight-item">
                        <span class="label">패턴:</span>
                        <span class="value" id="learnedPatterns">0</span>
                    </div>
                    <div class="insight-item">
                        <span class="label">규칙:</span>
                        <span class="value" id="learnedConventions">0</span>
                    </div>
                    <div class="insight-item">
                        <span class="label">결정:</span>
                        <span class="value" id="learnedDecisions">0</span>
                    </div>
                </div>
            </div>
            
            <!-- 실시간 로그 -->
            <div class="card activity-log">
                <h2>실시간 활동 로그</h2>
                <div class="log-container" id="activityLog">
                    <div class="log-entry">
                        <span class="timestamp">[시작]</span>
                        <span class="message">Master Claude 모니터링 시작</span>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="dashboard-footer">
            <button onclick="exportSession()" class="export-btn">세션 내보내기</button>
            <div class="last-update">마지막 업데이트: <span id="lastUpdate">-</span></div>
        </footer>
    </div>
    
    <script src="/static/dashboard.js"></script>
</body>
</html>"""
        
        with open(templates_dir / "dashboard.html", "w", encoding="utf-8") as f:
            f.write(html_template)
        
        # CSS 스타일
        css_style = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.dashboard-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.dashboard-header {
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.dashboard-header h1 {
    color: #4a5568;
    font-size: 2rem;
}

.connection-status {
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9rem;
}

.connection-status.connected {
    background: #48bb78;
    color: white;
}

.connection-status.disconnected {
    background: #f56565;
    color: white;
}

.connection-status.connecting {
    background: #ed8936;
    color: white;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 20px;
    margin-bottom: 20px;
}

.card {
    background: white;
    padding: 24px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.card h2 {
    color: #2d3748;
    margin-bottom: 16px;
    font-size: 1.25rem;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
}

.main-status {
    grid-column: 1;
    grid-row: 1;
}

.current-task {
    grid-column: 2;
    grid-row: 1;
}

.context-status {
    grid-column: 1;
    grid-row: 2;
}

.activity-log {
    grid-column: 2;
    grid-row: 2;
}

.status-info, .context-info, .insights-info {
    margin-bottom: 16px;
}

.status-item, .context-item, .insight-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    padding: 8px 0;
    border-bottom: 1px solid #f7fafc;
}

.label {
    font-weight: 600;
    color: #4a5568;
}

.value {
    font-weight: 500;
    color: #2d3748;
}

.insights-info h3 {
    color: #3182ce;
    margin-bottom: 12px;
    font-size: 1rem;
}

.task-info {
    margin-bottom: 16px;
}

.no-task {
    color: #718096;
    font-style: italic;
    text-align: center;
    padding: 20px;
}

.progress-container {
    margin-top: 16px;
}

.progress-bar {
    width: 100%;
    height: 20px;
    background: #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 8px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #48bb78, #38a169);
    transition: width 0.3s ease;
    width: 0%;
}

.progress-text {
    text-align: center;
    font-weight: 600;
    color: #4a5568;
}

.log-container {
    max-height: 300px;
    overflow-y: auto;
    padding: 8px;
    background: #f7fafc;
    border-radius: 6px;
}

.log-entry {
    margin-bottom: 8px;
    padding: 8px;
    background: white;
    border-radius: 4px;
    border-left: 3px solid #3182ce;
}

.timestamp {
    color: #718096;
    font-size: 0.85rem;
    margin-right: 8px;
}

.message {
    color: #2d3748;
}

.dashboard-footer {
    background: white;
    padding: 16px 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.export-btn {
    background: #3182ce;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    transition: background 0.2s;
}

.export-btn:hover {
    background: #2c5282;
}

.last-update {
    color: #718096;
    font-size: 0.9rem;
}

/* 상태별 색상 */
.state-initializing { color: #3182ce; }
.state-analyzing_context { color: #0bc5ea; }
.state-planning_task { color: #ecc94b; }
.state-supervising_execution { color: #48bb78; }
.state-evaluating_result { color: #9f7aea; }
.state-compressing_context { color: #ed8936; }
.state-waiting { color: #a0aec0; }
.state-error { color: #f56565; }

@media (max-width: 768px) {
    .dashboard-grid {
        grid-template-columns: 1fr;
    }
    
    .main-status, .current-task, .context-status, .activity-log {
        grid-column: 1;
    }
}
"""
        
        with open(static_dir / "dashboard.css", "w", encoding="utf-8") as f:
            f.write(css_style)
        
        # JavaScript
        js_script = """
class MasterClaudeDashboard {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.logs = [];
        
        this.connect();
        this.setupUI();
    }
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        this.updateConnectionStatus('connecting');
        
        this.ws.onopen = () => {
            console.log('WebSocket 연결됨');
            this.updateConnectionStatus('connected');
            this.reconnectAttempts = 0;
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateDashboard(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket 연결 종료');
            this.updateConnectionStatus('disconnected');
            this.attemptReconnect();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket 오류:', error);
            this.updateConnectionStatus('disconnected');
        };
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`재연결 시도 ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
        }
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        statusElement.className = `connection-status ${status}`;
        
        const statusText = {
            'connecting': '연결 중...',
            'connected': '연결됨',
            'disconnected': '연결 끊어짐'
        };
        
        statusElement.textContent = statusText[status] || status;
    }
    
    updateDashboard(data) {
        const status = data.status;
        
        // 메인 상태 업데이트
        this.updateElement('currentState', status.state, `state-${status.state}`);
        this.updateElement('currentMessage', status.current_message);
        this.updateElement('sessionDuration', this.formatDuration(status.session_duration));
        this.updateElement('completedTasks', status.total_tasks_completed);
        this.updateElement('failedTasks', status.total_tasks_failed);
        
        // 현재 작업 업데이트
        this.updateCurrentTask(status.current_task);
        
        // 컨텍스트 상태 업데이트
        this.updateContextStatus(status.context_stats);
        
        // 인사이트 업데이트
        this.updateInsights(status.insights_learned);
        
        // 마지막 업데이트 시간
        this.updateElement('lastUpdate', new Date(data.timestamp).toLocaleTimeString());
        
        // 로그 추가
        this.addLogEntry(status.current_message, data.timestamp);
    }
    
    updateElement(elementId, value, className = null) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
            if (className) {
                element.className = `value ${className}`;
            }
        }
    }
    
    updateCurrentTask(task) {
        const taskInfo = document.getElementById('taskInfo');
        const progressContainer = document.getElementById('progressContainer');
        
        if (!task) {
            taskInfo.innerHTML = '<div class="no-task">진행 중인 작업이 없습니다</div>';
            progressContainer.style.display = 'none';
            return;
        }
        
        taskInfo.innerHTML = `
            <div class="status-item">
                <span class="label">작업 ID:</span>
                <span class="value">${task.task_id}</span>
            </div>
            <div class="status-item">
                <span class="label">설명:</span>
                <span class="value">${task.description}</span>
            </div>
            <div class="status-item">
                <span class="label">상태:</span>
                <span class="value">${task.status}</span>
            </div>
            <div class="status-item">
                <span class="label">반복:</span>
                <span class="value">${task.current_iteration}/${task.max_iterations}</span>
            </div>
        `;
        
        // 진행률 업데이트
        const progress = task.max_iterations > 0 ? (task.current_iteration / task.max_iterations) * 100 : 0;
        document.getElementById('progressFill').style.width = `${progress}%`;
        document.getElementById('progressText').textContent = `${Math.round(progress)}%`;
        progressContainer.style.display = 'block';
    }
    
    updateContextStatus(contextStats) {
        this.updateElement('tokenUtilization', `${(contextStats.utilization * 100).toFixed(1)}%`);
        this.updateElement('contextWindows', contextStats.context_windows);
        this.updateElement('summaries', contextStats.summaries);
        this.updateElement('compressionRatio', `${contextStats.compression_ratio.toFixed(2)}x`);
    }
    
    updateInsights(insights) {
        this.updateElement('learnedPatterns', insights.patterns || 0);
        this.updateElement('learnedConventions', insights.conventions || 0);
        this.updateElement('learnedDecisions', insights.decisions || 0);
    }
    
    addLogEntry(message, timestamp) {
        const logContainer = document.getElementById('activityLog');
        const time = new Date(timestamp).toLocaleTimeString();
        
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `
            <span class="timestamp">[${time}]</span>
            <span class="message">${message}</span>
        `;
        
        logContainer.appendChild(logEntry);
        
        // 최대 50개 로그 유지
        while (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.firstChild);
        }
        
        // 스크롤을 맨 아래로
        logContainer.scrollTop = logContainer.scrollHeight;
    }
    
    formatDuration(durationStr) {
        // Python timedelta 문자열 파싱 (예: "0:01:23.456789")
        if (!durationStr) return "0초";
        
        try {
            const parts = durationStr.split(':');
            if (parts.length >= 3) {
                const hours = parseInt(parts[0]);
                const minutes = parseInt(parts[1]);
                const seconds = Math.floor(parseFloat(parts[2]));
                
                if (hours > 0) {
                    return `${hours}시간 ${minutes}분 ${seconds}초`;
                } else if (minutes > 0) {
                    return `${minutes}분 ${seconds}초`;
                } else {
                    return `${seconds}초`;
                }
            }
        } catch (e) {
            console.error('Duration parsing error:', e);
        }
        
        return durationStr;
    }
    
    setupUI() {
        // 초기 데이터 로드
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                this.updateDashboard({
                    status: data,
                    timestamp: new Date().toISOString()
                });
            })
            .catch(error => console.error('초기 데이터 로드 실패:', error));
    }
}

// 전역 함수들
function exportSession() {
    fetch('/api/export')
        .then(response => response.json())
        .then(data => {
            alert(data.message);
        })
        .catch(error => {
            console.error('Export error:', error);
            alert('세션 내보내기 실패');
        });
}

// 페이지 로드 시 대시보드 시작
document.addEventListener('DOMContentLoaded', () => {
    new MasterClaudeDashboard();
});
"""
        
        with open(static_dir / "dashboard.js", "w", encoding="utf-8") as f:
            f.write(js_script)
    
    async def _handle_websocket(self, websocket: WebSocket):
        """WebSocket 연결 처리."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # 모니터에서 업데이트 구독
        update_queue = await self.monitor.subscribe_to_updates()
        
        try:
            while True:
                # 업데이트 대기
                update = await update_queue.get()
                
                # 모든 활성 연결에 전송
                await self._broadcast_to_websockets(update)
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket 오류: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def _broadcast_to_websockets(self, data: Dict[str, Any]):
        """모든 WebSocket 연결에 데이터 브로드캐스트."""
        if not self.active_connections:
            return
        
        message = json.dumps(data, ensure_ascii=False)
        
        # 연결이 끊어진 WebSocket 제거를 위한 리스트
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.warning(f"WebSocket 전송 실패: {e}")
                disconnected.append(connection)
        
        # 끊어진 연결 제거
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
    
    async def start_server(self):
        """대시보드 서버 시작."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info(f"웹 대시보드 시작: http://{self.host}:{self.port}")
        await server.serve()
    
    def start_server_background(self):
        """백그라운드에서 대시보드 서버 시작."""
        import threading
        
        def run_server():
            asyncio.new_event_loop().run_until_complete(self.start_server())
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        logger.info(f"웹 대시보드가 백그라운드에서 시작됨: http://{self.host}:{self.port}")