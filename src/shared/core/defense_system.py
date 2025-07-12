"""Defense System Integration - 통합 방어 시스템 초기화 및 관리"""

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from shared.utils.config import Config
from shared.core.file_operation_guard import global_file_guard
from shared.core.token_overflow_guard import global_token_guard
from shared.core.deadlock_detector import global_deadlock_detector
from shared.core.circuit_breaker import global_circuit_breaker_manager
from shared.core.system_health_monitor import create_system_health_monitor
from shared.core.process_manager import global_process_manager, initialize_process_manager, shutdown_process_manager

logger = logging.getLogger(__name__)


class DefenseSystem:
    """통합 방어 시스템 관리자"""
    
    def __init__(self, config: Config, workspace_path: str):
        self.config = config
        self.workspace_path = workspace_path
        
        # 방어 시스템 컴포넌트들
        self.file_guard = global_file_guard
        self.token_guard = global_token_guard
        self.deadlock_detector = global_deadlock_detector
        self.circuit_manager = global_circuit_breaker_manager
        self.process_manager = global_process_manager
        self.health_monitor = None
        
        # 시스템 상태
        self.initialized = False
        self.monitoring_active = False
        
    async def initialize(self):
        """방어 시스템 초기화"""
        if self.initialized:
            logger.warning("Defense system already initialized")
            return
        
        logger.info("Initializing VIBE Defense System...")
        
        try:
            # 1. 프로세스 매니저 초기화
            await initialize_process_manager()
            logger.info("✓ Process manager initialized")
            
            # 2. 토큰 가드 모니터링 시작
            await self.token_guard.start_monitoring()
            logger.info("✓ Token overflow guard initialized")
            
            # 3. 데드락 감지기 시작
            await self.deadlock_detector.start_monitoring()
            logger.info("✓ Deadlock detector initialized")
            
            # 4. 헬스 모니터 생성 및 시작
            self.health_monitor = create_system_health_monitor(
                workspace_path=self.workspace_path,
                token_guard=self.token_guard,
                deadlock_detector=self.deadlock_detector,
                circuit_manager=self.circuit_manager,
                process_manager=self.process_manager
            )
            
            # 알림 콜백 등록
            self.health_monitor.add_alert_callback(self._handle_health_alert)
            
            await self.health_monitor.start_monitoring()
            logger.info("✓ System health monitor initialized")
            
            # 4. 초기 헬스체크 수행
            health_status = await self.health_monitor.force_health_check()
            logger.info(f"✓ Initial health check completed: {len(health_status['health_results'])} components checked")
            
            self.initialized = True
            self.monitoring_active = True
            
            logger.info("🛡️ VIBE Defense System fully initialized and active")
            
        except Exception as e:
            logger.error(f"Failed to initialize defense system: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """방어 시스템 종료"""
        if not self.initialized:
            return
        
        logger.info("Shutting down VIBE Defense System...")
        
        try:
            # 1. 헬스 모니터 중지
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
                logger.info("✓ Health monitor stopped")
            
            # 2. 데드락 감지기 중지
            await self.deadlock_detector.stop_monitoring()
            logger.info("✓ Deadlock detector stopped")
            
            # 3. 토큰 가드 모니터링 중지
            await self.token_guard.stop_monitoring()
            logger.info("✓ Token guard monitoring stopped")
            
            # 4. 프로세스 매니저 종료
            await shutdown_process_manager()
            logger.info("✓ Process manager shutdown")
            
            # 5. 강제 락 해제 (긴급상황 대비)
            released_locks = await self.file_guard.force_release_locks()
            if released_locks > 0:
                logger.warning(f"Force released {released_locks} file locks during shutdown")
            
            self.initialized = False
            self.monitoring_active = False
            
            logger.info("🛡️ VIBE Defense System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during defense system shutdown: {e}")
    
    async def _handle_health_alert(self, alert_data: Dict[str, Any]):
        """헬스 알림 처리"""
        critical_count = alert_data.get('critical_issues', 0)
        warning_count = alert_data.get('warning_issues', 0)
        
        if critical_count > 0:
            logger.critical(f"🚨 CRITICAL HEALTH ALERT: {critical_count} critical issues detected")
            critical_components = alert_data.get('critical_components', [])
            logger.critical(f"Affected components: {', '.join(critical_components)}")
            
            # 긴급 조치 수행
            await self._emergency_response(alert_data)
            
        elif warning_count > 0:
            logger.warning(f"⚠️ Health warning: {warning_count} issues detected")
            warning_components = alert_data.get('warning_components', [])
            logger.warning(f"Affected components: {', '.join(warning_components)}")
    
    async def _emergency_response(self, alert_data: Dict[str, Any]):
        """긴급 상황 대응"""
        critical_components = alert_data.get('critical_components', [])
        
        # 메모리 관련 긴급상황
        if 'memory' in critical_components:
            logger.critical("🆘 Emergency memory cleanup initiated")
            import gc
            gc.collect()
            
            # 토큰 가드에 긴급 압축 요청
            try:
                # Context manager는 직접 접근할 수 없으므로 토큰 가드를 통해 처리
                current_tokens = self.token_guard.stats.current_tokens
                if current_tokens > self.token_guard.emergency_threshold:
                    logger.critical("🆘 Emergency token cleanup - forcing context reduction")
            except Exception as e:
                logger.error(f"Emergency memory cleanup failed: {e}")
        
        # 데드락 관련 긴급상황
        if 'deadlock' in critical_components:
            logger.critical("🆘 Emergency deadlock resolution initiated")
            try:
                # 가장 오래된 작업들 강제 종료
                status_report = self.deadlock_detector.get_status_report()
                task_status = status_report.get('task_status', {})
                
                if task_status.get('blocked', 0) > 0 or task_status.get('waiting', 0) > 2:
                    logger.critical("🆘 Forcing release of blocked resources")
                    await self.file_guard.force_release_locks()
                    
            except Exception as e:
                logger.error(f"Emergency deadlock resolution failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """전체 시스템 상태 반환"""
        if not self.initialized:
            return {
                'status': 'not_initialized',
                'components': {}
            }
        
        try:
            # 각 컴포넌트 상태 수집
            status = {
                'status': 'active' if self.monitoring_active else 'inactive',
                'initialized': self.initialized,
                'components': {
                    'file_guard': {
                        'status': 'active',
                        'stats': self.file_guard.get_stats()
                    },
                    'token_guard': {
                        'status': 'active',
                        'stats': self.token_guard.get_detailed_stats()
                    },
                    'deadlock_detector': {
                        'status': 'active' if self.deadlock_detector.monitoring_active else 'inactive',
                        'stats': self.deadlock_detector.get_status_report()
                    },
                    'circuit_breakers': {
                        'status': 'active',
                        'stats': self.circuit_manager.get_health_summary()
                    },
                    'process_manager': {
                        'status': 'active' if self.process_manager.monitoring_active else 'inactive',
                        'stats': self.process_manager.get_process_stats()
                    }
                }
            }
            
            # 헬스 모니터 상태 추가
            if self.health_monitor:
                health_status = self.health_monitor.get_current_health_status()
                status['components']['health_monitor'] = {
                    'status': 'active' if self.health_monitor.monitoring_active else 'inactive',
                    'overall_health': health_status.get('overall_status', 'unknown'),
                    'component_count': len(health_status.get('components', {}))
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def run_diagnostics(self) -> Dict[str, Any]:
        """종합 진단 실행"""
        logger.info("Running comprehensive system diagnostics...")
        
        diagnostics = {
            'timestamp': asyncio.get_event_loop().time(),
            'components': {}
        }
        
        try:
            # 1. 파일 가드 진단
            file_stats = self.file_guard.get_stats()
            diagnostics['components']['file_guard'] = {
                'active_locks': file_stats['current_active_locks'],
                'total_locks_acquired': file_stats['total_locks_acquired'],
                'total_locks_failed': file_stats['total_locks_failed'],
                'status': 'healthy' if file_stats['current_active_locks'] < 10 else 'warning'
            }
            
            # 2. 토큰 가드 진단
            token_stats = self.token_guard.get_detailed_stats()
            token_utilization = token_stats['current_stats']['utilization_rate']
            diagnostics['components']['token_guard'] = {
                'utilization_rate': token_utilization,
                'compression_success_rate': token_stats['compression_stats']['success_rate'],
                'status': 'healthy' if token_utilization < 80 else 'critical' if token_utilization > 95 else 'warning'
            }
            
            # 3. 데드락 감지기 진단
            deadlock_report = self.deadlock_detector.get_status_report()
            diagnostics['components']['deadlock_detector'] = {
                'monitoring_active': deadlock_report['monitoring_status']['active'],
                'total_deadlocks': deadlock_report['statistics']['total_deadlocks_detected'],
                'recent_deadlocks': len(deadlock_report['recent_deadlocks']),
                'status': 'healthy' if len(deadlock_report['recent_deadlocks']) == 0 else 'warning'
            }
            
            # 4. 서킷 브레이커 진단
            circuit_health = self.circuit_manager.get_health_summary()
            diagnostics['components']['circuit_breakers'] = {
                'overall_health': circuit_health['overall_health'],
                'open_breakers': circuit_health['open_circuit_breakers'],
                'total_breakers': circuit_health['total_circuit_breakers'],
                'status': circuit_health['overall_health']
            }
            
            # 5. 헬스 모니터 진단
            if self.health_monitor:
                health_check = await self.health_monitor.force_health_check()
                health_results = health_check['health_results']
                critical_count = sum(1 for r in health_results if r['status'] == 'critical')
                
                diagnostics['components']['health_monitor'] = {
                    'components_checked': len(health_results),
                    'critical_issues': critical_count,
                    'monitoring_active': self.health_monitor.monitoring_active,
                    'status': 'critical' if critical_count > 0 else 'healthy'
                }
            
            # 전체 상태 결정
            component_statuses = [comp['status'] for comp in diagnostics['components'].values()]
            if 'critical' in component_statuses:
                diagnostics['overall_status'] = 'critical'
            elif 'warning' in component_statuses:
                diagnostics['overall_status'] = 'warning'
            else:
                diagnostics['overall_status'] = 'healthy'
            
            logger.info(f"Diagnostics completed - Overall status: {diagnostics['overall_status']}")
            return diagnostics
            
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}")
            diagnostics['overall_status'] = 'error'
            diagnostics['error'] = str(e)
            return diagnostics
    
    async def emergency_reset(self):
        """긴급 시스템 리셋"""
        logger.critical("🆘 EMERGENCY SYSTEM RESET INITIATED")
        
        try:
            # 1. 모든 파일 락 강제 해제
            released_locks = await self.file_guard.force_release_locks()
            logger.warning(f"Force released {released_locks} file locks")
            
            # 2. 모든 서킷 브레이커 강제 닫기
            for name, breaker in self.circuit_manager.circuit_breakers.items():
                breaker.force_close()
                logger.warning(f"Force closed circuit breaker: {name}")
            
            # 3. 통계 리셋
            self.file_guard.stats = {
                'total_locks_acquired': 0,
                'total_locks_failed': 0,
                'total_timeouts': 0,
                'current_active_locks': 0
            }
            
            self.token_guard.reset_stats()
            
            logger.critical("🆘 Emergency reset completed")
            
        except Exception as e:
            logger.error(f"Emergency reset failed: {e}")
            raise


# 전역 방어 시스템 인스턴스
_global_defense_system: Optional[DefenseSystem] = None


async def initialize_defense_system(config: Config, workspace_path: str) -> DefenseSystem:
    """전역 방어 시스템 초기화"""
    global _global_defense_system
    
    if _global_defense_system is None:
        _global_defense_system = DefenseSystem(config, workspace_path)
        await _global_defense_system.initialize()
    
    return _global_defense_system


def get_defense_system() -> Optional[DefenseSystem]:
    """현재 방어 시스템 인스턴스 반환"""
    return _global_defense_system


async def shutdown_defense_system():
    """전역 방어 시스템 종료"""
    global _global_defense_system
    
    if _global_defense_system:
        await _global_defense_system.shutdown()
        _global_defense_system = None