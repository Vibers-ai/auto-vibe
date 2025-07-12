"""Enhanced Context Manager with background compression and segmentation."""

import asyncio
import threading
import time
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import queue
from pathlib import Path

import google.generativeai as genai

from shared.utils.config import Config
from shared.core.schema import Task, TasksPlan
from shared.agents.context_manager import ContextManager, ContextWindow, ContextCompressionStrategy

logger = logging.getLogger(__name__)


class CompressionPriority(Enum):
    """Priority levels for compression operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CompressionJob:
    """Represents a compression job for background processing."""
    job_id: str
    priority: CompressionPriority
    strategy: ContextCompressionStrategy
    context_windows: List[ContextWindow]
    target_tokens: int
    created_at: datetime
    project_area: Optional[str] = None
    task_ids: Set[str] = field(default_factory=set)


@dataclass
class ContextSegment:
    """Represents a segmented context for a specific project area."""
    segment_id: str
    project_area: str  # frontend, backend, database, shared
    context_windows: List[ContextWindow]
    current_tokens: int
    max_tokens: int
    last_accessed: datetime
    compression_level: float = 0.0  # 0.0 = no compression, 1.0 = fully compressed


class BackgroundCompressionWorker:
    """Background worker for context compression."""
    
    def __init__(self, config: Config):
        self.config = config
        self.job_queue = queue.PriorityQueue()
        self.compression_results = {}
        self.worker_thread = None
        self.running = False
        self.stats = {
            'jobs_processed': 0,
            'total_compression_time': 0.0,
            'tokens_saved': 0,
            'average_compression_ratio': 0.0
        }
        
        # Initialize Gemini for compression
        genai.configure(api_key=config.gemini_api_key)
        self.gemini_model = genai.GenerativeModel(config.gemini_model)
    
    def start(self):
        """Start the background compression worker."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Background compression worker started")
    
    def stop(self):
        """Stop the background compression worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Background compression worker stopped")
    
    def submit_job(self, job: CompressionJob) -> str:
        """Submit a compression job to the background worker."""
        priority_value = {
            CompressionPriority.LOW: 4,
            CompressionPriority.MEDIUM: 3,
            CompressionPriority.HIGH: 2,
            CompressionPriority.CRITICAL: 1
        }[job.priority]
        
        self.job_queue.put((priority_value, time.time(), job))
        logger.debug(f"Submitted compression job {job.job_id} with priority {job.priority.value}")
        return job.job_id
    
    def get_result(self, job_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Get compression result by job ID."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if job_id in self.compression_results:
                result = self.compression_results.pop(job_id)
                return result
            time.sleep(0.1)
        
        return None
    
    def _worker_loop(self):
        """Main worker loop for processing compression jobs."""
        while self.running:
            try:
                # Get job with timeout to allow checking running flag
                try:
                    priority, timestamp, job = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the compression job
                start_time = time.time()
                result = self._process_compression_job(job)
                compression_time = time.time() - start_time
                
                # Store result
                self.compression_results[job.job_id] = result
                
                # Update statistics
                self.stats['jobs_processed'] += 1
                self.stats['total_compression_time'] += compression_time
                if result.get('success'):
                    self.stats['tokens_saved'] += result.get('tokens_saved', 0)
                
                logger.debug(f"Completed compression job {job.job_id} in {compression_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in background compression worker: {e}")
    
    def _process_compression_job(self, job: CompressionJob) -> Dict[str, Any]:
        """Process a single compression job."""
        try:
            # Calculate current tokens
            current_tokens = sum(window.token_count for window in job.context_windows)
            
            if current_tokens <= job.target_tokens:
                return {
                    'success': True,
                    'compressed_windows': job.context_windows,
                    'tokens_saved': 0,
                    'compression_ratio': 1.0
                }
            
            # Perform compression based on strategy
            if job.strategy == ContextCompressionStrategy.SUMMARIZE:
                return self._summarize_compression(job)
            elif job.strategy == ContextCompressionStrategy.HIERARCHICAL:
                return self._hierarchical_compression(job)
            elif job.strategy == ContextCompressionStrategy.SLIDING_WINDOW:
                return self._sliding_window_compression(job)
            elif job.strategy == ContextCompressionStrategy.SEMANTIC_FILTERING:
                return self._semantic_filtering_compression(job)
            elif job.strategy == ContextCompressionStrategy.HYBRID:
                return self._hybrid_compression(job)
            else:
                raise ValueError(f"Unknown compression strategy: {job.strategy}")
                
        except Exception as e:
            logger.error(f"Compression job {job.job_id} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'compressed_windows': job.context_windows
            }
    
    def _summarize_compression(self, job: CompressionJob) -> Dict[str, Any]:
        """Compress using summarization strategy."""
        # Group windows by content type
        grouped_windows = {}
        for window in job.context_windows:
            content_type = window.content_type
            if content_type not in grouped_windows:
                grouped_windows[content_type] = []
            grouped_windows[content_type].append(window)
        
        compressed_windows = []
        tokens_saved = 0
        
        # Summarize each group
        for content_type, windows in grouped_windows.items():
            if len(windows) <= 1:
                compressed_windows.extend(windows)
                continue
            
            # Combine content for summarization
            combined_content = "\n\n".join(window.content for window in windows)
            combined_tokens = sum(window.token_count for window in windows)
            
            # Create summarization prompt
            summary_prompt = f"""Summarize the following {content_type} content while preserving all critical information:

{combined_content}

Provide a comprehensive summary that maintains:
1. All key decisions and outcomes
2. Important code patterns and conventions
3. Error information and resolutions
4. Dependencies between components

Summary:"""
            
            try:
                response = self.gemini_model.generate_content(
                    summary_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=min(4000, combined_tokens // 2)
                    )
                )
                
                summary_content = response.text
                summary_tokens = len(summary_content.split()) * 1.3  # Rough token estimate
                
                # Create compressed window
                compressed_window = ContextWindow(
                    content=summary_content,
                    token_count=int(summary_tokens),
                    importance_score=max(window.importance_score for window in windows),
                    timestamp=datetime.now(),
                    content_type=f"summary_{content_type}",
                    task_ids=[task_id for window in windows for task_id in window.task_ids]
                )
                
                compressed_windows.append(compressed_window)
                tokens_saved += combined_tokens - int(summary_tokens)
                
            except Exception as e:
                logger.warning(f"Failed to summarize {content_type}: {e}")
                compressed_windows.extend(windows)
        
        total_tokens_after = sum(window.token_count for window in compressed_windows)
        compression_ratio = total_tokens_after / sum(window.token_count for window in job.context_windows)
        
        return {
            'success': True,
            'compressed_windows': compressed_windows,
            'tokens_saved': tokens_saved,
            'compression_ratio': compression_ratio
        }
    
    def _hierarchical_compression(self, job: CompressionJob) -> Dict[str, Any]:
        """Compress using hierarchical importance strategy."""
        # Sort windows by importance score
        sorted_windows = sorted(job.context_windows, key=lambda w: w.importance_score, reverse=True)
        
        compressed_windows = []
        current_tokens = 0
        tokens_saved = 0
        
        # Keep high-importance windows and compress low-importance ones
        high_importance_threshold = 0.7
        
        for window in sorted_windows:
            if current_tokens + window.token_count <= job.target_tokens:
                compressed_windows.append(window)
                current_tokens += window.token_count
            elif window.importance_score >= high_importance_threshold:
                # Try to compress this important window
                compressed_content = window.content[:len(window.content)//2] + "...[truncated]"
                compressed_tokens = len(compressed_content.split()) * 1.3
                
                compressed_window = ContextWindow(
                    content=compressed_content,
                    token_count=int(compressed_tokens),
                    importance_score=window.importance_score,
                    timestamp=window.timestamp,
                    content_type=f"truncated_{window.content_type}",
                    task_ids=window.task_ids
                )
                
                if current_tokens + compressed_tokens <= job.target_tokens:
                    compressed_windows.append(compressed_window)
                    current_tokens += compressed_tokens
                    tokens_saved += window.token_count - compressed_tokens
            else:
                # Skip low-importance windows
                tokens_saved += window.token_count
        
        compression_ratio = current_tokens / sum(window.token_count for window in job.context_windows)
        
        return {
            'success': True,
            'compressed_windows': compressed_windows,
            'tokens_saved': int(tokens_saved),
            'compression_ratio': compression_ratio
        }
    
    def _sliding_window_compression(self, job: CompressionJob) -> Dict[str, Any]:
        """Compress using sliding window strategy."""
        # Sort by timestamp, keep most recent
        sorted_windows = sorted(job.context_windows, key=lambda w: w.timestamp, reverse=True)
        
        compressed_windows = []
        current_tokens = 0
        tokens_saved = 0
        
        for window in sorted_windows:
            if current_tokens + window.token_count <= job.target_tokens:
                compressed_windows.append(window)
                current_tokens += window.token_count
            else:
                tokens_saved += window.token_count
        
        compression_ratio = current_tokens / sum(window.token_count for window in job.context_windows)
        
        return {
            'success': True,
            'compressed_windows': compressed_windows,
            'tokens_saved': int(tokens_saved),
            'compression_ratio': compression_ratio
        }
    
    def _semantic_filtering_compression(self, job: CompressionJob) -> Dict[str, Any]:
        """Compress using semantic filtering strategy."""
        # For now, use a combination of importance and recency
        # In a full implementation, this would use semantic similarity
        
        current_time = datetime.now()
        scored_windows = []
        
        for window in job.context_windows:
            age_hours = (current_time - window.timestamp).total_seconds() / 3600
            age_score = max(0, 1 - age_hours / 24)  # Decay over 24 hours
            
            # Check task relevance
            task_relevance = 1.0
            if job.task_ids:
                common_tasks = set(window.task_ids) & job.task_ids
                task_relevance = len(common_tasks) / len(job.task_ids) if job.task_ids else 0
            
            semantic_score = (window.importance_score * 0.4 + 
                            age_score * 0.3 + 
                            task_relevance * 0.3)
            
            scored_windows.append((semantic_score, window))
        
        # Sort by semantic score and select top windows
        scored_windows.sort(key=lambda x: x[0], reverse=True)
        
        compressed_windows = []
        current_tokens = 0
        tokens_saved = 0
        
        for score, window in scored_windows:
            if current_tokens + window.token_count <= job.target_tokens:
                compressed_windows.append(window)
                current_tokens += window.token_count
            else:
                tokens_saved += window.token_count
        
        compression_ratio = current_tokens / sum(window.token_count for window in job.context_windows)
        
        return {
            'success': True,
            'compressed_windows': compressed_windows,
            'tokens_saved': int(tokens_saved),
            'compression_ratio': compression_ratio
        }
    
    def _hybrid_compression(self, job: CompressionJob) -> Dict[str, Any]:
        """Compress using hybrid strategy combining multiple approaches."""
        # First, apply semantic filtering to reduce the set
        semantic_result = self._semantic_filtering_compression(job)
        
        if not semantic_result['success']:
            return semantic_result
        
        # If still over target, apply summarization
        filtered_windows = semantic_result['compressed_windows']
        current_tokens = sum(window.token_count for window in filtered_windows)
        
        if current_tokens <= job.target_tokens:
            return semantic_result
        
        # Apply summarization to further compress
        summary_job = CompressionJob(
            job_id=f"{job.job_id}_summary",
            priority=job.priority,
            strategy=ContextCompressionStrategy.SUMMARIZE,
            context_windows=filtered_windows,
            target_tokens=job.target_tokens,
            created_at=job.created_at,
            project_area=job.project_area,
            task_ids=job.task_ids
        )
        
        summary_result = self._summarize_compression(summary_job)
        
        if summary_result['success']:
            # Combine tokens saved from both steps
            total_tokens_saved = semantic_result['tokens_saved'] + summary_result['tokens_saved']
            final_tokens = sum(window.token_count for window in summary_result['compressed_windows'])
            final_ratio = final_tokens / sum(window.token_count for window in job.context_windows)
            
            return {
                'success': True,
                'compressed_windows': summary_result['compressed_windows'],
                'tokens_saved': total_tokens_saved,
                'compression_ratio': final_ratio
            }
        
        return semantic_result


class EnhancedContextManager(ContextManager):
    """Enhanced context manager with background compression and segmentation."""
    
    def __init__(self, config: Config, max_context_tokens: int = 128000):
        super().__init__(config, max_context_tokens)
        
        # Enhanced features
        self.compression_worker = BackgroundCompressionWorker(config)
        self.context_segments: Dict[str, ContextSegment] = {}
        self.compression_threshold = 0.8  # Start compression at 80% capacity
        self.prediction_enabled = True
        self.segmentation_enabled = True
        
        # Background compression tracking
        self.pending_compressions: Dict[str, str] = {}  # job_id -> segment_id
        self.compression_queue_size = 0
        
        # Performance metrics
        self.metrics = {
            'background_compressions': 0,
            'predictive_compressions': 0,
            'compression_time_saved': 0.0,
            'average_segment_utilization': 0.0
        }
        
        # Start background worker
        self.compression_worker.start()
        
        logger.info("Enhanced context manager initialized with background compression")
    
    def initialize_segments(self, project_areas: List[str]) -> None:
        """Initialize context segments for different project areas."""
        if not self.segmentation_enabled:
            return
        
        # Calculate tokens per segment
        segment_tokens = self.max_context_tokens // max(len(project_areas), 1)
        
        for area in project_areas:
            segment_id = f"segment_{area}"
            self.context_segments[segment_id] = ContextSegment(
                segment_id=segment_id,
                project_area=area,
                context_windows=[],
                current_tokens=0,
                max_tokens=segment_tokens,
                last_accessed=datetime.now()
            )
        
        # Always create a shared segment
        if 'shared' not in project_areas:
            self.context_segments['segment_shared'] = ContextSegment(
                segment_id='segment_shared',
                project_area='shared',
                context_windows=[],
                current_tokens=0,
                max_tokens=segment_tokens,
                last_accessed=datetime.now()
            )
        
        logger.info(f"Initialized {len(self.context_segments)} context segments")
    
    def add_context_segmented(
        self, 
        content: str, 
        content_type: str, 
        task_ids: List[str],
        importance: float = 0.5,
        project_area: str = "shared"
    ) -> None:
        """Add context to appropriate segment with background compression."""
        
        if not self.segmentation_enabled:
            return super().add_context(content, content_type, task_ids, importance)
        
        # Determine target segment
        segment_id = f"segment_{project_area}"
        if segment_id not in self.context_segments:
            segment_id = "segment_shared"
        
        segment = self.context_segments[segment_id]
        
        # Create context window
        tokens = self._estimate_tokens(content)
        window = ContextWindow(
            content=content,
            token_count=tokens,
            importance_score=importance,
            timestamp=datetime.now(),
            content_type=content_type,
            task_ids=task_ids
        )
        
        # Check if predictive compression needed
        future_tokens = segment.current_tokens + tokens
        compression_threshold_tokens = segment.max_tokens * self.compression_threshold
        
        if future_tokens > compression_threshold_tokens and self.prediction_enabled:
            # Trigger predictive background compression
            self._trigger_predictive_compression(segment)
        
        # Add to segment
        segment.context_windows.append(window)
        segment.current_tokens += tokens
        segment.last_accessed = datetime.now()
        
        logger.debug(f"Added {tokens} tokens to {segment_id} ({segment.current_tokens}/{segment.max_tokens})")
        
        # Check for immediate compression if over limit
        if segment.current_tokens > segment.max_tokens:
            self._trigger_immediate_compression(segment)
    
    def _trigger_predictive_compression(self, segment: ContextSegment) -> None:
        """Trigger predictive background compression before hitting the limit."""
        
        if segment.segment_id in self.pending_compressions:
            return  # Compression already in progress
        
        # Calculate target tokens (leave room for growth)
        target_tokens = int(segment.max_tokens * 0.6)
        
        # Create compression job
        job = CompressionJob(
            job_id=f"predictive_{segment.segment_id}_{int(time.time())}",
            priority=CompressionPriority.MEDIUM,
            strategy=ContextCompressionStrategy.HYBRID,
            context_windows=segment.context_windows.copy(),
            target_tokens=target_tokens,
            created_at=datetime.now(),
            project_area=segment.project_area
        )
        
        job_id = self.compression_worker.submit_job(job)
        self.pending_compressions[job_id] = segment.segment_id
        
        self.metrics['predictive_compressions'] += 1
        logger.info(f"Triggered predictive compression for {segment.segment_id}")
    
    def _trigger_immediate_compression(self, segment: ContextSegment) -> None:
        """Trigger immediate compression when over limit."""
        
        target_tokens = int(segment.max_tokens * 0.7)
        
        job = CompressionJob(
            job_id=f"immediate_{segment.segment_id}_{int(time.time())}",
            priority=CompressionPriority.HIGH,
            strategy=ContextCompressionStrategy.HYBRID,
            context_windows=segment.context_windows.copy(),
            target_tokens=target_tokens,
            created_at=datetime.now(),
            project_area=segment.project_area
        )
        
        job_id = self.compression_worker.submit_job(job)
        
        # Wait for immediate compression (with timeout)
        result = self.compression_worker.get_result(job_id, timeout=10.0)
        
        if result and result.get('success'):
            # Apply compression result
            segment.context_windows = result['compressed_windows']
            segment.current_tokens = sum(w.token_count for w in segment.context_windows)
            segment.compression_level = 1.0 - result['compression_ratio']
            
            self.metrics['background_compressions'] += 1
            logger.info(f"Applied immediate compression to {segment.segment_id}: "
                       f"{result['tokens_saved']} tokens saved")
        else:
            logger.warning(f"Immediate compression failed for {segment.segment_id}")
    
    async def build_context_for_task_segmented(self, task: Task, tasks_plan: TasksPlan) -> str:
        """Build context for task using segmented approach."""
        
        if not self.segmentation_enabled:
            return await super().build_context_for_task(task, tasks_plan)
        
        # Determine relevant segments
        task_area = task.project_area.lower()
        relevant_segments = []
        
        # Always include task's primary segment
        primary_segment_id = f"segment_{task_area}"
        if primary_segment_id in self.context_segments:
            relevant_segments.append(self.context_segments[primary_segment_id])
        
        # Include shared segment
        if 'segment_shared' in self.context_segments:
            relevant_segments.append(self.context_segments['segment_shared'])
        
        # Include related segments based on dependencies
        for dep_task_id in task.dependencies:
            for segment in self.context_segments.values():
                if any(dep_task_id in window.task_ids for window in segment.context_windows):
                    if segment not in relevant_segments:
                        relevant_segments.append(segment)
        
        # Process any pending compressions
        await self._process_pending_compressions()
        
        # Build context from relevant segments
        context_parts = []
        total_tokens = 0
        max_tokens_for_context = self.max_context_tokens // 2  # Leave room for task-specific content
        
        # Sort segments by relevance (primary first, then by last accessed)
        def segment_relevance(seg):
            if seg.project_area == task_area:
                return (0, seg.last_accessed)
            elif seg.project_area == 'shared':
                return (1, seg.last_accessed)
            else:
                return (2, seg.last_accessed)
        
        relevant_segments.sort(key=segment_relevance, reverse=True)
        
        for segment in relevant_segments:
            segment.last_accessed = datetime.now()
            
            # Sort windows within segment by importance and recency
            segment_windows = sorted(
                segment.context_windows,
                key=lambda w: (w.importance_score, w.timestamp.timestamp()),
                reverse=True
            )
            
            for window in segment_windows:
                if total_tokens + window.token_count <= max_tokens_for_context:
                    context_parts.append(f"[{segment.project_area.upper()}] {window.content}")
                    total_tokens += window.token_count
                else:
                    break
            
            if total_tokens >= max_tokens_for_context:
                break
        
        logger.info(f"Built segmented context for task {task.id}: "
                   f"{total_tokens} tokens from {len(relevant_segments)} segments")
        
        return "\n\n".join(context_parts)
    
    async def _process_pending_compressions(self) -> None:
        """Process any pending background compressions."""
        
        completed_jobs = []
        
        for job_id, segment_id in self.pending_compressions.items():
            result = self.compression_worker.get_result(job_id, timeout=0.1)  # Non-blocking check
            
            if result:
                completed_jobs.append(job_id)
                
                if result.get('success') and segment_id in self.context_segments:
                    segment = self.context_segments[segment_id]
                    
                    # Apply compression result
                    segment.context_windows = result['compressed_windows']
                    segment.current_tokens = sum(w.token_count for w in segment.context_windows)
                    segment.compression_level = 1.0 - result['compression_ratio']
                    
                    self.metrics['background_compressions'] += 1
                    logger.info(f"Applied background compression to {segment_id}: "
                               f"{result['tokens_saved']} tokens saved")
        
        # Remove completed jobs
        for job_id in completed_jobs:
            del self.pending_compressions[job_id]
    
    def get_context_stats_enhanced(self) -> Dict[str, Any]:
        """Get enhanced context statistics including segmentation info."""
        
        stats = super().get_context_stats()
        
        if self.segmentation_enabled:
            segment_stats = {}
            total_segment_tokens = 0
            total_max_tokens = 0
            
            for segment in self.context_segments.values():
                utilization = segment.current_tokens / segment.max_tokens
                
                segment_stats[segment.segment_id] = {
                    'project_area': segment.project_area,
                    'current_tokens': segment.current_tokens,
                    'max_tokens': segment.max_tokens,
                    'utilization': utilization,
                    'compression_level': segment.compression_level,
                    'window_count': len(segment.context_windows),
                    'last_accessed': segment.last_accessed.isoformat()
                }
                
                total_segment_tokens += segment.current_tokens
                total_max_tokens += segment.max_tokens
            
            self.metrics['average_segment_utilization'] = (
                total_segment_tokens / total_max_tokens if total_max_tokens > 0 else 0
            )
            
            stats.update({
                'segmentation_enabled': True,
                'segments': segment_stats,
                'pending_compressions': len(self.pending_compressions),
                'background_compression_metrics': self.metrics
            })
        
        return stats
    
    def cleanup(self):
        """Clean up resources including background worker."""
        super().cleanup()
        self.compression_worker.stop()
        logger.info("Enhanced context manager cleaned up")