"""Context Management for Master Claude to handle large projects."""


import sys
from pathlib import Path
# src 디렉토리를 Python path에 추가
src_dir = Path(__file__).parent.parent if 'src' in str(Path(__file__).parent) else Path(__file__).parent
while src_dir.name != 'src' and src_dir.parent != src_dir:
    src_dir = src_dir.parent
if src_dir.name == 'src':
    sys.path.insert(0, str(src_dir))

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import dataclass
from enum import Enum

import google.generativeai as genai

from shared.utils.config import Config
from shared.core.schema import Task, TasksPlan
from shared.utils.file_utils import write_json_file, read_json_file

logger = logging.getLogger(__name__)


class ContextCompressionStrategy(Enum):
    """Strategies for handling context overflow."""
    SUMMARIZE = "summarize"  # Create summaries of completed work
    HIERARCHICAL = "hierarchical"  # Use parent-child task relationships
    SLIDING_WINDOW = "sliding_window"  # Keep only recent context
    SEMANTIC_FILTERING = "semantic_filtering"  # Keep only relevant context
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class ContextWindow:
    """Represents a context window with metadata."""
    content: str
    token_count: int
    importance_score: float
    timestamp: datetime
    content_type: str  # 'task_result', 'project_state', 'execution_plan', etc.
    task_ids: List[str]
    
    def age_hours(self) -> float:
        """Get age of context in hours."""
        return (datetime.now() - self.timestamp).total_seconds() / 3600


@dataclass
class ContextSummary:
    """Compressed summary of context."""
    summary: str
    original_token_count: int
    compressed_token_count: int
    covered_tasks: List[str]
    key_insights: List[str]
    created_at: datetime


class ContextManager:
    """Manages Master Claude's context to prevent overflow."""
    
    def __init__(self, config: Config, max_tokens: int = 128000):
        self.config = config
        self.max_tokens = max_tokens
        self.reserved_tokens = 8000  # Reserve for new task and response
        self.available_tokens = max_tokens - self.reserved_tokens
        
        # Context storage
        self.context_windows: List[ContextWindow] = []
        self.summaries: List[ContextSummary] = []
        self.project_memory: Dict[str, Any] = {}
        
        # Strategy configuration
        self.compression_strategy = ContextCompressionStrategy.HYBRID
        self.max_context_age_hours = 24
        self.importance_threshold = 0.3
        
        # Setup Gemini for summarization
        self.gemini_model = None
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Initialize Gemini model for summarization."""
        genai.configure(api_key=self.config.gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-pro")  # Use efficient model for summarization
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def add_context(self, content: str, content_type: str, 
                   task_ids: List[str] = None, importance: float = 0.5) -> None:
        """Add new context to the manager."""
        if task_ids is None:
            task_ids = []
        
        context_window = ContextWindow(
            content=content,
            token_count=self.estimate_tokens(content),
            importance_score=importance,
            timestamp=datetime.now(),
            content_type=content_type,
            task_ids=task_ids
        )
        
        self.context_windows.append(context_window)
        logger.info(f"Added context: {content_type}, tokens: {context_window.token_count}")
        
        # Check if compression is needed
        if self._needs_compression():
            asyncio.create_task(self._compress_context())
    
    def _needs_compression(self) -> bool:
        """Check if context compression is needed."""
        total_tokens = sum(cw.token_count for cw in self.context_windows)
        return total_tokens > self.available_tokens
    
    async def _compress_context(self) -> None:
        """Compress context using the configured strategy."""
        logger.info("Starting context compression...")
        
        if self.compression_strategy == ContextCompressionStrategy.HYBRID:
            await self._hybrid_compression()
        elif self.compression_strategy == ContextCompressionStrategy.SUMMARIZE:
            await self._summarize_compression()
        elif self.compression_strategy == ContextCompressionStrategy.SLIDING_WINDOW:
            await self._sliding_window_compression()
        elif self.compression_strategy == ContextCompressionStrategy.SEMANTIC_FILTERING:
            await self._semantic_filtering_compression()
        elif self.compression_strategy == ContextCompressionStrategy.HIERARCHICAL:
            await self._hierarchical_compression()
    
    async def _hybrid_compression(self) -> None:
        """Hybrid approach combining multiple strategies."""
        
        # Step 1: Remove very old, low-importance context
        self._remove_stale_context()
        
        # Step 2: Summarize completed task groups
        await self._summarize_task_groups()
        
        # Step 3: If still too large, apply semantic filtering
        if self._needs_compression():
            await self._semantic_filtering_compression()
        
        # Step 4: Last resort - sliding window
        if self._needs_compression():
            await self._sliding_window_compression()
    
    def _remove_stale_context(self) -> None:
        """Remove old, low-importance context."""
        current_time = datetime.now()
        
        filtered_windows = []
        removed_count = 0
        
        for cw in self.context_windows:
            age_hours = cw.age_hours()
            
            # Keep if it's recent OR important
            if age_hours < self.max_context_age_hours or cw.importance_score > 0.7:
                filtered_windows.append(cw)
            else:
                removed_count += 1
        
        self.context_windows = filtered_windows
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} stale context windows")
    
    async def _summarize_task_groups(self) -> None:
        """Summarize groups of completed tasks."""
        
        # Group context by task completion
        task_groups = self._group_contexts_by_tasks()
        
        for task_group, contexts in task_groups.items():
            if len(contexts) > 3:  # Only summarize if there are multiple contexts
                await self._create_summary_for_group(task_group, contexts)
    
    def _group_contexts_by_tasks(self) -> Dict[str, List[ContextWindow]]:
        """Group context windows by related tasks."""
        groups = {}
        
        for cw in self.context_windows:
            # Group by first task ID or content type
            key = cw.task_ids[0] if cw.task_ids else cw.content_type
            
            if key not in groups:
                groups[key] = []
            groups[key].append(cw)
        
        return groups
    
    async def _create_summary_for_group(self, group_key: str, 
                                       contexts: List[ContextWindow]) -> None:
        """Create a summary for a group of related contexts."""
        
        # Combine all context content
        combined_content = "\n\n".join([
            f"[{cw.content_type}] {cw.content}" for cw in contexts
        ])
        
        # Create summarization prompt
        summary_prompt = f"""Analyze and summarize the following development context for task group '{group_key}'. 
Focus on:
1. What was accomplished
2. Key technical decisions made
3. Important insights or patterns discovered
4. Current state and any issues resolved

Context to summarize:
{combined_content}

Provide a concise but comprehensive summary that captures the essential information for future reference:"""

        try:
            response = await self._generate_summary(summary_prompt)
            
            # Calculate compression metrics
            original_tokens = sum(cw.token_count for cw in contexts)
            compressed_tokens = self.estimate_tokens(response)
            
            # Create summary object
            summary = ContextSummary(
                summary=response,
                original_token_count=original_tokens,
                compressed_token_count=compressed_tokens,
                covered_tasks=[task_id for cw in contexts for task_id in cw.task_ids],
                key_insights=self._extract_key_insights(response),
                created_at=datetime.now()
            )
            
            # Add summary and remove original contexts
            self.summaries.append(summary)
            self.context_windows = [cw for cw in self.context_windows if cw not in contexts]
            
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
            logger.info(f"Created summary for '{group_key}': {original_tokens} → {compressed_tokens} tokens (ratio: {compression_ratio:.2f})")
            
        except Exception as e:
            logger.error(f"Error creating summary for group '{group_key}': {e}")
    
    async def _generate_summary(self, prompt: str) -> str:
        """Generate summary using Gemini."""
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for consistent summaries
                    max_output_tokens=2000,  # Limit summary length
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary generation failed"
    
    def _extract_key_insights(self, summary: str) -> List[str]:
        """Extract key insights from summary text."""
        # Simple extraction - look for bullet points or numbered items
        insights = []
        lines = summary.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                insights.append(line[2:])
            elif line and line[0].isdigit() and '. ' in line:
                insights.append(line.split('. ', 1)[1])
        
        return insights[:5]  # Limit to top 5 insights
    
    async def _semantic_filtering_compression(self) -> None:
        """Remove contexts that are less semantically relevant to current work."""
        
        if not self.context_windows:
            return
        
        # Score contexts by relevance
        scored_contexts = []
        
        for cw in self.context_windows:
            relevance_score = await self._calculate_relevance_score(cw)
            scored_contexts.append((cw, relevance_score))
        
        # Sort by relevance and keep top contexts
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate how many to keep based on token budget
        kept_contexts = []
        total_tokens = 0
        
        for cw, score in scored_contexts:
            if total_tokens + cw.token_count <= self.available_tokens:
                kept_contexts.append(cw)
                total_tokens += cw.token_count
            else:
                break
        
        removed_count = len(self.context_windows) - len(kept_contexts)
        self.context_windows = kept_contexts
        
        if removed_count > 0:
            logger.info(f"Semantic filtering removed {removed_count} less relevant contexts")
    
    async def _calculate_relevance_score(self, context: ContextWindow) -> float:
        """Calculate relevance score for a context window."""
        score = context.importance_score
        
        # Boost score for recent contexts
        age_hours = context.age_hours()
        if age_hours < 1:
            score += 0.3
        elif age_hours < 6:
            score += 0.1
        
        # Boost score for certain content types
        if context.content_type in ['execution_plan', 'project_state', 'architecture']:
            score += 0.2
        
        # Boost score if it contains error information or lessons learned
        if any(keyword in context.content.lower() for keyword in ['error', 'failed', 'lesson', 'insight']):
            score += 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _sliding_window_compression(self) -> None:
        """Keep only the most recent contexts within token limit."""
        
        # Sort by timestamp (most recent first)
        self.context_windows.sort(key=lambda cw: cw.timestamp, reverse=True)
        
        kept_contexts = []
        total_tokens = 0
        
        for cw in self.context_windows:
            if total_tokens + cw.token_count <= self.available_tokens:
                kept_contexts.append(cw)
                total_tokens += cw.token_count
            else:
                break
        
        removed_count = len(self.context_windows) - len(kept_contexts)
        self.context_windows = kept_contexts
        
        if removed_count > 0:
            logger.info(f"Sliding window compression removed {removed_count} older contexts")
    
    async def _hierarchical_compression(self) -> None:
        """Organize contexts hierarchically and compress lower levels."""
        
        # Group contexts by hierarchy level
        high_level = []  # Project-level decisions, architecture
        mid_level = []   # Task groups, feature implementations
        low_level = []   # Individual task details, debug info
        
        for cw in self.context_windows:
            if cw.content_type in ['architecture', 'project_state', 'master_plan']:
                high_level.append(cw)
            elif cw.content_type in ['execution_plan', 'task_group_result']:
                mid_level.append(cw)
            else:
                low_level.append(cw)
        
        # Always keep high-level contexts
        kept_contexts = high_level.copy()
        remaining_tokens = self.available_tokens - sum(cw.token_count for cw in high_level)
        
        # Selectively keep mid-level and low-level based on available tokens
        for cw in mid_level + low_level:
            if remaining_tokens >= cw.token_count:
                kept_contexts.append(cw)
                remaining_tokens -= cw.token_count
        
        removed_count = len(self.context_windows) - len(kept_contexts)
        self.context_windows = kept_contexts
        
        if removed_count > 0:
            logger.info(f"Hierarchical compression removed {removed_count} lower-level contexts")
    
    async def build_context_for_task(self, current_task: Task, 
                                   tasks_plan: TasksPlan) -> str:
        """Build optimized context for the current task."""
        
        # Ensure context is within limits
        if self._needs_compression():
            await self._compress_context()
        
        context_parts = []
        
        # Add essential project information (high priority)
        project_context = self._build_project_context(tasks_plan)
        context_parts.append(("PROJECT_CONTEXT", project_context, 1.0))
        
        # Add relevant summaries
        relevant_summaries = self._get_relevant_summaries(current_task)
        if relevant_summaries:
            summary_text = "\n\n".join([s.summary for s in relevant_summaries])
            context_parts.append(("PREVIOUS_WORK_SUMMARY", summary_text, 0.9))
        
        # Add relevant recent context
        relevant_contexts = self._get_relevant_recent_context(current_task)
        for cw in relevant_contexts:
            context_parts.append((cw.content_type.upper(), cw.content, cw.importance_score))
        
        # Build final context string within token limits
        final_context = self._assemble_final_context(context_parts)
        
        return final_context
    
    def _build_project_context(self, tasks_plan: TasksPlan) -> str:
        """Build essential project context."""
        completed_tasks = [t for t in tasks_plan.tasks if t.id in self.project_memory.get('completed_tasks', [])]
        pending_tasks = [t for t in tasks_plan.tasks if t.id not in self.project_memory.get('completed_tasks', [])]
        
        return f"""PROJECT: {tasks_plan.project_id}
TOTAL TASKS: {len(tasks_plan.tasks)}
COMPLETED: {len(completed_tasks)}
REMAINING: {len(pending_tasks)}

COMPLETED TASKS:
{chr(10).join([f"- {t.id}: {t.description[:100]}..." for t in completed_tasks[:5]])}

PENDING TASKS:
{chr(10).join([f"- {t.id}: {t.description[:100]}..." for t in pending_tasks[:5]])}"""
    
    def _get_relevant_summaries(self, current_task: Task) -> List[ContextSummary]:
        """Get summaries relevant to the current task."""
        relevant = []
        
        for summary in self.summaries:
            # Check if summary covers related tasks or areas
            if (any(task_id in summary.covered_tasks for task_id in current_task.dependencies) or
                current_task.project_area in summary.summary.lower() or
                current_task.type in summary.summary.lower()):
                relevant.append(summary)
        
        # Sort by creation time (most recent first) and limit
        relevant.sort(key=lambda s: s.created_at, reverse=True)
        return relevant[:3]  # Limit to 3 most relevant summaries
    
    def _get_relevant_recent_context(self, current_task: Task) -> List[ContextWindow]:
        """Get recent context windows relevant to the current task."""
        relevant = []
        
        for cw in self.context_windows:
            # Check relevance to current task
            if (any(dep_id in cw.task_ids for dep_id in current_task.dependencies) or
                current_task.project_area in cw.content.lower() or
                cw.importance_score > 0.7):
                relevant.append(cw)
        
        # Sort by relevance score and timestamp
        relevant.sort(key=lambda cw: (cw.importance_score, cw.timestamp), reverse=True)
        return relevant[:10]  # Limit to top 10 relevant contexts
    
    def _assemble_final_context(self, context_parts: List[Tuple[str, str, float]]) -> str:
        """Assemble final context string within token limits."""
        
        # Sort by importance
        context_parts.sort(key=lambda x: x[2], reverse=True)
        
        final_parts = []
        total_tokens = 0
        
        for section_name, content, importance in context_parts:
            section_tokens = self.estimate_tokens(content)
            
            if total_tokens + section_tokens <= self.available_tokens:
                final_parts.append(f"## {section_name}\n{content}\n")
                total_tokens += section_tokens
            else:
                # Try to include a truncated version
                available_for_section = self.available_tokens - total_tokens
                if available_for_section > 100:  # Only if we have reasonable space
                    truncated_content = content[:available_for_section * 4]  # Rough char limit
                    final_parts.append(f"## {section_name} (TRUNCATED)\n{truncated_content}...\n")
                break
        
        logger.info(f"Assembled context with {total_tokens} tokens from {len(final_parts)} sections")
        
        return "\n".join(final_parts)
    
    def update_task_completion(self, task_id: str, result: Dict[str, Any]) -> None:
        """Update project memory with task completion."""
        if 'completed_tasks' not in self.project_memory:
            self.project_memory['completed_tasks'] = []
        
        if task_id not in self.project_memory['completed_tasks']:
            self.project_memory['completed_tasks'].append(task_id)
        
        # Store key insights from the task
        self.project_memory[f'task_{task_id}_result'] = {
            'success': result.get('success', False),
            'key_insights': result.get('key_insights', []),
            'completed_at': datetime.now().isoformat()
        }
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about current context usage."""
        total_context_tokens = sum(cw.token_count for cw in self.context_windows)
        total_summary_tokens = sum(s.compressed_token_count for s in self.summaries)
        
        return {
            'total_tokens': total_context_tokens + total_summary_tokens,
            'available_tokens': self.available_tokens,
            'utilization': (total_context_tokens + total_summary_tokens) / self.available_tokens,
            'context_windows': len(self.context_windows),
            'summaries': len(self.summaries),
            'compression_ratio': sum(s.original_token_count for s in self.summaries) / max(sum(s.compressed_token_count for s in self.summaries), 1)
        }
    
    async def save_context_state(self, file_path: str) -> None:
        """Save context state to file for persistence."""
        state = {
            'context_windows': [
                {
                    'content': cw.content,
                    'token_count': cw.token_count,
                    'importance_score': cw.importance_score,
                    'timestamp': cw.timestamp.isoformat(),
                    'content_type': cw.content_type,
                    'task_ids': cw.task_ids
                }
                for cw in self.context_windows
            ],
            'summaries': [
                {
                    'summary': s.summary,
                    'original_token_count': s.original_token_count,
                    'compressed_token_count': s.compressed_token_count,
                    'covered_tasks': s.covered_tasks,
                    'key_insights': s.key_insights,
                    'created_at': s.created_at.isoformat()
                }
                for s in self.summaries
            ],
            'project_memory': self.project_memory
        }
        
        write_json_file(file_path, state)
        logger.info(f"Context state saved to {file_path}")
    
    async def load_context_state(self, file_path: str) -> None:
        """Load context state from file."""
        try:
            state = read_json_file(file_path)
            
            # Restore context windows
            self.context_windows = [
                ContextWindow(
                    content=cw['content'],
                    token_count=cw['token_count'],
                    importance_score=cw['importance_score'],
                    timestamp=datetime.fromisoformat(cw['timestamp']),
                    content_type=cw['content_type'],
                    task_ids=cw['task_ids']
                )
                for cw in state.get('context_windows', [])
            ]
            
            # Restore summaries
            self.summaries = [
                ContextSummary(
                    summary=s['summary'],
                    original_token_count=s['original_token_count'],
                    compressed_token_count=s['compressed_token_count'],
                    covered_tasks=s['covered_tasks'],
                    key_insights=s['key_insights'],
                    created_at=datetime.fromisoformat(s['created_at'])
                )
                for s in state.get('summaries', [])
            ]
            
            # Restore project memory
            self.project_memory = state.get('project_memory', {})
            
            logger.info(f"Context state loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading context state: {e}")
            # Continue with empty state