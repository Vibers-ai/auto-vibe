"""Context Cache Manager for VIBE

This module provides efficient context caching with LRU eviction,
incremental updates, and project area separation.
"""

import json
import logging
import hashlib
import pickle
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from pathlib import Path
import asyncio
import difflib

from shared.utils.config import Config
from shared.core.enhanced_logger import get_logger, LogCategory

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached context entry."""
    key: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    compression_ratio: float = 1.0
    project_area: Optional[str] = None
    task_ids: Set[str] = field(default_factory=set)
    
    # Versioning
    version: int = 1
    parent_key: Optional[str] = None
    diff_from_parent: Optional[str] = None
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_entries: int = 0
    total_size_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    compressions: int = 0
    incremental_updates: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def average_compression_ratio(self) -> float:
        return 1.0  # Would calculate from entries


class ContextCacheManager:
    """Manages context caching with LRU eviction and incremental updates."""
    
    def __init__(self, config: Config, max_entries: int = 10, max_size_mb: int = 500):
        self.config = config
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # LRU cache implementation using OrderedDict
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Project area caches
        self.area_caches: Dict[str, List[str]] = {
            'backend': [],
            'frontend': [],
            'shared': [],
            'testing': []
        }
        
        # Cache statistics
        self.stats = CacheStats()
        
        # Persistence
        self.cache_dir = Path(".vibe_context_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Compression settings
        self.enable_compression = True
        self.compression_threshold = 10 * 1024  # 10KB
        
        # Background tasks
        self.cleanup_interval = 300  # 5 minutes
        self.persist_interval = 60   # 1 minute
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Enhanced logger
        self.logger = get_logger(
            component="context_cache",
            session_id=f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Load persisted cache
        self._load_cache()
    
    async def start(self):
        """Start background tasks."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.background_tasks.append(cleanup_task)
        
        # Start persistence task
        persist_task = asyncio.create_task(self._persist_loop())
        self.background_tasks.append(persist_task)
        
        logger.info("Context cache manager started")
    
    async def stop(self):
        """Stop background tasks and persist cache."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Final persistence
        self._persist_cache()
        
        logger.info("Context cache manager stopped")
    
    async def get(self, key: str, project_area: Optional[str] = None) -> Optional[str]:
        """Get context from cache."""
        async with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                entry = self.cache[key]
                entry.update_access()
                
                self.stats.cache_hits += 1
                
                self.logger.log(
                    level=logging.DEBUG,
                    category=LogCategory.PERFORMANCE,
                    message=f"Cache hit for key: {key}",
                    metadata={
                        'size': entry.size_bytes,
                        'compression_ratio': entry.compression_ratio,
                        'access_count': entry.access_count
                    }
                )
                
                return entry.content
            
            self.stats.cache_misses += 1
            
            # Try to load from disk
            cached_file = self.cache_dir / f"{key}.cache"
            if cached_file.exists():
                try:
                    entry = self._load_entry_from_disk(key)
                    if entry:
                        await self._add_to_cache(entry)
                        return entry.content
                except Exception as e:
                    logger.error(f"Failed to load cache entry from disk: {e}")
            
            return None
    
    async def put(self, key: str, content: str, metadata: Optional[Dict[str, Any]] = None,
                  project_area: Optional[str] = None, task_id: Optional[str] = None) -> bool:
        """Store context in cache."""
        async with self._lock:
            # Check if this is an update
            parent_entry = None
            if key in self.cache:
                parent_entry = self.cache[key]
            
            # Create entry
            entry = CacheEntry(
                key=key,
                content=content,
                metadata=metadata or {},
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=len(content.encode('utf-8')),
                project_area=project_area
            )
            
            if task_id:
                entry.task_ids.add(task_id)
            
            # Handle incremental update
            if parent_entry and self._should_use_incremental(content, parent_entry.content):
                entry.parent_key = f"{key}_v{parent_entry.version}"
                entry.version = parent_entry.version + 1
                entry.diff_from_parent = self._create_diff(parent_entry.content, content)
                
                # Store parent with versioned key
                parent_entry.key = entry.parent_key
                self.cache[entry.parent_key] = parent_entry
                
                self.stats.incremental_updates += 1
            
            # Compress if needed
            if self.enable_compression and entry.size_bytes > self.compression_threshold:
                compressed = self._compress_content(content)
                if compressed:
                    entry.content = compressed
                    entry.compression_ratio = entry.size_bytes / len(compressed.encode('utf-8'))
                    self.stats.compressions += 1
            
            # Add to cache
            await self._add_to_cache(entry)
            
            # Update area cache
            if project_area and project_area in self.area_caches:
                if key not in self.area_caches[project_area]:
                    self.area_caches[project_area].append(key)
            
            return True
    
    async def update_incremental(self, key: str, changes: Dict[str, Any]) -> Optional[str]:
        """Update context incrementally."""
        async with self._lock:
            existing = await self.get(key)
            if not existing:
                return None
            
            try:
                # Parse existing content as JSON
                existing_data = json.loads(existing)
                
                # Apply changes
                for change_key, change_value in changes.items():
                    if change_value is None and change_key in existing_data:
                        del existing_data[change_key]
                    else:
                        existing_data[change_key] = change_value
                
                # Convert back to string
                updated_content = json.dumps(existing_data, indent=2)
                
                # Store updated version
                await self.put(key, updated_content)
                
                return updated_content
                
            except json.JSONDecodeError:
                logger.error(f"Cannot perform incremental update on non-JSON content: {key}")
                return None
    
    async def get_project_area_contexts(self, project_area: str) -> Dict[str, str]:
        """Get all contexts for a project area."""
        contexts = {}
        
        if project_area in self.area_caches:
            for key in self.area_caches[project_area]:
                content = await self.get(key)
                if content:
                    contexts[key] = content
        
        return contexts
    
    async def merge_contexts(self, keys: List[str], merge_key: str) -> Optional[str]:
        """Merge multiple contexts into one."""
        contents = []
        
        for key in keys:
            content = await self.get(key)
            if content:
                contents.append(content)
        
        if not contents:
            return None
        
        # Simple merge strategy - concatenate with separators
        merged = "\n\n=== MERGED CONTEXT ===\n\n".join(contents)
        
        # Store merged context
        await self.put(merge_key, merged, metadata={'source_keys': keys})
        
        return merged
    
    async def invalidate(self, key: str):
        """Invalidate a cache entry."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                
                # Remove from area caches
                for area_keys in self.area_caches.values():
                    if key in area_keys:
                        area_keys.remove(key)
                
                # Remove from disk
                cached_file = self.cache_dir / f"{key}.cache"
                if cached_file.exists():
                    cached_file.unlink()
    
    async def invalidate_task_contexts(self, task_id: str):
        """Invalidate all contexts associated with a task."""
        keys_to_invalidate = []
        
        for key, entry in self.cache.items():
            if task_id in entry.task_ids:
                keys_to_invalidate.append(key)
        
        for key in keys_to_invalidate:
            await self.invalidate(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            'total_entries': len(self.cache),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': total_size / self.max_size_bytes,
            'hit_rate': self.stats.hit_rate,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'evictions': self.stats.evictions,
            'compressions': self.stats.compressions,
            'incremental_updates': self.stats.incremental_updates,
            'area_distribution': {
                area: len(keys) for area, keys in self.area_caches.items()
            }
        }
    
    # Private methods
    
    async def _add_to_cache(self, entry: CacheEntry):
        """Add entry to cache with eviction if needed."""
        # Check size constraints
        total_size = sum(e.size_bytes for e in self.cache.values())
        
        while (len(self.cache) >= self.max_entries or 
               total_size + entry.size_bytes > self.max_size_bytes):
            # Evict least recently used
            if self.cache:
                lru_key, lru_entry = self.cache.popitem(last=False)
                total_size -= lru_entry.size_bytes
                self.stats.evictions += 1
                
                logger.debug(f"Evicted cache entry: {lru_key}")
        
        # Add new entry
        self.cache[entry.key] = entry
        self.stats.total_entries = len(self.cache)
        self.stats.total_size_bytes = total_size + entry.size_bytes
    
    def _should_use_incremental(self, new_content: str, old_content: str) -> bool:
        """Decide if incremental update should be used."""
        # Use incremental if the diff is smaller than storing full content
        if len(new_content) < 1024:  # Small content, just store full
            return False
        
        diff = self._create_diff(old_content, new_content)
        diff_size = len(diff.encode('utf-8'))
        new_size = len(new_content.encode('utf-8'))
        
        # Use incremental if diff is less than 50% of new content
        return diff_size < new_size * 0.5
    
    def _create_diff(self, old_content: str, new_content: str) -> str:
        """Create a diff between two content versions."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile='old',
            tofile='new',
            n=3
        ))
        
        return ''.join(diff)
    
    def _apply_diff(self, base_content: str, diff: str) -> str:
        """Apply a diff to reconstruct content."""
        # This is a simplified implementation
        # In production, you'd use a proper patch application library
        return base_content  # Placeholder
    
    def _compress_content(self, content: str) -> Optional[str]:
        """Compress content if beneficial."""
        try:
            import zlib
            compressed = zlib.compress(content.encode('utf-8'), level=6)
            # Base64 encode for storage as string
            import base64
            return base64.b64encode(compressed).decode('utf-8')
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return None
    
    def _decompress_content(self, compressed: str) -> Optional[str]:
        """Decompress content."""
        try:
            import zlib
            import base64
            compressed_bytes = base64.b64decode(compressed.encode('utf-8'))
            return zlib.decompress(compressed_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return None
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_index_file = self.cache_dir / "cache_index.json"
        
        if cache_index_file.exists():
            try:
                with open(cache_index_file, 'r') as f:
                    index = json.load(f)
                
                for key in index.get('keys', []):
                    entry = self._load_entry_from_disk(key)
                    if entry:
                        self.cache[key] = entry
                
                logger.info(f"Loaded {len(self.cache)} cache entries from disk")
                
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")
    
    def _persist_cache(self):
        """Persist cache to disk."""
        try:
            # Save index
            index = {
                'keys': list(self.cache.keys()),
                'timestamp': datetime.now().isoformat(),
                'stats': {
                    'total_entries': self.stats.total_entries,
                    'hit_rate': self.stats.hit_rate
                }
            }
            
            cache_index_file = self.cache_dir / "cache_index.json"
            with open(cache_index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
            # Save individual entries
            for key, entry in self.cache.items():
                self._save_entry_to_disk(entry)
            
            logger.debug(f"Persisted {len(self.cache)} cache entries to disk")
            
        except Exception as e:
            logger.error(f"Failed to persist cache: {e}")
    
    def _save_entry_to_disk(self, entry: CacheEntry):
        """Save a cache entry to disk."""
        try:
            cache_file = self.cache_dir / f"{entry.key}.cache"
            
            # Convert to serializable format
            data = {
                'key': entry.key,
                'content': entry.content,
                'metadata': entry.metadata,
                'created_at': entry.created_at.isoformat(),
                'last_accessed': entry.last_accessed.isoformat(),
                'access_count': entry.access_count,
                'size_bytes': entry.size_bytes,
                'compression_ratio': entry.compression_ratio,
                'project_area': entry.project_area,
                'task_ids': list(entry.task_ids),
                'version': entry.version,
                'parent_key': entry.parent_key,
                'diff_from_parent': entry.diff_from_parent
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save cache entry {entry.key}: {e}")
    
    def _load_entry_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load a cache entry from disk."""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct entry
            entry = CacheEntry(
                key=data['key'],
                content=data['content'],
                metadata=data['metadata'],
                created_at=datetime.fromisoformat(data['created_at']),
                last_accessed=datetime.fromisoformat(data['last_accessed']),
                access_count=data['access_count'],
                size_bytes=data['size_bytes'],
                compression_ratio=data['compression_ratio'],
                project_area=data['project_area'],
                task_ids=set(data['task_ids']),
                version=data['version'],
                parent_key=data['parent_key'],
                diff_from_parent=data['diff_from_parent']
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to load cache entry {key}: {e}")
            return None
    
    async def _cleanup_loop(self):
        """Periodically clean up old entries."""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                async with self._lock:
                    # Remove entries not accessed in last 24 hours
                    cutoff = datetime.now() - timedelta(hours=24)
                    keys_to_remove = []
                    
                    for key, entry in self.cache.items():
                        if entry.last_accessed < cutoff and entry.access_count < 5:
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        await self.invalidate(key)
                    
                    if keys_to_remove:
                        logger.info(f"Cleaned up {len(keys_to_remove)} stale cache entries")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _persist_loop(self):
        """Periodically persist cache to disk."""
        while self.is_running:
            try:
                await asyncio.sleep(self.persist_interval)
                self._persist_cache()
                
            except Exception as e:
                logger.error(f"Error in persist loop: {e}")


# Singleton instance
_context_cache_manager: Optional[ContextCacheManager] = None


def get_context_cache_manager(config: Config) -> ContextCacheManager:
    """Get or create singleton cache manager."""
    global _context_cache_manager
    if _context_cache_manager is None:
        _context_cache_manager = ContextCacheManager(config)
    return _context_cache_manager