"""Failure Pattern Database for VIBE

This module provides a database for storing, analyzing, and learning from
failure patterns to improve error recovery and prevention.
"""

import json
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import re
from collections import defaultdict, Counter
import pickle

from shared.core.smart_retry_engine import FailureType, FailureContext
from shared.core.enhanced_logger import get_logger, LogCategory

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """Represents a failure pattern."""
    pattern_id: str
    pattern_type: str  # regex, exact, semantic
    pattern_value: str
    failure_type: FailureType
    
    # Occurrence tracking
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int = 0
    
    # Context patterns
    common_contexts: List[Dict[str, Any]] = field(default_factory=list)
    affected_files: Set[str] = field(default_factory=set)
    affected_functions: Set[str] = field(default_factory=set)
    
    # Solutions
    successful_fixes: List[Dict[str, Any]] = field(default_factory=list)
    failed_fixes: List[Dict[str, Any]] = field(default_factory=list)
    recommended_fix: Optional[str] = None
    
    # Metrics
    average_resolution_time: float = 0.0
    success_rate: float = 0.0
    severity_score: float = 0.0
    
    # Relationships
    related_patterns: List[str] = field(default_factory=list)
    parent_pattern: Optional[str] = None


@dataclass
class FailureInstance:
    """Individual failure occurrence."""
    instance_id: str
    pattern_id: Optional[str]
    timestamp: datetime
    failure_context: FailureContext
    
    # Resolution
    resolution_time: Optional[float] = None
    resolution_method: Optional[str] = None
    was_successful: bool = False
    
    # Task context
    task_id: Optional[str] = None
    task_type: Optional[str] = None
    project_area: Optional[str] = None
    
    # Code context
    code_snippet: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class PatternCluster:
    """Cluster of related patterns."""
    cluster_id: str
    pattern_ids: Set[str]
    cluster_type: str  # syntax, import, resource, etc.
    common_characteristics: Dict[str, Any]
    centroid_pattern: Optional[str] = None
    confidence_score: float = 0.0


class FailurePatternDB:
    """Database for failure pattern storage and analysis."""
    
    def __init__(self, db_path: str = ".vibe_failure_patterns.db"):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        
        # In-memory caches
        self.pattern_cache: Dict[str, FailurePattern] = {}
        self.regex_patterns: List[Tuple[re.Pattern, str]] = []  # (pattern, pattern_id)
        self.pattern_clusters: Dict[str, PatternCluster] = {}
        
        # Statistics
        self.stats = {
            'total_patterns': 0,
            'total_instances': 0,
            'patterns_by_type': defaultdict(int),
            'successful_resolutions': 0,
            'failed_resolutions': 0,
            'average_resolution_time': 0.0
        }
        
        # Learning parameters
        self.min_occurrences_for_pattern = 3
        self.pattern_similarity_threshold = 0.7
        self.cluster_update_interval = 100  # Update clusters every N failures
        self.failures_since_cluster_update = 0
        
        # Enhanced logger
        self.logger = get_logger(
            component="failure_pattern_db",
            session_id=f"pattern_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Initialize database
        self._init_database()
        self._load_patterns()
    
    def _init_database(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS failure_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_value TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                first_seen TIMESTAMP NOT NULL,
                last_seen TIMESTAMP NOT NULL,
                occurrence_count INTEGER DEFAULT 0,
                common_contexts TEXT,
                affected_files TEXT,
                affected_functions TEXT,
                successful_fixes TEXT,
                failed_fixes TEXT,
                recommended_fix TEXT,
                average_resolution_time REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0,
                severity_score REAL DEFAULT 0.0,
                related_patterns TEXT,
                parent_pattern TEXT
            );
            
            CREATE TABLE IF NOT EXISTS failure_instances (
                instance_id TEXT PRIMARY KEY,
                pattern_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                failure_context TEXT NOT NULL,
                resolution_time REAL,
                resolution_method TEXT,
                was_successful BOOLEAN DEFAULT FALSE,
                task_id TEXT,
                task_type TEXT,
                project_area TEXT,
                code_snippet TEXT,
                imports TEXT,
                dependencies TEXT,
                FOREIGN KEY (pattern_id) REFERENCES failure_patterns(pattern_id)
            );
            
            CREATE TABLE IF NOT EXISTS pattern_clusters (
                cluster_id TEXT PRIMARY KEY,
                pattern_ids TEXT NOT NULL,
                cluster_type TEXT NOT NULL,
                common_characteristics TEXT,
                centroid_pattern TEXT,
                confidence_score REAL DEFAULT 0.0
            );
            
            CREATE INDEX IF NOT EXISTS idx_pattern_type ON failure_patterns(failure_type);
            CREATE INDEX IF NOT EXISTS idx_instance_timestamp ON failure_instances(timestamp);
            CREATE INDEX IF NOT EXISTS idx_instance_pattern ON failure_instances(pattern_id);
        """)
        
        self.conn.commit()
    
    def record_failure(self, failure_context: FailureContext,
                      task_id: Optional[str] = None,
                      task_type: Optional[str] = None,
                      project_area: Optional[str] = None,
                      code_snippet: Optional[str] = None) -> str:
        """Record a new failure instance."""
        instance_id = f"fail_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Find matching pattern
        pattern_id = self._find_matching_pattern(failure_context)
        
        # Create instance
        instance = FailureInstance(
            instance_id=instance_id,
            pattern_id=pattern_id,
            timestamp=datetime.now(),
            failure_context=failure_context,
            task_id=task_id,
            task_type=task_type,
            project_area=project_area,
            code_snippet=code_snippet
        )
        
        # Extract imports and dependencies if code snippet provided
        if code_snippet:
            instance.imports = self._extract_imports(code_snippet)
            instance.dependencies = self._extract_dependencies(code_snippet)
        
        # Store instance
        self._store_instance(instance)
        
        # Update or create pattern
        if pattern_id:
            self._update_pattern(pattern_id, instance)
        else:
            # Check if we should create a new pattern
            similar_failures = self._find_similar_failures(failure_context)
            if len(similar_failures) >= self.min_occurrences_for_pattern:
                pattern_id = self._create_pattern_from_failures(
                    failure_context, similar_failures
                )
                instance.pattern_id = pattern_id
                self._update_instance(instance)
        
        # Update statistics
        self.stats['total_instances'] += 1
        self.failures_since_cluster_update += 1
        
        # Update clusters periodically
        if self.failures_since_cluster_update >= self.cluster_update_interval:
            self._update_clusters()
            self.failures_since_cluster_update = 0
        
        logger.info(f"Recorded failure instance {instance_id} with pattern {pattern_id}")
        return instance_id
    
    def update_resolution(self, instance_id: str, 
                         resolution_method: str,
                         was_successful: bool,
                         resolution_time: float):
        """Update failure resolution information."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE failure_instances
            SET resolution_method = ?, was_successful = ?, resolution_time = ?
            WHERE instance_id = ?
        """, (resolution_method, was_successful, resolution_time, instance_id))
        
        # Get pattern_id for this instance
        cursor.execute("""
            SELECT pattern_id, failure_context FROM failure_instances
            WHERE instance_id = ?
        """, (instance_id,))
        
        row = cursor.fetchone()
        if row and row['pattern_id']:
            pattern_id = row['pattern_id']
            
            # Update pattern statistics
            if pattern_id in self.pattern_cache:
                pattern = self.pattern_cache[pattern_id]
                
                fix_info = {
                    'method': resolution_method,
                    'time': resolution_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                if was_successful:
                    pattern.successful_fixes.append(fix_info)
                    self.stats['successful_resolutions'] += 1
                else:
                    pattern.failed_fixes.append(fix_info)
                    self.stats['failed_resolutions'] += 1
                
                # Update average resolution time
                successful_times = [f['time'] for f in pattern.successful_fixes]
                if successful_times:
                    pattern.average_resolution_time = sum(successful_times) / len(successful_times)
                
                # Update success rate
                total_attempts = len(pattern.successful_fixes) + len(pattern.failed_fixes)
                if total_attempts > 0:
                    pattern.success_rate = len(pattern.successful_fixes) / total_attempts
                
                # Determine recommended fix
                if pattern.successful_fixes:
                    # Find most common successful method
                    method_counts = Counter(f['method'] for f in pattern.successful_fixes)
                    pattern.recommended_fix = method_counts.most_common(1)[0][0]
                
                # Save updated pattern
                self._save_pattern(pattern)
        
        self.conn.commit()
        
        self.logger.log(
            level=logging.INFO,
            category=LogCategory.ERROR_RECOVERY,
            message=f"Updated resolution for {instance_id}",
            metadata={
                'method': resolution_method,
                'successful': was_successful,
                'time': resolution_time
            }
        )
    
    def get_pattern_suggestions(self, failure_context: FailureContext) -> List[Dict[str, Any]]:
        """Get suggested fixes based on pattern matching."""
        suggestions = []
        
        # Find matching pattern
        pattern_id = self._find_matching_pattern(failure_context)
        
        if pattern_id and pattern_id in self.pattern_cache:
            pattern = self.pattern_cache[pattern_id]
            
            # Primary suggestion from pattern
            if pattern.recommended_fix:
                suggestions.append({
                    'method': pattern.recommended_fix,
                    'confidence': pattern.success_rate,
                    'average_time': pattern.average_resolution_time,
                    'occurrences': pattern.occurrence_count,
                    'source': 'pattern_match'
                })
            
            # Add other successful fixes
            method_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
            for fix in pattern.successful_fixes:
                method_stats[fix['method']]['count'] += 1
                method_stats[fix['method']]['total_time'] += fix['time']
            
            for method, stats in method_stats.items():
                if method != pattern.recommended_fix:
                    suggestions.append({
                        'method': method,
                        'confidence': stats['count'] / pattern.occurrence_count,
                        'average_time': stats['total_time'] / stats['count'],
                        'occurrences': stats['count'],
                        'source': 'alternative_fix'
                    })
        
        # Check cluster suggestions
        cluster_suggestions = self._get_cluster_suggestions(failure_context)
        suggestions.extend(cluster_suggestions)
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def get_failure_trends(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Analyze failure trends over time."""
        cursor = self.conn.cursor()
        
        # Default to last 7 days
        if not time_window:
            time_window = timedelta(days=7)
        
        start_time = datetime.now() - time_window
        
        # Get failure counts by type
        cursor.execute("""
            SELECT 
                fp.failure_type,
                COUNT(fi.instance_id) as count,
                AVG(fi.resolution_time) as avg_resolution_time,
                SUM(CASE WHEN fi.was_successful THEN 1 ELSE 0 END) as successful_count
            FROM failure_instances fi
            LEFT JOIN failure_patterns fp ON fi.pattern_id = fp.pattern_id
            WHERE fi.timestamp > ?
            GROUP BY fp.failure_type
        """, (start_time,))
        
        type_trends = {}
        for row in cursor.fetchall():
            if row['failure_type']:
                type_trends[row['failure_type']] = {
                    'count': row['count'],
                    'avg_resolution_time': row['avg_resolution_time'] or 0,
                    'success_rate': row['successful_count'] / row['count'] if row['count'] > 0 else 0
                }
        
        # Get top patterns
        cursor.execute("""
            SELECT 
                pattern_id,
                pattern_value,
                failure_type,
                occurrence_count,
                success_rate
            FROM failure_patterns
            WHERE last_seen > ?
            ORDER BY occurrence_count DESC
            LIMIT 10
        """, (start_time,))
        
        top_patterns = []
        for row in cursor.fetchall():
            top_patterns.append({
                'pattern_id': row['pattern_id'],
                'pattern': row['pattern_value'],
                'type': row['failure_type'],
                'occurrences': row['occurrence_count'],
                'success_rate': row['success_rate']
            })
        
        # Time series data
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as count,
                SUM(CASE WHEN was_successful THEN 1 ELSE 0 END) as successful
            FROM failure_instances
            WHERE timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (start_time,))
        
        time_series = []
        for row in cursor.fetchall():
            time_series.append({
                'date': row['date'],
                'total': row['count'],
                'successful': row['successful'],
                'failed': row['count'] - row['successful']
            })
        
        return {
            'type_trends': type_trends,
            'top_patterns': top_patterns,
            'time_series': time_series,
            'summary': {
                'total_failures': sum(t['count'] for t in type_trends.values()),
                'unique_patterns': len(self.pattern_cache),
                'active_clusters': len(self.pattern_clusters)
            }
        }
    
    def export_patterns(self, output_file: str, format: str = 'json'):
        """Export patterns for analysis or sharing."""
        patterns_data = []
        
        for pattern in self.pattern_cache.values():
            pattern_dict = asdict(pattern)
            # Convert sets to lists for JSON serialization
            pattern_dict['affected_files'] = list(pattern.affected_files)
            pattern_dict['affected_functions'] = list(pattern.affected_functions)
            # Convert datetime objects
            pattern_dict['first_seen'] = pattern.first_seen.isoformat()
            pattern_dict['last_seen'] = pattern.last_seen.isoformat()
            patterns_data.append(pattern_dict)
        
        output_path = Path(output_file)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump({
                    'patterns': patterns_data,
                    'clusters': [asdict(c) for c in self.pattern_clusters.values()],
                    'statistics': self.stats,
                    'export_date': datetime.now().isoformat()
                }, f, indent=2)
        elif format == 'csv':
            # Simplified CSV export
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['pattern_id', 'type', 'failure_type', 'occurrences', 
                               'success_rate', 'avg_resolution_time', 'recommended_fix'])
                for pattern in self.pattern_cache.values():
                    writer.writerow([
                        pattern.pattern_id,
                        pattern.pattern_type,
                        pattern.failure_type.value,
                        pattern.occurrence_count,
                        pattern.success_rate,
                        pattern.average_resolution_time,
                        pattern.recommended_fix or ''
                    ])
        
        logger.info(f"Exported {len(patterns_data)} patterns to {output_file}")
    
    def import_patterns(self, input_file: str):
        """Import patterns from file."""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Pattern file not found: {input_file}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        imported_count = 0
        
        for pattern_data in data.get('patterns', []):
            # Convert back from dict
            pattern_data['first_seen'] = datetime.fromisoformat(pattern_data['first_seen'])
            pattern_data['last_seen'] = datetime.fromisoformat(pattern_data['last_seen'])
            pattern_data['failure_type'] = FailureType(pattern_data['failure_type'])
            pattern_data['affected_files'] = set(pattern_data['affected_files'])
            pattern_data['affected_functions'] = set(pattern_data['affected_functions'])
            
            pattern = FailurePattern(**pattern_data)
            
            # Check if pattern already exists
            if pattern.pattern_id not in self.pattern_cache:
                self.pattern_cache[pattern.pattern_id] = pattern
                self._save_pattern(pattern)
                imported_count += 1
        
        logger.info(f"Imported {imported_count} new patterns from {input_file}")
    
    # Private methods
    
    def _load_patterns(self):
        """Load patterns from database into memory."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM failure_patterns")
        
        for row in cursor.fetchall():
            pattern = self._pattern_from_row(row)
            self.pattern_cache[pattern.pattern_id] = pattern
            
            # Cache regex patterns
            if pattern.pattern_type == 'regex':
                try:
                    regex = re.compile(pattern.pattern_value, re.IGNORECASE)
                    self.regex_patterns.append((regex, pattern.pattern_id))
                except re.error:
                    logger.error(f"Invalid regex pattern: {pattern.pattern_value}")
        
        self.stats['total_patterns'] = len(self.pattern_cache)
        
        # Load clusters
        cursor.execute("SELECT * FROM pattern_clusters")
        for row in cursor.fetchall():
            cluster = PatternCluster(
                cluster_id=row['cluster_id'],
                pattern_ids=set(json.loads(row['pattern_ids'])),
                cluster_type=row['cluster_type'],
                common_characteristics=json.loads(row['common_characteristics']),
                centroid_pattern=row['centroid_pattern'],
                confidence_score=row['confidence_score']
            )
            self.pattern_clusters[cluster.cluster_id] = cluster
        
        logger.info(f"Loaded {len(self.pattern_cache)} patterns and {len(self.pattern_clusters)} clusters")
    
    def _find_matching_pattern(self, failure_context: FailureContext) -> Optional[str]:
        """Find pattern matching the failure context."""
        error_message = failure_context.error_message
        
        # Try exact match first
        for pattern in self.pattern_cache.values():
            if pattern.pattern_type == 'exact' and pattern.pattern_value == error_message:
                return pattern.pattern_id
        
        # Try regex patterns
        for regex, pattern_id in self.regex_patterns:
            if regex.search(error_message):
                return pattern_id
        
        # Try semantic matching (simplified)
        for pattern in self.pattern_cache.values():
            if pattern.pattern_type == 'semantic':
                if self._semantic_similarity(error_message, pattern.pattern_value) > self.pattern_similarity_threshold:
                    return pattern.pattern_id
        
        return None
    
    def _find_similar_failures(self, failure_context: FailureContext) -> List[FailureInstance]:
        """Find similar unmatched failures."""
        cursor = self.conn.cursor()
        
        # Find failures without patterns of the same type
        cursor.execute("""
            SELECT instance_id, failure_context
            FROM failure_instances
            WHERE pattern_id IS NULL
            AND timestamp > datetime('now', '-30 days')
        """)
        
        similar = []
        target_type = failure_context.failure_type
        
        for row in cursor.fetchall():
            try:
                stored_context = pickle.loads(row['failure_context'])
                if stored_context.failure_type == target_type:
                    similarity = self._calculate_similarity(failure_context, stored_context)
                    if similarity > self.pattern_similarity_threshold:
                        # Load full instance
                        instance = self._load_instance(row['instance_id'])
                        if instance:
                            similar.append(instance)
            except Exception as e:
                logger.error(f"Error comparing failures: {e}")
        
        return similar
    
    def _create_pattern_from_failures(self, failure_context: FailureContext,
                                    similar_failures: List[FailureInstance]) -> str:
        """Create a new pattern from similar failures."""
        pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Determine pattern type and value
        all_messages = [failure_context.error_message] + [
            f.failure_context.error_message for f in similar_failures
        ]
        
        pattern_type, pattern_value = self._extract_pattern(all_messages)
        
        # Create pattern
        pattern = FailurePattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            pattern_value=pattern_value,
            failure_type=failure_context.failure_type,
            first_seen=min(f.timestamp for f in similar_failures),
            last_seen=datetime.now(),
            occurrence_count=len(similar_failures) + 1
        )
        
        # Extract common contexts
        all_contexts = [failure_context] + [f.failure_context for f in similar_failures]
        pattern.common_contexts = self._extract_common_contexts(all_contexts)
        
        # Extract affected files and functions
        for instance in similar_failures:
            if instance.failure_context.file_path:
                pattern.affected_files.add(instance.failure_context.file_path)
            if instance.failure_context.function_name:
                pattern.affected_functions.add(instance.failure_context.function_name)
        
        # Calculate initial severity
        pattern.severity_score = self._calculate_severity(pattern)
        
        # Save pattern
        self.pattern_cache[pattern_id] = pattern
        self._save_pattern(pattern)
        
        # Update instances to reference this pattern
        for instance in similar_failures:
            instance.pattern_id = pattern_id
            self._update_instance(instance)
        
        logger.info(f"Created new pattern {pattern_id} from {len(similar_failures)} similar failures")
        
        return pattern_id
    
    def _extract_pattern(self, messages: List[str]) -> Tuple[str, str]:
        """Extract pattern from similar messages."""
        if len(messages) == 1:
            return 'exact', messages[0]
        
        # Find common substrings
        common_parts = []
        first_msg = messages[0]
        
        # Simple approach - find common prefixes/suffixes
        prefix = ""
        for i in range(min(len(m) for m in messages)):
            if all(m[i] == first_msg[i] for m in messages):
                prefix += first_msg[i]
            else:
                break
        
        if len(prefix) > 10:  # Meaningful prefix
            # Create regex pattern
            pattern_value = re.escape(prefix) + ".*"
            return 'regex', pattern_value
        
        # If no clear pattern, use semantic
        return 'semantic', first_msg
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts."""
        # Simplified - in production would use embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_similarity(self, context1: FailureContext, context2: FailureContext) -> float:
        """Calculate similarity between failure contexts."""
        score = 0.0
        
        # Same failure type
        if context1.failure_type == context2.failure_type:
            score += 0.3
        
        # Similar error messages
        msg_similarity = self._semantic_similarity(context1.error_message, context2.error_message)
        score += msg_similarity * 0.4
        
        # Same file
        if context1.file_path and context1.file_path == context2.file_path:
            score += 0.2
        
        # Same function
        if context1.function_name and context1.function_name == context2.function_name:
            score += 0.1
        
        return score
    
    def _extract_common_contexts(self, contexts: List[FailureContext]) -> List[Dict[str, Any]]:
        """Extract common context patterns."""
        common = []
        
        # Find common error details
        detail_keys = defaultdict(Counter)
        for context in contexts:
            for key, value in context.error_details.items():
                if isinstance(value, (str, int, float, bool)):
                    detail_keys[key][str(value)] += 1
        
        for key, value_counts in detail_keys.items():
            if len(value_counts) == 1:  # All have same value
                common.append({
                    'type': 'constant_detail',
                    'key': key,
                    'value': list(value_counts.keys())[0]
                })
            elif len(value_counts) < len(contexts) / 2:  # Few distinct values
                common.append({
                    'type': 'common_values',
                    'key': key,
                    'values': list(value_counts.keys())
                })
        
        return common
    
    def _calculate_severity(self, pattern: FailurePattern) -> float:
        """Calculate pattern severity score."""
        severity = 0.0
        
        # Base severity by failure type
        type_severities = {
            FailureType.SYNTAX_ERROR: 0.3,
            FailureType.IMPORT_ERROR: 0.4,
            FailureType.RUNTIME_ERROR: 0.6,
            FailureType.TIMEOUT: 0.5,
            FailureType.RESOURCE_LIMIT: 0.7,
            FailureType.PERMISSION_ERROR: 0.8,
            FailureType.NETWORK_ERROR: 0.4
        }
        
        severity = type_severities.get(pattern.failure_type, 0.5)
        
        # Adjust by occurrence frequency
        if pattern.occurrence_count > 10:
            severity += 0.1
        if pattern.occurrence_count > 50:
            severity += 0.1
        
        # Adjust by success rate
        if pattern.success_rate < 0.5:
            severity += 0.2
        
        return min(severity, 1.0)
    
    def _update_clusters(self):
        """Update pattern clusters."""
        # Group patterns by failure type
        patterns_by_type = defaultdict(list)
        for pattern in self.pattern_cache.values():
            patterns_by_type[pattern.failure_type].append(pattern)
        
        # Create or update clusters for each type
        for failure_type, patterns in patterns_by_type.items():
            if len(patterns) < 3:
                continue
            
            # Simple clustering based on pattern similarity
            clusters = self._cluster_patterns(patterns)
            
            for cluster_patterns in clusters:
                cluster_id = f"cluster_{failure_type.value}_{len(self.pattern_clusters)}"
                
                cluster = PatternCluster(
                    cluster_id=cluster_id,
                    pattern_ids={p.pattern_id for p in cluster_patterns},
                    cluster_type=failure_type.value,
                    common_characteristics=self._extract_cluster_characteristics(cluster_patterns),
                    confidence_score=len(cluster_patterns) / len(patterns)
                )
                
                # Find centroid pattern (most representative)
                if cluster_patterns:
                    cluster.centroid_pattern = max(
                        cluster_patterns,
                        key=lambda p: p.occurrence_count
                    ).pattern_id
                
                self.pattern_clusters[cluster_id] = cluster
                self._save_cluster(cluster)
        
        logger.info(f"Updated clusters: {len(self.pattern_clusters)} total")
    
    def _cluster_patterns(self, patterns: List[FailurePattern]) -> List[List[FailurePattern]]:
        """Simple pattern clustering."""
        clusters = []
        used = set()
        
        for pattern in patterns:
            if pattern.pattern_id in used:
                continue
            
            cluster = [pattern]
            used.add(pattern.pattern_id)
            
            # Find similar patterns
            for other in patterns:
                if other.pattern_id in used:
                    continue
                
                # Compare patterns
                if self._patterns_similar(pattern, other):
                    cluster.append(other)
                    used.add(other.pattern_id)
            
            if len(cluster) >= 2:
                clusters.append(cluster)
        
        return clusters
    
    def _patterns_similar(self, p1: FailurePattern, p2: FailurePattern) -> bool:
        """Check if two patterns are similar."""
        # Same failure type
        if p1.failure_type != p2.failure_type:
            return False
        
        # Similar pattern values
        if p1.pattern_type == p2.pattern_type:
            if p1.pattern_type == 'exact':
                return p1.pattern_value == p2.pattern_value
            else:
                return self._semantic_similarity(p1.pattern_value, p2.pattern_value) > 0.7
        
        # Check affected files/functions overlap
        file_overlap = len(p1.affected_files.intersection(p2.affected_files))
        func_overlap = len(p1.affected_functions.intersection(p2.affected_functions))
        
        return (file_overlap + func_overlap) > 2
    
    def _extract_cluster_characteristics(self, patterns: List[FailurePattern]) -> Dict[str, Any]:
        """Extract common characteristics of pattern cluster."""
        characteristics = {
            'pattern_count': len(patterns),
            'total_occurrences': sum(p.occurrence_count for p in patterns),
            'average_success_rate': sum(p.success_rate for p in patterns) / len(patterns),
            'common_files': list(set.intersection(*[p.affected_files for p in patterns if p.affected_files])),
            'common_functions': list(set.intersection(*[p.affected_functions for p in patterns if p.affected_functions]))
        }
        
        return characteristics
    
    def _get_cluster_suggestions(self, failure_context: FailureContext) -> List[Dict[str, Any]]:
        """Get suggestions from pattern clusters."""
        suggestions = []
        
        for cluster in self.pattern_clusters.values():
            if cluster.cluster_type == failure_context.failure_type.value:
                # Check if failure matches cluster characteristics
                score = self._score_cluster_match(failure_context, cluster)
                
                if score > 0.5:
                    # Get suggestions from cluster patterns
                    cluster_patterns = [
                        self.pattern_cache[pid] 
                        for pid in cluster.pattern_ids 
                        if pid in self.pattern_cache
                    ]
                    
                    # Aggregate fixes from cluster
                    method_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
                    
                    for pattern in cluster_patterns:
                        for fix in pattern.successful_fixes:
                            method_stats[fix['method']]['count'] += 1
                            method_stats[fix['method']]['total_time'] += fix['time']
                    
                    for method, stats in method_stats.items():
                        suggestions.append({
                            'method': method,
                            'confidence': score * (stats['count'] / len(cluster_patterns)),
                            'average_time': stats['total_time'] / stats['count'],
                            'occurrences': stats['count'],
                            'source': f'cluster_{cluster.cluster_id}'
                        })
        
        return suggestions
    
    def _score_cluster_match(self, failure_context: FailureContext, cluster: PatternCluster) -> float:
        """Score how well a failure matches a cluster."""
        score = 0.0
        
        # Check file match
        if failure_context.file_path:
            common_files = cluster.common_characteristics.get('common_files', [])
            if failure_context.file_path in common_files:
                score += 0.3
        
        # Check function match
        if failure_context.function_name:
            common_funcs = cluster.common_characteristics.get('common_functions', [])
            if failure_context.function_name in common_funcs:
                score += 0.2
        
        # Base confidence from cluster
        score += cluster.confidence_score * 0.5
        
        return score
    
    def _extract_imports(self, code_snippet: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        
        # Python imports
        import_patterns = [
            r'^import\s+(\S+)',
            r'^from\s+(\S+)\s+import',
        ]
        
        for line in code_snippet.split('\n'):
            for pattern in import_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    imports.append(match.group(1))
        
        return imports
    
    def _extract_dependencies(self, code_snippet: str) -> List[str]:
        """Extract dependencies from code."""
        # Simplified - would need language-specific parsing
        return self._extract_imports(code_snippet)
    
    def _store_instance(self, instance: FailureInstance):
        """Store failure instance in database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO failure_instances (
                instance_id, pattern_id, timestamp, failure_context,
                resolution_time, resolution_method, was_successful,
                task_id, task_type, project_area, code_snippet,
                imports, dependencies
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            instance.instance_id,
            instance.pattern_id,
            instance.timestamp,
            pickle.dumps(instance.failure_context),
            instance.resolution_time,
            instance.resolution_method,
            instance.was_successful,
            instance.task_id,
            instance.task_type,
            instance.project_area,
            instance.code_snippet,
            json.dumps(instance.imports),
            json.dumps(instance.dependencies)
        ))
        
        self.conn.commit()
    
    def _update_instance(self, instance: FailureInstance):
        """Update failure instance."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE failure_instances
            SET pattern_id = ?
            WHERE instance_id = ?
        """, (instance.pattern_id, instance.instance_id))
        self.conn.commit()
    
    def _load_instance(self, instance_id: str) -> Optional[FailureInstance]:
        """Load instance from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM failure_instances WHERE instance_id = ?", (instance_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return FailureInstance(
            instance_id=row['instance_id'],
            pattern_id=row['pattern_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            failure_context=pickle.loads(row['failure_context']),
            resolution_time=row['resolution_time'],
            resolution_method=row['resolution_method'],
            was_successful=row['was_successful'],
            task_id=row['task_id'],
            task_type=row['task_type'],
            project_area=row['project_area'],
            code_snippet=row['code_snippet'],
            imports=json.loads(row['imports']) if row['imports'] else [],
            dependencies=json.loads(row['dependencies']) if row['dependencies'] else []
        )
    
    def _save_pattern(self, pattern: FailurePattern):
        """Save pattern to database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO failure_patterns (
                pattern_id, pattern_type, pattern_value, failure_type,
                first_seen, last_seen, occurrence_count, common_contexts,
                affected_files, affected_functions, successful_fixes,
                failed_fixes, recommended_fix, average_resolution_time,
                success_rate, severity_score, related_patterns, parent_pattern
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_id,
            pattern.pattern_type,
            pattern.pattern_value,
            pattern.failure_type.value,
            pattern.first_seen,
            pattern.last_seen,
            pattern.occurrence_count,
            json.dumps(pattern.common_contexts),
            json.dumps(list(pattern.affected_files)),
            json.dumps(list(pattern.affected_functions)),
            json.dumps(pattern.successful_fixes),
            json.dumps(pattern.failed_fixes),
            pattern.recommended_fix,
            pattern.average_resolution_time,
            pattern.success_rate,
            pattern.severity_score,
            json.dumps(pattern.related_patterns),
            pattern.parent_pattern
        ))
        
        self.conn.commit()
    
    def _update_pattern(self, pattern_id: str, instance: FailureInstance):
        """Update pattern with new instance data."""
        if pattern_id not in self.pattern_cache:
            return
        
        pattern = self.pattern_cache[pattern_id]
        pattern.last_seen = instance.timestamp
        pattern.occurrence_count += 1
        
        # Update affected files/functions
        if instance.failure_context.file_path:
            pattern.affected_files.add(instance.failure_context.file_path)
        if instance.failure_context.function_name:
            pattern.affected_functions.add(instance.failure_context.function_name)
        
        # Update severity
        pattern.severity_score = self._calculate_severity(pattern)
        
        self._save_pattern(pattern)
    
    def _save_cluster(self, cluster: PatternCluster):
        """Save cluster to database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO pattern_clusters (
                cluster_id, pattern_ids, cluster_type,
                common_characteristics, centroid_pattern, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            cluster.cluster_id,
            json.dumps(list(cluster.pattern_ids)),
            cluster.cluster_type,
            json.dumps(cluster.common_characteristics),
            cluster.centroid_pattern,
            cluster.confidence_score
        ))
        
        self.conn.commit()
    
    def _pattern_from_row(self, row: sqlite3.Row) -> FailurePattern:
        """Create pattern from database row."""
        return FailurePattern(
            pattern_id=row['pattern_id'],
            pattern_type=row['pattern_type'],
            pattern_value=row['pattern_value'],
            failure_type=FailureType(row['failure_type']),
            first_seen=datetime.fromisoformat(row['first_seen']),
            last_seen=datetime.fromisoformat(row['last_seen']),
            occurrence_count=row['occurrence_count'],
            common_contexts=json.loads(row['common_contexts']) if row['common_contexts'] else [],
            affected_files=set(json.loads(row['affected_files'])) if row['affected_files'] else set(),
            affected_functions=set(json.loads(row['affected_functions'])) if row['affected_functions'] else set(),
            successful_fixes=json.loads(row['successful_fixes']) if row['successful_fixes'] else [],
            failed_fixes=json.loads(row['failed_fixes']) if row['failed_fixes'] else [],
            recommended_fix=row['recommended_fix'],
            average_resolution_time=row['average_resolution_time'],
            success_rate=row['success_rate'],
            severity_score=row['severity_score'],
            related_patterns=json.loads(row['related_patterns']) if row['related_patterns'] else [],
            parent_pattern=row['parent_pattern']
        )


# Singleton instance
_failure_pattern_db: Optional[FailurePatternDB] = None


def get_failure_pattern_db() -> FailurePatternDB:
    """Get or create singleton pattern database."""
    global _failure_pattern_db
    if _failure_pattern_db is None:
        _failure_pattern_db = FailurePatternDB()
    return _failure_pattern_db