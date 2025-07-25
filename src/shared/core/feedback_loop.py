"""Feedback Loop System for continuous improvement between Master Planner and Code Claude."""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Manages feedback between execution results and planning phase."""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.feedback_file = os.path.join(project_path, ".vibe_feedback.json")
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> Dict[str, Any]:
        """Load existing feedback data."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load feedback file: {e}")
        
        return {
            "sessions": [],
            "learned_patterns": {
                "file_mappings": {},
                "tech_stack_corrections": {},
                "structure_patterns": {}
            },
            "execution_stats": {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "common_issues": []
            }
        }
    
    def _save_feedback(self):
        """Save feedback data to file."""
        try:
            # Ensure directory exists
            feedback_dir = os.path.dirname(self.feedback_file)
            if feedback_dir and not os.path.exists(feedback_dir):
                os.makedirs(feedback_dir, exist_ok=True)
            
            # Write to temporary file first
            temp_file = self.feedback_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            
            # Atomic rename
            os.replace(temp_file, self.feedback_file)
        except Exception as e:
            logger.error(f"Could not save feedback file: {e}")
    
    def record_task_execution(self, task_id: str, expected_files: List[str], 
                            actual_files: List[str], success: bool, 
                            tech_stack: Optional[Dict[str, Any]] = None):
        """Record the results of a task execution."""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "expected_files": expected_files,
            "actual_files": actual_files,
            "success": success,
            "tech_stack": tech_stack or {},
            "file_mappings": []
        }
        
        # Learn file mappings
        for expected in expected_files:
            found = False
            for actual in actual_files:
                if self._are_files_related(expected, actual):
                    session_data["file_mappings"].append({
                        "expected": expected,
                        "actual": actual
                    })
                    self._learn_file_mapping(expected, actual)
                    found = True
                    break
            
            if not found:
                logger.info(f"No match found for expected file: {expected}")
        
        # Update execution stats
        self.feedback_data["execution_stats"]["total_tasks"] += 1
        if success:
            self.feedback_data["execution_stats"]["successful_tasks"] += 1
        else:
            self.feedback_data["execution_stats"]["failed_tasks"] += 1
        
        # Add session data
        self.feedback_data["sessions"].append(session_data)
        
        # Keep only last 100 sessions to avoid file bloat
        if len(self.feedback_data["sessions"]) > 100:
            self.feedback_data["sessions"] = self.feedback_data["sessions"][-100:]
        
        self._save_feedback()
    
    def _are_files_related(self, expected: str, actual: str) -> bool:
        """Check if two file paths are related (same file with different extension/path)."""
        # Remove extensions
        expected_base = os.path.splitext(expected)[0]
        actual_base = os.path.splitext(actual)[0]
        
        # Get just the filename
        expected_name = os.path.basename(expected_base)
        actual_name = os.path.basename(actual_base)
        
        # Check if filenames match
        if expected_name == actual_name:
            return True
        
        # Check if paths are similar (ignoring src/ directory differences)
        expected_parts = expected_base.split('/')
        actual_parts = actual_base.split('/')
        
        # Remove 'src' from paths for comparison
        expected_parts = [p for p in expected_parts if p != 'src']
        actual_parts = [p for p in actual_parts if p != 'src']
        
        # Check if the important parts match
        if len(expected_parts) >= 2 and len(actual_parts) >= 2:
            # Compare last 2 parts (usually component/filename)
            return expected_parts[-2:] == actual_parts[-2:]
        
        return False
    
    def _learn_file_mapping(self, expected: str, actual: str):
        """Learn a file mapping pattern."""
        mappings = self.feedback_data["learned_patterns"]["file_mappings"]
        
        # Learn extension mappings
        expected_ext = os.path.splitext(expected)[1]
        actual_ext = os.path.splitext(actual)[1]
        
        if expected_ext and actual_ext and expected_ext != actual_ext:
            ext_key = f"{expected_ext}->{actual_ext}"
            mappings[ext_key] = mappings.get(ext_key, 0) + 1
        
        # Learn path pattern mappings
        if '/frontend/components/' in expected and '/frontend/src/components/' in actual:
            mappings["add_src_to_frontend"] = mappings.get("add_src_to_frontend", 0) + 1
        elif '/frontend/app/' in expected and '/frontend/src/app/' in actual:
            mappings["add_src_to_app"] = mappings.get("add_src_to_app", 0) + 1
    
    def get_learned_mappings(self) -> Dict[str, Any]:
        """Get the learned file mappings and patterns."""
        return self.feedback_data["learned_patterns"]
    
    def suggest_file_correction(self, original_path: str) -> List[str]:
        """Suggest corrections for a file path based on learned patterns."""
        suggestions = [original_path]
        mappings = self.feedback_data["learned_patterns"]["file_mappings"]
        
        # Check extension mappings
        original_ext = os.path.splitext(original_path)[1]
        for mapping, count in mappings.items():
            if count > 2 and '->' in mapping:  # Only use patterns seen more than twice
                from_ext, to_ext = mapping.split('->')
                if original_ext == from_ext:
                    suggestion = original_path[:-len(from_ext)] + to_ext
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)
        
        # Check path mappings
        if mappings.get("add_src_to_frontend", 0) > 2:
            if '/frontend/components/' in original_path and '/src/' not in original_path:
                suggestion = original_path.replace('/frontend/components/', '/frontend/src/components/')
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
        
        if mappings.get("add_src_to_app", 0) > 2:
            if '/frontend/app/' in original_path and '/src/' not in original_path:
                suggestion = original_path.replace('/frontend/app/', '/frontend/src/app/')
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of execution statistics."""
        stats = self.feedback_data["execution_stats"]
        success_rate = 0
        if stats["total_tasks"] > 0:
            success_rate = stats["successful_tasks"] / stats["total_tasks"]
        
        return {
            "total_tasks": stats["total_tasks"],
            "success_rate": success_rate,
            "common_patterns": self._analyze_common_patterns(),
            "recommendations": self._generate_recommendations()
        }
    
    def _analyze_common_patterns(self) -> List[str]:
        """Analyze and return common patterns from feedback."""
        patterns = []
        mappings = self.feedback_data["learned_patterns"]["file_mappings"]
        
        # Find most common mappings
        sorted_mappings = sorted(mappings.items(), key=lambda x: x[1], reverse=True)
        for mapping, count in sorted_mappings[:5]:
            if count > 2:
                patterns.append(f"Pattern '{mapping}' occurred {count} times")
        
        return patterns
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on feedback."""
        recommendations = []
        mappings = self.feedback_data["learned_patterns"]["file_mappings"]
        
        # Extension recommendations
        if mappings.get(".js->.tsx", 0) > 3:
            recommendations.append("Consider using TypeScript (.tsx) by default for React components")
        
        if mappings.get(".js->.ts", 0) > 3:
            recommendations.append("Consider using TypeScript (.ts) by default for utility files")
        
        # Path recommendations
        if mappings.get("add_src_to_frontend", 0) > 3:
            recommendations.append("Frontend files should be placed under 'src/' directory")
        
        # Success rate recommendation
        stats = self.feedback_data["execution_stats"]
        if stats["total_tasks"] > 10:
            success_rate = stats["successful_tasks"] / stats["total_tasks"]
            if success_rate < 0.8:
                recommendations.append(f"Success rate is {success_rate:.1%}. Consider reviewing task definitions")
        
        return recommendations