"""JSON Repair Utilities - Fix common JSON parsing issues."""

import json
import re
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class JSONRepair:
    """Repair malformed JSON strings with various strategies."""
    
    @staticmethod
    def repair(json_str: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Attempt to repair and parse JSON string.
        
        Returns:
            Tuple of (parsed_dict, was_repaired)
        """
        # First try parsing as-is
        try:
            return json.loads(json_str), False
        except json.JSONDecodeError:
            pass
        
        # Apply repair strategies
        repaired = json_str
        
        # Strategy 1: Extract JSON from markdown code blocks
        repaired = JSONRepair._extract_from_codeblock(repaired)
        
        # Strategy 2: Fix truncated strings
        repaired = JSONRepair._fix_truncated_strings(repaired)
        
        # Strategy 3: Balance braces
        repaired = JSONRepair._balance_braces(repaired)
        
        # Strategy 4: Fix common syntax errors
        repaired = JSONRepair._fix_syntax_errors(repaired)
        
        # Strategy 5: Remove trailing content after valid JSON
        repaired = JSONRepair._extract_valid_json_object(repaired)
        
        # Try parsing repaired JSON
        try:
            return json.loads(repaired), True
        except json.JSONDecodeError as e:
            logger.debug(f"JSON repair failed: {e}")
            
            # Last resort: Try to extract and parse partial JSON
            partial = JSONRepair._extract_partial_json(json_str)
            if partial:
                try:
                    return json.loads(partial), True
                except:
                    pass
            
            return None, False
    
    @staticmethod
    def _extract_from_codeblock(text: str) -> str:
        """Extract JSON from markdown code blocks."""
        # Look for ```json blocks
        json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Look for generic code blocks
        code_match = re.search(r'```\s*\n(\{.*?\})\s*\n```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        return text
    
    @staticmethod
    def _fix_truncated_strings(text: str) -> str:
        """Fix strings that were truncated mid-way."""
        # Find unclosed strings
        in_string = False
        escape_next = False
        fixed = []
        quote_char = None
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                fixed.append(char)
                continue
                
            if char == '\\':
                escape_next = True
                fixed.append(char)
                continue
            
            if not in_string and char in ['"', "'"]:
                in_string = True
                quote_char = char
                fixed.append(char)
            elif in_string and char == quote_char:
                in_string = False
                quote_char = None
                fixed.append(char)
            else:
                fixed.append(char)
        
        # If string is still open, close it
        if in_string:
            fixed.append(quote_char)
            # Add necessary closing braces/brackets
            result = ''.join(fixed)
            
            # Count open braces/brackets
            brace_count = result.count('{') - result.count('}')
            bracket_count = result.count('[') - result.count(']')
            
            # Add closing characters
            if bracket_count > 0:
                result += ']' * bracket_count
            if brace_count > 0:
                result += '}' * brace_count
                
            return result
        
        return ''.join(fixed)
    
    @staticmethod
    def _balance_braces(text: str) -> str:
        """Balance curly braces and square brackets."""
        brace_count = text.count('{') - text.count('}')
        bracket_count = text.count('[') - text.count(']')
        
        if brace_count > 0:
            text += '}' * brace_count
        elif brace_count < 0:
            # Remove extra closing braces
            for _ in range(abs(brace_count)):
                text = text.rsplit('}', 1)[0]
        
        if bracket_count > 0:
            text += ']' * bracket_count
        elif bracket_count < 0:
            # Remove extra closing brackets
            for _ in range(abs(bracket_count)):
                text = text.rsplit(']', 1)[0]
        
        return text
    
    @staticmethod
    def _fix_syntax_errors(text: str) -> str:
        """Fix common JSON syntax errors."""
        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Fix missing commas between elements
        text = re.sub(r'}\s*{', '},{', text)
        text = re.sub(r']\s*\[', '],[', text)
        text = re.sub(r'"\s*"', '","', text)
        
        # Fix single quotes (convert to double quotes)
        # This is tricky - need to avoid replacing quotes inside strings
        # Simple approach: replace ' with " if it's preceded/followed by : or ,
        text = re.sub(r"(?<=[:{,\[])\s*'", ' "', text)
        text = re.sub(r"'\s*(?=[,}\]])", '" ', text)
        
        return text
    
    @staticmethod
    def _extract_valid_json_object(text: str) -> str:
        """Extract the first valid JSON object from text."""
        # Find the first { and match it with its closing }
        start = text.find('{')
        if start == -1:
            return text
        
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1]
        
        # If we didn't find matching brace, return original
        return text
    
    @staticmethod
    def _extract_partial_json(text: str) -> Optional[str]:
        """Try to extract a valid partial JSON object."""
        # Look for common JSON patterns
        patterns = [
            r'(\{[^{}]*"situation_analysis"[^{}]*:[^{}]*"[^"]*"[^{}]*\})',
            r'(\{[^{}]*"execution_strategy"[^{}]*:[^{}]*"[^"]*"[^{}]*\})',
            r'(\{[^{}]*:[^{}]*\})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json.loads(match.group(1))
                    return match.group(1)
                except:
                    continue
        
        return None
    
    @staticmethod
    def create_fallback_json(task_id: str, task_description: str) -> Dict[str, Any]:
        """Create a valid fallback JSON response."""
        return {
            "situation_analysis": f"Executing task {task_id}: {task_description}",
            "execution_strategy": "Standard implementation approach",
            "step_by_step_plan": [
                {
                    "step": 1,
                    "action": "Analyze requirements",
                    "verification": "Requirements understood"
                },
                {
                    "step": 2,
                    "action": "Implement solution",
                    "verification": "Code complete and functional"
                },
                {
                    "step": 3,
                    "action": "Verify implementation",
                    "verification": "All acceptance criteria met"
                }
            ],
            "code_claude_cli_instructions": f"Implement the task '{task_description}' following best practices",
            "success_criteria": [
                "All required files created/modified",
                "Code follows project conventions",
                "Implementation is functional"
            ],
            "risk_factors": ["Potential complexity in requirements"],
            "cli_advantages": ["Direct file manipulation", "Integrated development environment"]
        }