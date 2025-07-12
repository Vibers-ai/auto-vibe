"""Response Validation Layer for AI-generated content in VIBE."""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import jsonschema
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning" 
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    issues: List[Dict[str, Any]]
    corrected_response: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0  # 0-1 confidence in the response
    
    def add_issue(self, severity: ValidationSeverity, message: str, field: str = None, suggestion: str = None):
        """Add a validation issue."""
        self.issues.append({
            "severity": severity.value,
            "message": message,
            "field": field,
            "suggestion": suggestion
        })
        
        if severity == ValidationSeverity.ERROR:
            self.is_valid = False


class ResponseValidator:
    """Validates AI responses for consistency, completeness, and correctness."""
    
    def __init__(self):
        self.schemas = self._load_validation_schemas()
        self.common_patterns = self._load_common_patterns()
    
    def _load_validation_schemas(self) -> Dict[str, Dict]:
        """Load JSON schemas for different response types."""
        return {
            "task_evaluation": {
                "type": "object",
                "required": ["success", "feedback"],
                "properties": {
                    "success": {"type": "boolean"},
                    "feedback": {"type": "string", "minLength": 10},
                    "quality_score": {"type": "number", "minimum": 0, "maximum": 10},
                    "completion_percentage": {"type": "number", "minimum": 0, "maximum": 100},
                    "ready_for_final_verification": {"type": "boolean"}
                }
            },
            "execution_plan": {
                "type": "object", 
                "required": ["commands", "situation_analysis"],
                "properties": {
                    "commands": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1}
                    },
                    "situation_analysis": {"type": "string", "minLength": 20},
                    "expected_outcome": {"type": "string"},
                    "risk_assessment": {"type": "string"}
                }
            },
            "file_generation": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {"type": "string", "pattern": r"^[a-zA-Z0-9_\-./]+\.[a-zA-Z]+$"},
                    "content": {"type": "string", "minLength": 1},
                    "file_type": {"type": "string"},
                    "description": {"type": "string"}
                }
            }
        }
    
    def _load_common_patterns(self) -> Dict[str, re.Pattern]:
        """Load common regex patterns for validation."""
        return {
            "file_path": re.compile(r'^[a-zA-Z0-9_\-./]+\.[a-zA-Z]+$'),
            "function_name": re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
            "class_name": re.compile(r'^[A-Z][a-zA-Z0-9_]*$'),
            "variable_name": re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
            "url": re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            "email": re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')
        }
    
    def validate_response(self, response: Union[str, Dict], response_type: str = "generic") -> ValidationResult:
        """Main validation method for AI responses."""
        result = ValidationResult(is_valid=True, issues=[])
        
        # Convert string response to dict if needed
        parsed_response = self._parse_response(response, result)
        if not parsed_response:
            return result
        
        # Schema validation
        self._validate_schema(parsed_response, response_type, result)
        
        # Content validation
        self._validate_content_quality(parsed_response, result)
        
        # Consistency validation  
        self._validate_consistency(parsed_response, result)
        
        # Security validation
        self._validate_security(parsed_response, result)
        
        # Calculate confidence score
        result.confidence_score = self._calculate_confidence_score(result)
        
        # Attempt auto-correction for minor issues
        if not result.is_valid:
            result.corrected_response = self._attempt_auto_correction(parsed_response, result)
        
        return result
    
    def _parse_response(self, response: Union[str, Dict], result: ValidationResult) -> Optional[Dict]:
        """Parse response string to dictionary."""
        if isinstance(response, dict):
            return response
        
        if not isinstance(response, str):
            result.add_issue(ValidationSeverity.ERROR, f"Response must be string or dict, got {type(response)}")
            return None
        
        # Try multiple parsing strategies
        parsing_strategies = [
            self._parse_direct_json,
            self._parse_code_block_json,
            self._parse_partial_json,
            self._parse_key_value_pairs
        ]
        
        for strategy in parsing_strategies:
            try:
                parsed = strategy(response)
                if parsed:
                    result.add_issue(ValidationSeverity.INFO, f"Parsed using {strategy.__name__}")
                    return parsed
            except Exception as e:
                logger.debug(f"Parsing strategy {strategy.__name__} failed: {e}")
        
        result.add_issue(ValidationSeverity.ERROR, "Could not parse response as valid JSON or structured data")
        return None
    
    def _parse_direct_json(self, text: str) -> Optional[Dict]:
        """Try direct JSON parsing."""
        return json.loads(text.strip())
    
    def _parse_code_block_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from markdown code blocks."""
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return None
    
    def _parse_partial_json(self, text: str) -> Optional[Dict]:
        """Find and parse the first complete JSON object."""
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        brace_count = 0
        end_idx = start_idx
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)
    
    def _parse_key_value_pairs(self, text: str) -> Optional[Dict]:
        """Extract key-value pairs manually as last resort."""
        result = {}
        
        # Common patterns for key-value extraction
        patterns = [
            r'"(\w+)"\s*:\s*"([^"]*)"',  # "key": "value"
            r'"(\w+)"\s*:\s*(\w+)',      # "key": value  
            r'(\w+)\s*:\s*"([^"]*)"',    # key: "value"
            r'(\w+)\s*:\s*(\w+)',        # key: value
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                # Try to convert value to appropriate type
                if value.lower() == 'true':
                    result[key] = True
                elif value.lower() == 'false':
                    result[key] = False
                elif value.isdigit():
                    result[key] = int(value)
                else:
                    result[key] = value
        
        return result if result else None
    
    def _validate_schema(self, response: Dict, response_type: str, result: ValidationResult):
        """Validate response against JSON schema."""
        schema = self.schemas.get(response_type)
        if not schema:
            result.add_issue(ValidationSeverity.WARNING, f"No schema defined for response type: {response_type}")
            return
        
        try:
            jsonschema.validate(response, schema)
            result.add_issue(ValidationSeverity.INFO, "Schema validation passed")
        except jsonschema.ValidationError as e:
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Schema validation failed: {e.message}",
                field=e.path[-1] if e.path else None,
                suggestion=self._get_schema_fix_suggestion(e)
            )
    
    def _validate_content_quality(self, response: Dict, result: ValidationResult):
        """Validate content quality and completeness."""
        
        # Check for empty or placeholder content
        for key, value in response.items():
            if isinstance(value, str):
                if not value.strip():
                    result.add_issue(ValidationSeverity.ERROR, f"Empty value for {key}")
                elif value.strip().lower() in ['todo', 'placeholder', 'tbd', 'fix me']:
                    result.add_issue(ValidationSeverity.WARNING, f"Placeholder content in {key}: {value}")
                elif len(value.strip()) < 3:
                    result.add_issue(ValidationSeverity.WARNING, f"Very short content in {key}: {value}")
        
        # Check for meaningful feedback
        if 'feedback' in response:
            feedback = response['feedback'].lower()
            generic_phrases = ['good', 'ok', 'fine', 'done', 'complete']
            if any(phrase in feedback for phrase in generic_phrases) and len(feedback) < 20:
                result.add_issue(ValidationSeverity.WARNING, "Feedback appears too generic or brief")
    
    def _validate_consistency(self, response: Dict, result: ValidationResult):
        """Validate internal consistency of the response."""
        
        # Check success/failure consistency
        if 'success' in response and 'feedback' in response:
            success = response['success']
            feedback = response['feedback'].lower()
            
            success_indicators = ['success', 'complete', 'done', 'passed']
            failure_indicators = ['fail', 'error', 'issue', 'problem', 'missing']
            
            feedback_suggests_success = any(word in feedback for word in success_indicators)
            feedback_suggests_failure = any(word in feedback for word in failure_indicators)
            
            if success and feedback_suggests_failure:
                result.add_issue(
                    ValidationSeverity.WARNING, 
                    "Success flag is True but feedback suggests failure",
                    suggestion="Check if success should be False"
                )
            elif not success and feedback_suggests_success:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Success flag is False but feedback suggests success", 
                    suggestion="Check if success should be True"
                )
        
        # Check quality score consistency
        if 'success' in response and 'quality_score' in response:
            success = response['success']
            quality_score = response.get('quality_score', 0)
            
            if success and quality_score < 5:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Success is True but quality score is low: {quality_score}"
                )
            elif not success and quality_score > 7:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Success is False but quality score is high: {quality_score}"
                )
    
    def _validate_security(self, response: Dict, result: ValidationResult):
        """Validate response for security issues."""
        
        # Check for potential code injection in file paths
        if 'file_path' in response:
            file_path = response['file_path']
            dangerous_patterns = ['../', '.\\', '/etc/', '/root/', 'rm -rf', 'sudo']
            
            for pattern in dangerous_patterns:
                if pattern in file_path:
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        f"Potentially dangerous file path: {file_path}",
                        field="file_path"
                    )
        
        # Check for sensitive information in content
        sensitive_patterns = [
            r'password\s*[=:]\s*["\']?[^"\'\s]+',
            r'api[_-]?key\s*[=:]\s*["\']?[^"\'\s]+',
            r'secret\s*[=:]\s*["\']?[^"\'\s]+',
            r'token\s*[=:]\s*["\']?[^"\'\s]+'
        ]
        
        content_str = json.dumps(response)
        for pattern in sensitive_patterns:
            if re.search(pattern, content_str, re.IGNORECASE):
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Potential sensitive information detected in response"
                )
    
    def _calculate_confidence_score(self, result: ValidationResult) -> float:
        """Calculate confidence score based on validation results."""
        if not result.issues:
            return 1.0
        
        score = 1.0
        
        for issue in result.issues:
            severity = issue['severity']
            if severity == 'error':
                score -= 0.3
            elif severity == 'warning':
                score -= 0.1
            # info issues don't reduce score
        
        return max(0.0, score)
    
    def _attempt_auto_correction(self, response: Dict, result: ValidationResult) -> Optional[Dict]:
        """Attempt to auto-correct minor issues."""
        corrected = response.copy()
        corrections_made = []
        
        for issue in result.issues:
            if issue['severity'] == 'error' and issue.get('field'):
                field = issue['field']
                
                # Auto-correct missing required fields
                if 'required' in issue['message']:
                    if field == 'success' and field not in corrected:
                        corrected[field] = False
                        corrections_made.append(f"Added missing {field} field")
                    elif field == 'feedback' and field not in corrected:
                        corrected[field] = "Auto-generated feedback due to missing field"
                        corrections_made.append(f"Added missing {field} field")
        
        if corrections_made:
            logger.info(f"Auto-corrections made: {corrections_made}")
            return corrected
        
        return None
    
    def _get_schema_fix_suggestion(self, error: jsonschema.ValidationError) -> str:
        """Generate helpful suggestions for schema validation errors."""
        if error.validator == 'required':
            return f"Add required field: {error.message.split()[-1]}"
        elif error.validator == 'type':
            return f"Expected {error.validator_value}, got {type(error.instance).__name__}"
        elif error.validator == 'minLength':
            return f"Content too short, minimum length: {error.validator_value}"
        else:
            return "Check the field format and try again"


# Global validator instance
response_validator = ResponseValidator()


def validate_ai_response(response: Union[str, Dict], response_type: str = "generic") -> ValidationResult:
    """Convenience function for response validation."""
    return response_validator.validate_response(response, response_type)