"""API Rate Limiting and Retry Management for VIBE."""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class APIUsageStats:
    """Track API usage statistics."""
    provider: APIProvider
    requests_count: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    last_request_time: Optional[datetime] = None
    daily_limit_reached: bool = False
    hourly_limit_reached: bool = False
    rate_limit_reset_time: Optional[datetime] = None


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 300.0  # Maximum delay in seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add random jitter to prevent thundering herd


class APIManager:
    """Manages API rate limiting, retry logic, and usage tracking."""
    
    def __init__(self, config_path: str = "api_limits.json"):
        self.config_path = config_path
        self.usage_stats: Dict[APIProvider, APIUsageStats] = {}
        self.rate_limits = self._load_rate_limits()
        self.retry_config = RetryConfig()
        self._initialize_stats()
    
    def _load_rate_limits(self) -> Dict[APIProvider, Dict[str, Any]]:
        """Load API rate limits from configuration."""
        default_limits = {
            APIProvider.CLAUDE: {
                "requests_per_minute": 50,
                "requests_per_hour": 1000,
                "requests_per_day": 10000,
                "tokens_per_minute": 40000,
                "cost_per_day_usd": 100.0
            },
            APIProvider.GEMINI: {
                "requests_per_minute": 60,
                "requests_per_hour": 1500,
                "requests_per_day": 15000,
                "tokens_per_minute": 32000,
                "cost_per_day_usd": 50.0
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    loaded_limits = json.load(f)
                    # Convert string keys back to enum
                    return {APIProvider(k): v for k, v in loaded_limits.items()}
        except Exception as e:
            logger.warning(f"Could not load API limits config: {e}. Using defaults.")
        
        return default_limits
    
    def _initialize_stats(self):
        """Initialize usage statistics for all providers."""
        for provider in APIProvider:
            self.usage_stats[provider] = APIUsageStats(provider=provider)
    
    async def execute_with_retry(
        self, 
        api_call: Callable,
        provider: APIProvider,
        *args,
        **kwargs
    ) -> Any:
        """Execute API call with rate limiting and retry logic."""
        
        # Check rate limits before attempting
        await self._check_rate_limits(provider)
        
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Record request start
                start_time = time.time()
                
                # Execute the API call
                result = await api_call(*args, **kwargs)
                
                # Record successful request
                self._record_successful_request(provider, start_time, result)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    await self._handle_rate_limit_error(provider, e)
                    continue
                
                # Check if it's a retryable error
                if not self._is_retryable_error(e):
                    logger.error(f"Non-retryable error for {provider}: {e}")
                    raise
                
                # Calculate delay for retry
                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}). "
                        f"Retrying in {delay:.2f}s. Error: {e}"
                    )
                    await asyncio.sleep(delay)
                
                # Record failed request
                self._record_failed_request(provider, e)
        
        # All retries exhausted
        logger.error(f"All retry attempts exhausted for {provider}. Last error: {last_exception}")
        raise last_exception
    
    async def _check_rate_limits(self, provider: APIProvider):
        """Check if we can make a request without hitting rate limits."""
        stats = self.usage_stats[provider]
        limits = self.rate_limits.get(provider, {})
        now = datetime.now()
        
        # Check if we need to wait for rate limit reset
        if stats.rate_limit_reset_time and now < stats.rate_limit_reset_time:
            wait_time = (stats.rate_limit_reset_time - now).total_seconds()
            logger.info(f"Rate limit active for {provider}. Waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        # Check daily limits
        if stats.daily_limit_reached:
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            wait_time = (tomorrow - now).total_seconds()
            logger.warning(f"Daily limit reached for {provider}. Waiting until {tomorrow}")
            await asyncio.sleep(wait_time)
            stats.daily_limit_reached = False
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the error is due to rate limiting."""
        error_msg = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "too many requests", 
            "quota exceeded",
            "usage limit",
            "429",
            "rate_limit_exceeded"
        ]
        return any(indicator in error_msg for indicator in rate_limit_indicators)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if the error is retryable."""
        error_msg = str(error).lower()
        retryable_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "server error",
            "503",
            "502",
            "500"
        ]
        return any(indicator in error_msg for indicator in retryable_indicators)
    
    async def _handle_rate_limit_error(self, provider: APIProvider, error: Exception):
        """Handle rate limit errors by setting appropriate wait times."""
        stats = self.usage_stats[provider]
        
        # Extract wait time from error message if available
        error_msg = str(error)
        wait_time = 60  # Default wait time
        
        # Try to extract wait time from common error message formats
        import re
        time_match = re.search(r'retry after (\d+)', error_msg, re.IGNORECASE)
        if time_match:
            wait_time = int(time_match.group(1))
        
        stats.rate_limit_reset_time = datetime.now() + timedelta(seconds=wait_time)
        logger.warning(f"Rate limit hit for {provider}. Waiting {wait_time}s")
        await asyncio.sleep(wait_time)
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay for retry with exponential backoff and jitter."""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay
    
    def _record_successful_request(self, provider: APIProvider, start_time: float, result: Any):
        """Record metrics for successful API request."""
        stats = self.usage_stats[provider]
        stats.requests_count += 1
        stats.last_request_time = datetime.now()
        
        # Try to extract token usage from result
        if hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
            stats.tokens_used += result.usage.total_tokens
        elif isinstance(result, dict) and 'usage' in result:
            stats.tokens_used += result['usage'].get('total_tokens', 0)
        
        execution_time = time.time() - start_time
        logger.info(f"API request to {provider} completed in {execution_time:.2f}s")
    
    def _record_failed_request(self, provider: APIProvider, error: Exception):
        """Record metrics for failed API request."""
        stats = self.usage_stats[provider]
        stats.requests_count += 1  # Still counts towards rate limit
        logger.warning(f"API request to {provider} failed: {error}")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get current usage summary for all providers."""
        summary = {}
        for provider, stats in self.usage_stats.items():
            summary[provider.value] = {
                "requests_count": stats.requests_count,
                "tokens_used": stats.tokens_used,
                "cost_usd": stats.cost_usd,
                "last_request": stats.last_request_time.isoformat() if stats.last_request_time else None,
                "daily_limit_reached": stats.daily_limit_reached,
                "rate_limit_active": stats.rate_limit_reset_time > datetime.now() if stats.rate_limit_reset_time else False
            }
        return summary
    
    def save_usage_stats(self, filepath: str = "api_usage_stats.json"):
        """Save current usage statistics to file."""
        summary = self.get_usage_summary()
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Usage stats saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save usage stats: {e}")


# Global instance for easy access
api_manager = APIManager()


# Decorator for easy integration
def with_api_retry(provider: APIProvider):
    """Decorator to add retry logic to API calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await api_manager.execute_with_retry(func, provider, *args, **kwargs)
        return wrapper
    return decorator


# Usage example:
# @with_api_retry(APIProvider.CLAUDE)
# async def call_claude_api(prompt):
#     # Your Claude API call here
#     pass