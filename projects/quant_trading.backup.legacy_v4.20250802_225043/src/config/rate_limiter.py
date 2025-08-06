"""
DEFINITIVE Rate Limiting Solution - Research-Based Implementation

This is the AUTHORITATIVE rate limiting implementation based on:
/research/hyperliquid_documentation/6_rate_limits_and_constraints.md

ALL OTHER RATE LIMITING IMPLEMENTATIONS SHOULD IMPORT FROM HERE.

Key Research Findings:
- IP Limits: 1200 weight per minute (20 per second sustained)
- Address Limits: 1 request per $1 USDC traded + 10k initial buffer
- Batch Weight: 1 + floor(batch_length / 40)
- Info API Weights: l2Book=2, most others=20, userRole=60, explorer=40
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque

logger = logging.getLogger(__name__)


class APIEndpointType(Enum):
    """API endpoint types with their respective weights."""
    INFO_LIGHT = 2      # l2Book, allMids, clearinghouseState, orderStatus, spotClearinghouseState, exchangeStatus  
    INFO_STANDARD = 20  # Most other documented info requests
    INFO_HEAVY = 60     # userRole
    EXPLORER = 40       # All explorer API requests
    EXCHANGE = 1        # Exchange API (base weight, increases with batch size)


@dataclass
class RateLimitState:
    """Thread-safe rate limit state tracking."""
    ip_weight_remaining: int = 1200  # Per minute limit
    ip_weight_reset_time: float = field(default_factory=time.time)
    
    address_requests_remaining: int = 10000  # Initial buffer
    last_trading_volume: float = 0.0  # Cumulative USDC traded
    
    # Request tracking
    requests_in_window: deque = field(default_factory=lambda: deque(maxlen=100))
    consecutive_429s: int = 0
    backoff_until: float = 0.0
    
    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock)


class HyperliquidRateLimiter:
    """
    DEFINITIVE Hyperliquid rate limiting implementation.
    
    Based on research from /research/hyperliquid_documentation/6_rate_limits_and_constraints.md
    This is the single source of truth for all rate limiting logic.
    """
    
    def __init__(self, initial_trading_volume: float = 0.0):
        """
        Initialize rate limiter with optional trading volume.
        
        Args:
            initial_trading_volume: Cumulative USDC traded for address limit calculation
        """
        self.state = RateLimitState()
        self.state.last_trading_volume = initial_trading_volume
        self._update_address_limit()
        
        logger.info(f"🚀 Definitive rate limiter initialized")
        logger.info(f"   IP limit: {self.state.ip_weight_remaining}/1200 per minute")
        logger.info(f"   Address limit: {self.state.address_requests_remaining} requests")
    
    def _update_address_limit(self):
        """Update address-based request limit based on trading volume."""
        # Research formula: 1 request per 1 USDC traded + 10k initial buffer
        base_limit = 10000 + int(self.state.last_trading_volume)
        self.state.address_requests_remaining = min(
            self.state.address_requests_remaining, 
            base_limit
        )
    
    def calculate_batch_weight(self, batch_size: int) -> int:
        """
        Calculate batch weight using research formula.
        
        Research: Exchange API weight = 1 + floor(batch_length / 40)
        """
        return 1 + (batch_size // 40)
    
    def get_endpoint_weight(self, endpoint_type: APIEndpointType, batch_size: int = 1) -> int:
        """Get weight for specific endpoint type."""
        if endpoint_type == APIEndpointType.EXCHANGE:
            return self.calculate_batch_weight(batch_size)
        return endpoint_type.value
    
    def can_make_request(
        self, 
        endpoint_type: APIEndpointType = APIEndpointType.INFO_STANDARD,
        batch_size: int = 1,
        require_address_limit: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if request can be made within rate limits.
        
        Args:
            endpoint_type: Type of API endpoint
            batch_size: Size of batch for weight calculation
            require_address_limit: Whether this request counts against address limits
            
        Returns:
            (can_make_request, reason_if_not)
        """
        with self.state._lock:
            # Check if in backoff period
            if time.time() < self.state.backoff_until:
                remaining = self.state.backoff_until - time.time()
                return False, f"In backoff period for {remaining:.1f}s"
            
            # Reset IP limits if minute has passed
            current_time = time.time()
            if current_time - self.state.ip_weight_reset_time >= 60:
                self.state.ip_weight_remaining = 1200
                self.state.ip_weight_reset_time = current_time
                logger.debug("IP rate limit window reset")
            
            # Calculate required weight
            weight = self.get_endpoint_weight(endpoint_type, batch_size)
            
            # Check IP limits
            if self.state.ip_weight_remaining < weight:
                return False, f"IP weight insufficient: need {weight}, have {self.state.ip_weight_remaining}"
            
            # Check address limits for trading operations
            if require_address_limit and self.state.address_requests_remaining < 1:
                return False, f"Address request limit exceeded: {self.state.address_requests_remaining} remaining"
            
            return True, "OK"
    
    async def wait_for_rate_limit(
        self,
        endpoint_type: APIEndpointType = APIEndpointType.INFO_STANDARD,
        batch_size: int = 1,
        require_address_limit: bool = False,
        max_wait_seconds: float = 60.0
    ) -> bool:
        """
        Wait until request can be made within rate limits.
        
        Returns:
            True if can proceed, False if max wait time exceeded
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            can_proceed, reason = self.can_make_request(
                endpoint_type, batch_size, require_address_limit
            )
            
            if can_proceed:
                return True
            
            # Calculate optimal wait time
            if "backoff" in reason.lower():
                wait_time = self.state.backoff_until - time.time()
            elif "ip weight" in reason.lower():
                # Wait for next minute window
                wait_time = 60 - (time.time() - self.state.ip_weight_reset_time)
            elif "address" in reason.lower():
                # Address limits require trading volume increase - can't wait
                logger.warning(f"Address rate limit hit - requires increased trading volume")
                return False
            else:
                wait_time = 1.0  # Default wait
            
            wait_time = min(wait_time, max_wait_seconds - (time.time() - start_time))
            if wait_time > 0:
                logger.debug(f"Rate limit wait: {wait_time:.1f}s ({reason})")
                await asyncio.sleep(wait_time)
            else:
                break
        
        return False
    
    def consume_request(
        self,
        endpoint_type: APIEndpointType = APIEndpointType.INFO_STANDARD,
        batch_size: int = 1,
        require_address_limit: bool = False,
        response_code: Optional[int] = None
    ):
        """
        Consume rate limit quota after making request.
        
        Args:
            endpoint_type: Type of API endpoint used
            batch_size: Size of batch processed
            require_address_limit: Whether this consumed address limit
            response_code: HTTP response code (for 429 handling)
        """
        with self.state._lock:
            weight = self.get_endpoint_weight(endpoint_type, batch_size)
            
            # Consume IP weight
            self.state.ip_weight_remaining = max(0, self.state.ip_weight_remaining - weight)
            
            # Consume address requests
            if require_address_limit:
                self.state.address_requests_remaining = max(0, self.state.address_requests_remaining - 1)
            
            # Handle 429 responses
            if response_code == 429:
                self.state.consecutive_429s += 1
                # Exponential backoff: 2^n seconds, max 60s
                backoff_seconds = min(2 ** self.state.consecutive_429s, 60)
                self.state.backoff_until = time.time() + backoff_seconds
                logger.warning(f"429 Rate limit hit - backing off for {backoff_seconds}s")
            else:
                self.state.consecutive_429s = 0
            
            # Record request
            self.state.requests_in_window.append({
                'timestamp': time.time(),
                'weight': weight,
                'endpoint': endpoint_type.name,
                'response_code': response_code
            })
    
    def update_trading_volume(self, new_volume: float):
        """Update cumulative trading volume to increase address limits."""
        with self.state._lock:
            if new_volume > self.state.last_trading_volume:
                old_limit = self.state.address_requests_remaining
                self.state.last_trading_volume = new_volume
                self._update_address_limit()
                
                volume_increase = new_volume - self.state.last_trading_volume
                logger.info(f"Trading volume updated: +${volume_increase:.2f} USDC")
                logger.info(f"Address limit: {old_limit} → {self.state.address_requests_remaining}")
    
    def get_status(self) -> Dict:
        """Get current rate limit status."""
        with self.state._lock:
            current_time = time.time()
            ip_reset_in = 60 - (current_time - self.state.ip_weight_reset_time)
            
            return {
                'ip_weight_remaining': self.state.ip_weight_remaining,
                'ip_reset_in_seconds': ip_reset_in if ip_reset_in > 0 else 0,
                'address_requests_remaining': self.state.address_requests_remaining,
                'in_backoff': current_time < self.state.backoff_until,
                'backoff_remaining': max(0, self.state.backoff_until - current_time),
                'consecutive_429s': self.state.consecutive_429s,
                'recent_requests': len(self.state.requests_in_window)
            }
    
    def reset_for_testing(self):
        """Reset rate limiter state for testing purposes."""
        with self.state._lock:
            self.state.ip_weight_remaining = 1200
            self.state.ip_weight_reset_time = time.time()
            self.state.address_requests_remaining = 10000
            self.state.consecutive_429s = 0
            self.state.backoff_until = 0.0
            self.state.requests_in_window.clear()
            logger.info("Rate limiter reset for testing")


# Global singleton instance
_global_rate_limiter: Optional[HyperliquidRateLimiter] = None


def get_rate_limiter() -> HyperliquidRateLimiter:
    """Get global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = HyperliquidRateLimiter()
    return _global_rate_limiter


def reset_rate_limiter():
    """Reset global rate limiter (for testing)."""
    global _global_rate_limiter
    _global_rate_limiter = None