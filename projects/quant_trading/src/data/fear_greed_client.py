"""
Crypto Fear & Greed Index Client - Market Sentiment Analysis

This module provides integration with the Alternative.me Crypto Fear & Greed Index API
for market regime detection and sentiment-based trading signals.

Key Features:
- Real-time fear/greed index polling (0-100 scale)
- Market regime classification (extreme fear, greed, neutral)
- Historical data retrieval for backtesting
- Contrarian signal generation (extreme fear = buy, extreme greed = sell)
- Integration with genetic algorithm environmental pressure

Based on comprehensive research from:
- Alternative.me API documentation and specifications
- Multi-factor analysis framework (6 components)
- Production-ready market regime detection patterns
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import aiohttp
from pydantic import BaseModel, Field, validator

# Import our centralized configuration system
from src.config.settings import get_settings, Settings


class MarketRegime(str, Enum):
    """Market regime classifications based on Fear & Greed Index."""
    
    EXTREME_FEAR = "extreme_fear"      # 0-25: Strong buy signal
    FEAR = "fear"                      # 26-45: Potential buy signal
    NEUTRAL = "neutral"                # 46-54: No clear signal
    GREED = "greed"                    # 55-75: Caution signal
    EXTREME_GREED = "extreme_greed"    # 76-100: Strong sell signal


class TradingSignal(str, Enum):
    """Trading signals derived from market sentiment."""
    
    STRONG_BUY = "strong_buy"          # Extreme fear
    WEAK_BUY = "weak_buy"              # Fear
    HOLD = "hold"                      # Neutral
    WEAK_SELL = "weak_sell"            # Greed
    STRONG_SELL = "strong_sell"        # Extreme greed


class FearGreedData(BaseModel):
    """Validated Fear & Greed Index data point."""
    
    value: int = Field(..., ge=0, le=100, description="Fear/Greed index value (0-100)")
    value_classification: str = Field(..., description="Text classification (e.g., 'Extreme Fear')")
    timestamp: datetime = Field(..., description="Data timestamp")
    time_until_update: Optional[int] = Field(None, description="Seconds until next update")
    
    # Computed fields
    regime: MarketRegime = Field(..., description="Market regime classification")
    trading_signal: TradingSignal = Field(..., description="Derived trading signal")
    contrarian_strength: float = Field(..., ge=0.0, le=1.0, description="Contrarian signal strength")
    
    @validator('regime', pre=True, always=True)
    def classify_regime(cls, v, values):
        """Classify market regime based on index value."""
        if 'value' not in values:
            return MarketRegime.NEUTRAL
        
        value = values['value']
        if value <= 25:
            return MarketRegime.EXTREME_FEAR
        elif value <= 45:
            return MarketRegime.FEAR
        elif value <= 54:
            return MarketRegime.NEUTRAL
        elif value <= 75:
            return MarketRegime.GREED
        else:
            return MarketRegime.EXTREME_GREED
    
    @validator('trading_signal', pre=True, always=True)
    def derive_trading_signal(cls, v, values):
        """Derive contrarian trading signal from regime."""
        if 'regime' not in values:
            return TradingSignal.HOLD
        
        regime = values['regime']
        signal_map = {
            MarketRegime.EXTREME_FEAR: TradingSignal.STRONG_BUY,
            MarketRegime.FEAR: TradingSignal.WEAK_BUY,
            MarketRegime.NEUTRAL: TradingSignal.HOLD,
            MarketRegime.GREED: TradingSignal.WEAK_SELL,
            MarketRegime.EXTREME_GREED: TradingSignal.STRONG_SELL
        }
        return signal_map.get(regime, TradingSignal.HOLD)
    
    @validator('contrarian_strength', pre=True, always=True)
    def calculate_contrarian_strength(cls, v, values):
        """Calculate contrarian signal strength (0.0 = weak, 1.0 = strong)."""
        if 'value' not in values:
            return 0.0
        
        value = values['value']
        
        # Extreme values have highest contrarian strength
        if value <= 25:  # Extreme fear
            return 1.0 - (value / 25.0)  # 0 = 1.0, 25 = 0.0
        elif value >= 75:  # Extreme greed
            return (value - 75.0) / 25.0  # 75 = 0.0, 100 = 1.0
        else:  # Neutral zone
            return 0.0
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FearGreedTrend(BaseModel):
    """Trend analysis of Fear & Greed Index over time."""
    
    current_value: int = Field(..., description="Current index value")
    previous_value: int = Field(..., description="Previous index value")
    trend_direction: str = Field(..., description="Trend direction (up/down/stable)")
    trend_strength: float = Field(..., description="Trend strength (0.0-1.0)")
    volatility: float = Field(..., description="Index volatility measure")
    
    @validator('trend_direction', pre=True, always=True)
    def calculate_trend_direction(cls, v, values):
        """Calculate trend direction."""
        if 'current_value' not in values or 'previous_value' not in values:
            return "stable"
        
        current = values['current_value']
        previous = values['previous_value']
        
        if current > previous + 2:
            return "up"
        elif current < previous - 2:
            return "down"
        else:
            return "stable"
    
    @validator('trend_strength', pre=True, always=True)
    def calculate_trend_strength(cls, v, values):
        """Calculate trend strength based on change magnitude."""
        if 'current_value' not in values or 'previous_value' not in values:
            return 0.0
        
        current = values['current_value']
        previous = values['previous_value']
        
        # Normalize change to 0-1 scale
        change = abs(current - previous)
        return min(change / 50.0, 1.0)  # 50-point change = max strength


class FearGreedClient:
    """Client for Crypto Fear & Greed Index API with market regime analysis."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize Fear & Greed client.
        
        Args:
            settings: Configuration settings (uses global settings if None)
        """
        self.settings = settings or get_settings()  # KEY CONNECTION TO SETTINGS.PY
        self.api_url = self.settings.market_regime.fear_greed_api_url
        self.fear_threshold = self.settings.market_regime.fear_threshold
        self.greed_threshold = self.settings.market_regime.greed_threshold
        
        # HTTP session configuration
        self.timeout = aiohttp.ClientTimeout(total=30.0)
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_is_shared = False  # Track if session is externally managed
        
        # Data cache
        self._cache: Dict[str, FearGreedData] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        # Logger setup
        self.logger = logging.getLogger(f"{__name__}.FearGreed")
        self.logger.setLevel(logging.DEBUG if self.settings.debug else logging.INFO)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Initialize HTTP session."""
        if self.session is None:
            headers = {
                "User-Agent": "QuantTradingOrganism/1.0",
                "Accept": "application/json"
            }
            
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers
            )
            
            self.logger.info(f"Fear & Greed client connected to {self.api_url}")
    
    def set_shared_session(self, session: aiohttp.ClientSession) -> None:
        """Set a shared session that should not be closed by this client."""
        self.session = session
        self._session_is_shared = True
        self.logger.info("Fear & Greed client using shared session")
    
    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self.session and not self._session_is_shared:
            await self.session.close()
            self.session = None
            self.logger.info("Fear & Greed client disconnected")
        elif self.session and self._session_is_shared:
            # Don't close shared sessions, just remove reference
            self.session = None
            self.logger.info("Fear & Greed client disconnected (shared session preserved)")
    
    async def get_current_index(self, use_cache: bool = True) -> FearGreedData:
        """Get current Fear & Greed Index value.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Current Fear & Greed data with regime classification
            
        Raises:
            aiohttp.ClientError: On HTTP errors
            ValueError: On invalid response format
        """
        cache_key = "current"
        
        # Check cache first
        if use_cache and cache_key in self._cache:
            cached_data = self._cache[cache_key]
            cache_age = (datetime.now(timezone.utc) - cached_data.timestamp).total_seconds()
            if cache_age < self._cache_ttl:
                self.logger.debug("Returning cached Fear & Greed data")
                return cached_data
        
        if not self.session:
            if self._session_is_shared:
                raise RuntimeError("Shared session is None - TradingSystemManager may not be active")
            else:
                await self.connect()
        
        try:
            async with self.session.get(self.api_url) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Parse response
                if not data or 'data' not in data or not data['data']:
                    raise ValueError("Invalid API response format")
                
                # Get most recent data point
                latest = data['data'][0]
                
                # Create validated data object
                fear_greed_data = FearGreedData(
                    value=int(latest['value']),
                    value_classification=latest['value_classification'],
                    timestamp=datetime.fromtimestamp(int(latest['timestamp']), tz=timezone.utc),
                    time_until_update=latest.get('time_until_update'),
                    regime=None,  # Will be computed by validator
                    trading_signal=None,  # Will be computed by validator
                    contrarian_strength=None  # Will be computed by validator
                )
                
                # Cache the result
                self._cache[cache_key] = fear_greed_data
                
                self.logger.info(f"Retrieved Fear & Greed Index: {fear_greed_data.value} ({fear_greed_data.regime.value})")
                return fear_greed_data
                
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error retrieving Fear & Greed Index: {e}")
            raise
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing Fear & Greed response: {e}")
            raise ValueError(f"Invalid response format: {e}")
    
    async def get_historical_data(self, days: int = 30) -> List[FearGreedData]:
        """Get historical Fear & Greed Index data.
        
        Args:
            days: Number of days of historical data to retrieve
            
        Returns:
            List of historical Fear & Greed data points
        """
        if not self.session:
            if self._session_is_shared:
                raise RuntimeError("Shared session is None - TradingSystemManager may not be active")
            else:
                await self.connect()
        
        try:
            params = {"limit": str(days), "format": "json"}
            async with self.session.get(self.api_url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if not data or 'data' not in data:
                    raise ValueError("Invalid historical data response")
                
                historical_data = []
                for item in data['data']:
                    fear_greed_data = FearGreedData(
                        value=int(item['value']),
                        value_classification=item['value_classification'],
                        timestamp=datetime.fromtimestamp(int(item['timestamp']), tz=timezone.utc),
                        time_until_update=item.get('time_until_update'),
                        regime=None,  # Computed by validator
                        trading_signal=None,  # Computed by validator
                        contrarian_strength=None  # Computed by validator
                    )
                    historical_data.append(fear_greed_data)
                
                self.logger.info(f"Retrieved {len(historical_data)} historical Fear & Greed data points")
                return historical_data
                
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error retrieving historical data: {e}")
            raise
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing historical response: {e}")
            raise ValueError(f"Invalid historical data format: {e}")
    
    async def analyze_trend(self, lookback_days: int = 7) -> FearGreedTrend:
        """Analyze Fear & Greed Index trend over specified period.
        
        Args:
            lookback_days: Number of days to analyze for trend
            
        Returns:
            Trend analysis with direction and strength
        """
        historical_data = await self.get_historical_data(lookback_days + 1)
        
        if len(historical_data) < 2:
            raise ValueError("Insufficient data for trend analysis")
        
        # Sort by timestamp (most recent first)
        historical_data.sort(key=lambda x: x.timestamp, reverse=True)
        
        current = historical_data[0]
        previous = historical_data[1]
        
        # Calculate volatility (standard deviation of recent values)
        recent_values = [item.value for item in historical_data[:min(7, len(historical_data))]]
        if len(recent_values) > 1:
            mean_value = sum(recent_values) / len(recent_values)
            variance = sum((x - mean_value) ** 2 for x in recent_values) / len(recent_values)
            volatility = (variance ** 0.5) / 100.0  # Normalize to 0-1
        else:
            volatility = 0.0
        
        trend = FearGreedTrend(
            current_value=current.value,
            previous_value=previous.value,
            trend_direction=None,  # Computed by validator
            trend_strength=None,   # Computed by validator
            volatility=volatility
        )
        
        self.logger.info(f"Trend analysis: {trend.trend_direction} (strength: {trend.trend_strength:.2f})")
        return trend
    
    def get_genetic_algorithm_pressure(self, fear_greed_data: FearGreedData) -> Dict[str, float]:
        """Convert Fear & Greed data to genetic algorithm environmental pressure.
        
        This method translates market sentiment into parameters that can influence
        genetic algorithm evolution, encouraging different strategy types based on
        market conditions.
        
        Args:
            fear_greed_data: Current Fear & Greed Index data
            
        Returns:
            Dictionary of environmental pressure parameters
        """
        regime = fear_greed_data.regime
        value = fear_greed_data.value
        contrarian_strength = fear_greed_data.contrarian_strength
        
        # Base pressure parameters
        pressure = {
            "contrarian_bias": 0.0,      # Favor contrarian strategies
            "momentum_bias": 0.0,        # Favor momentum strategies  
            "volatility_tolerance": 0.5,  # Risk tolerance adjustment
            "position_sizing": 0.5,      # Position size preference
            "holding_period": 0.5        # Time horizon preference
        }
        
        # Adjust based on regime
        if regime == MarketRegime.EXTREME_FEAR:
            pressure.update({
                "contrarian_bias": 0.8,      # Strong contrarian advantage
                "momentum_bias": 0.2,        # Weak momentum preference
                "volatility_tolerance": 0.7,  # Higher risk tolerance
                "position_sizing": 0.6,      # Larger positions in fear
                "holding_period": 0.7        # Longer time horizons
            })
        elif regime == MarketRegime.EXTREME_GREED:
            pressure.update({
                "contrarian_bias": 0.8,      # Strong contrarian advantage
                "momentum_bias": 0.2,        # Weak momentum preference
                "volatility_tolerance": 0.3,  # Lower risk tolerance
                "position_sizing": 0.4,      # Smaller positions in greed
                "holding_period": 0.3        # Shorter time horizons
            })
        elif regime in [MarketRegime.FEAR, MarketRegime.GREED]:
            pressure.update({
                "contrarian_bias": 0.6,      # Moderate contrarian bias
                "momentum_bias": 0.4,        # Moderate momentum bias
                "volatility_tolerance": 0.5,  # Neutral risk tolerance
                "position_sizing": 0.5,      # Neutral position sizing
                "holding_period": 0.5        # Neutral time horizon
            })
        # Neutral regime keeps default values
        
        self.logger.debug(f"Genetic algorithm pressure for {regime.value}: {pressure}")
        return pressure
    
    async def get_regime_signals(self) -> Tuple[MarketRegime, TradingSignal, float]:
        """Get current market regime with trading signals.
        
        Returns:
            Tuple of (regime, signal, strength)
        """
        data = await self.get_current_index()
        return data.regime, data.trading_signal, data.contrarian_strength


async def test_fear_greed_client():
    """Test function to validate Fear & Greed client functionality."""
    
    # Use our settings system - DEMONSTRATES THE CONNECTION
    settings = get_settings()
    
    print("=== Fear & Greed Index Client Test ===")
    print(f"API URL: {settings.market_regime.fear_greed_api_url}")
    print(f"Fear Threshold: {settings.market_regime.fear_threshold}")
    print(f"Greed Threshold: {settings.market_regime.greed_threshold}")
    
    async with FearGreedClient(settings) as client:
        try:
            # Test current index
            print("\n=== Current Fear & Greed Index ===")
            current_data = await client.get_current_index()
            print(f"Value: {current_data.value}")
            print(f"Classification: {current_data.value_classification}")
            print(f"Regime: {current_data.regime.value}")
            print(f"Trading Signal: {current_data.trading_signal.value}")
            print(f"Contrarian Strength: {current_data.contrarian_strength:.2f}")
            
            # Test trend analysis
            print("\n=== Trend Analysis ===")
            trend = await client.analyze_trend(7)
            print(f"Trend Direction: {trend.trend_direction}")
            print(f"Trend Strength: {trend.trend_strength:.2f}")
            print(f"Volatility: {trend.volatility:.2f}")
            
            # Test genetic algorithm pressure
            print("\n=== Genetic Algorithm Environmental Pressure ===")
            pressure = client.get_genetic_algorithm_pressure(current_data)
            for param, value in pressure.items():
                print(f"{param}: {value:.2f}")
            
            # Test historical data (last 5 days)
            print("\n=== Historical Data (Last 5 Days) ===")
            historical = await client.get_historical_data(5)
            for data_point in historical:
                print(f"{data_point.timestamp.strftime('%Y-%m-%d')}: {data_point.value} ({data_point.regime.value})")
            
            print("\n✅ Fear & Greed client test completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    """Test the Fear & Greed client."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_fear_greed_client())