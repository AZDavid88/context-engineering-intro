"""
Unit tests for Fear & Greed Index Client

Tests cover:
- Settings integration and configuration
- API response parsing and validation
- Market regime classification
- Trading signal generation
- Trend analysis calculations
- Genetic algorithm environmental pressure
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

# Import our modules
import sys
sys.path.append('/workspaces/context-engineering-intro/projects/quant_trading')

from src.data.fear_greed_client import (
    FearGreedClient, 
    FearGreedData, 
    FearGreedTrend,
    MarketRegime, 
    TradingSignal
)
from src.config.settings import Settings


class TestFearGreedData:
    """Test FearGreedData model validation and computed fields."""
    
    def test_extreme_fear_classification(self):
        """Test extreme fear regime classification."""
        data = FearGreedData(
            value=15,
            value_classification="Extreme Fear",
            timestamp=datetime.now(timezone.utc),
            regime=None,  # Will be computed
            trading_signal=None,  # Will be computed
            contrarian_strength=None  # Will be computed
        )
        
        assert data.regime == MarketRegime.EXTREME_FEAR
        assert data.trading_signal == TradingSignal.STRONG_BUY
        assert data.contrarian_strength > 0.5  # High contrarian strength
    
    def test_extreme_greed_classification(self):
        """Test extreme greed regime classification."""
        data = FearGreedData(
            value=85,
            value_classification="Extreme Greed",
            timestamp=datetime.now(timezone.utc),
            regime=None,
            trading_signal=None,
            contrarian_strength=None
        )
        
        assert data.regime == MarketRegime.EXTREME_GREED
        assert data.trading_signal == TradingSignal.STRONG_SELL
        assert data.contrarian_strength > 0.3  # Moderate to high contrarian strength
    
    def test_neutral_classification(self):
        """Test neutral regime classification."""
        data = FearGreedData(
            value=50,
            value_classification="Neutral",
            timestamp=datetime.now(timezone.utc),
            regime=None,
            trading_signal=None,
            contrarian_strength=None
        )
        
        assert data.regime == MarketRegime.NEUTRAL
        assert data.trading_signal == TradingSignal.HOLD
        assert data.contrarian_strength == 0.0  # No contrarian signal
    
    def test_value_validation(self):
        """Test value range validation."""
        # Valid value
        data = FearGreedData(
            value=50,
            value_classification="Neutral",
            timestamp=datetime.now(timezone.utc),
            regime=None,
            trading_signal=None,
            contrarian_strength=None
        )
        assert data.value == 50
        
        # Invalid values should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            FearGreedData(
                value=-1,  # Below minimum
                value_classification="Invalid",
                timestamp=datetime.now(timezone.utc),
                regime=None,
                trading_signal=None,
                contrarian_strength=None
            )
        
        with pytest.raises(Exception):  # Pydantic validation error
            FearGreedData(
                value=101,  # Above maximum
                value_classification="Invalid",
                timestamp=datetime.now(timezone.utc),
                regime=None,
                trading_signal=None,
                contrarian_strength=None
            )


class TestFearGreedTrend:
    """Test FearGreedTrend analysis calculations."""
    
    def test_upward_trend(self):
        """Test upward trend detection."""
        trend = FearGreedTrend(
            current_value=60,
            previous_value=45,
            trend_direction=None,  # Will be computed
            trend_strength=None,   # Will be computed
            volatility=0.1
        )
        
        assert trend.trend_direction == "up"
        assert trend.trend_strength > 0.2  # Significant change
    
    def test_downward_trend(self):
        """Test downward trend detection."""
        trend = FearGreedTrend(
            current_value=30,
            previous_value=55,
            trend_direction=None,
            trend_strength=None,
            volatility=0.15
        )
        
        assert trend.trend_direction == "down"
        assert trend.trend_strength > 0.4  # Large change
    
    def test_stable_trend(self):
        """Test stable trend detection."""
        trend = FearGreedTrend(
            current_value=50,
            previous_value=51,
            trend_direction=None,
            trend_strength=None,
            volatility=0.05
        )
        
        assert trend.trend_direction == "stable"
        assert trend.trend_strength < 0.1  # Small change


class TestFearGreedClient:
    """Test FearGreedClient functionality with mocked responses."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = MagicMock(spec=Settings)
        settings.market_regime.fear_greed_api_url = "https://api.alternative.me/fng/"
        settings.market_regime.fear_threshold = 25
        settings.market_regime.greed_threshold = 75
        settings.debug = False
        return settings
    
    @pytest.fixture
    def mock_api_response(self):
        """Mock API response data."""
        return {
            "name": "Fear and Greed Index",
            "data": [
                {
                    "value": "45",
                    "value_classification": "Fear",
                    "timestamp": "1642723200",
                    "time_until_update": "85022"
                }
            ],
            "metadata": {
                "error": None
            }
        }
    
    @pytest.fixture
    def mock_historical_response(self):
        """Mock historical API response data."""
        return {
            "name": "Fear and Greed Index",
            "data": [
                {
                    "value": "45",
                    "value_classification": "Fear",
                    "timestamp": "1642723200"
                },
                {
                    "value": "50",
                    "value_classification": "Neutral",
                    "timestamp": "1642636800"
                },
                {
                    "value": "55",
                    "value_classification": "Greed",
                    "timestamp": "1642550400"
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_settings):
        """Test client initialization with settings."""
        client = FearGreedClient(mock_settings)
        
        assert client.settings == mock_settings
        assert client.api_url == mock_settings.market_regime.fear_greed_api_url
        assert client.fear_threshold == mock_settings.market_regime.fear_threshold
        assert client.greed_threshold == mock_settings.market_regime.greed_threshold
    
    @pytest.mark.asyncio
    async def test_get_current_index(self, mock_settings, mock_api_response):
        """Test getting current Fear & Greed index."""
        client = FearGreedClient(mock_settings)
        
        # Mock the HTTP session and response
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_api_response)
            mock_response.raise_for_status = MagicMock()
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            # Connect and test
            await client.connect()
            result = await client.get_current_index(use_cache=False)
            
            assert isinstance(result, FearGreedData)
            assert result.value == 45
            assert result.regime == MarketRegime.FEAR
            assert result.trading_signal == TradingSignal.WEAK_BUY
            assert result.value_classification == "Fear"
            
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, mock_settings, mock_historical_response):
        """Test getting historical Fear & Greed data."""
        client = FearGreedClient(mock_settings)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_historical_response)
            mock_response.raise_for_status = MagicMock()
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            await client.connect()
            results = await client.get_historical_data(3)
            
            assert len(results) == 3
            assert all(isinstance(item, FearGreedData) for item in results)
            
            # Check different regimes
            regimes = [item.regime for item in results]
            assert MarketRegime.FEAR in regimes
            assert MarketRegime.NEUTRAL in regimes
            assert MarketRegime.GREED in regimes
            
            await client.disconnect()
    
    def test_genetic_algorithm_pressure_extreme_fear(self, mock_settings):
        """Test genetic algorithm pressure calculation for extreme fear."""
        client = FearGreedClient(mock_settings)
        
        fear_data = FearGreedData(
            value=10,
            value_classification="Extreme Fear",
            timestamp=datetime.now(timezone.utc),
            regime=None,
            trading_signal=None,
            contrarian_strength=None
        )
        
        pressure = client.get_genetic_algorithm_pressure(fear_data)
        
        # Extreme fear should favor contrarian strategies
        assert pressure["contrarian_bias"] > 0.7
        assert pressure["momentum_bias"] < 0.3
        assert pressure["volatility_tolerance"] > 0.6  # Higher risk tolerance
        assert pressure["position_sizing"] > 0.5       # Larger positions
        assert pressure["holding_period"] > 0.6        # Longer time horizons
    
    def test_genetic_algorithm_pressure_extreme_greed(self, mock_settings):
        """Test genetic algorithm pressure calculation for extreme greed."""
        client = FearGreedClient(mock_settings)
        
        greed_data = FearGreedData(
            value=90,
            value_classification="Extreme Greed",
            timestamp=datetime.now(timezone.utc),
            regime=None,
            trading_signal=None,
            contrarian_strength=None
        )
        
        pressure = client.get_genetic_algorithm_pressure(greed_data)
        
        # Extreme greed should favor contrarian strategies with caution
        assert pressure["contrarian_bias"] > 0.7
        assert pressure["momentum_bias"] < 0.3
        assert pressure["volatility_tolerance"] < 0.4  # Lower risk tolerance
        assert pressure["position_sizing"] < 0.5       # Smaller positions
        assert pressure["holding_period"] < 0.4        # Shorter time horizons
    
    def test_genetic_algorithm_pressure_neutral(self, mock_settings):
        """Test genetic algorithm pressure calculation for neutral regime."""
        client = FearGreedClient(mock_settings)
        
        neutral_data = FearGreedData(
            value=50,
            value_classification="Neutral",
            timestamp=datetime.now(timezone.utc),
            regime=None,
            trading_signal=None,
            contrarian_strength=None
        )
        
        pressure = client.get_genetic_algorithm_pressure(neutral_data)
        
        # Neutral should have balanced parameters
        for param, value in pressure.items():
            assert 0.4 <= value <= 0.6  # All values should be near neutral (0.5)
    
    @pytest.mark.asyncio
    async def test_analyze_trend(self, mock_settings, mock_historical_response):
        """Test trend analysis functionality."""
        client = FearGreedClient(mock_settings)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_historical_response)
            mock_response.raise_for_status = MagicMock()
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            await client.connect()
            trend = await client.analyze_trend(3)
            
            assert isinstance(trend, FearGreedTrend)
            assert trend.trend_direction in ["up", "down", "stable"]
            assert 0.0 <= trend.trend_strength <= 1.0
            assert 0.0 <= trend.volatility <= 1.0
            
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_settings):
        """Test error handling for API failures."""
        client = FearGreedClient(mock_settings)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Simulate HTTP error
            mock_response = AsyncMock()
            mock_response.raise_for_status.side_effect = aiohttp.ClientError("API Error")
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            await client.connect()
            
            with pytest.raises(aiohttp.ClientError):
                await client.get_current_index(use_cache=False)
            
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, mock_settings, mock_api_response):
        """Test caching behavior for API responses."""
        client = FearGreedClient(mock_settings)
        client._cache_ttl = 1  # 1 second cache for testing
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_api_response)
            mock_response.raise_for_status = MagicMock()
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            await client.connect()
            
            # First call should hit the API
            result1 = await client.get_current_index(use_cache=True)
            assert mock_session.get.call_count == 1
            
            # Second call should use cache
            result2 = await client.get_current_index(use_cache=True)
            assert mock_session.get.call_count == 1  # No additional API call
            assert result1.value == result2.value
            
            # Wait for cache to expire
            await asyncio.sleep(1.1)
            
            # Third call should hit API again
            result3 = await client.get_current_index(use_cache=True)
            assert mock_session.get.call_count == 2  # New API call
            
            await client.disconnect()


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v"])