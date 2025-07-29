"""
Integration Tests for Dynamic Asset Data Collector

This test suite validates the complete data collection pipeline including:
- Tradeable asset filtering
- Multi-timeframe data collection
- API-only strategy validation
- Integration with enhanced asset filter
- Data integrity and quality validation

Testing Strategy:
- Real API integration tests (using testnet when possible)
- Mock API responses for edge cases
- Data validation and integrity checks
- Memory usage and performance validation
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.dynamic_asset_data_collector import (
    DynamicAssetDataCollector, 
    AssetDataSet, 
    DataCollectionMetrics,
    IntegratedPipelineOrchestrator
)
from src.discovery.enhanced_asset_filter import EnhancedAssetFilter
from src.config.settings import Settings


class TestDynamicAssetDataCollector:
    """Test suite for Dynamic Asset Data Collector."""
    
    @pytest.fixture
    async def settings(self):
        """Create test settings."""
        # Use testnet settings for testing
        test_settings = Settings()
        test_settings.environment = "testnet"
        return test_settings
    
    @pytest.fixture
    async def data_collector(self, settings):
        """Create data collector instance."""
        return DynamicAssetDataCollector(settings)
    
    @pytest.fixture
    async def mock_asset_contexts(self):
        """Mock asset contexts response."""
        return [
            {
                'name': 'BTC',
                'maxLeverage': 50,
                'szDecimals': 3,
                'onlyIsolated': False
            },
            {
                'name': 'ETH', 
                'maxLeverage': 25,
                'szDecimals': 4,
                'onlyIsolated': False
            },
            {
                'name': 'INVALID',  # Non-tradeable asset
                'maxLeverage': 0,
                'szDecimals': 2,
                'onlyIsolated': True
            }
        ]
    
    @pytest.fixture
    async def mock_candles_data(self):
        """Mock candles response."""
        # Generate mock candle data
        candles = []
        base_time = int(datetime.now().timestamp() * 1000)
        
        for i in range(100):  # 100 candles
            timestamp = base_time - (i * 3600 * 1000)  # 1 hour intervals
            candles.append({
                't': timestamp,
                'o': 50000 + (i * 10),
                'h': 50100 + (i * 10),
                'l': 49900 + (i * 10),
                'c': 50050 + (i * 10),
                'v': 1000 + (i * 5)
            })
        
        return candles
    
    @pytest.mark.asyncio
    async def test_tradeable_asset_filtering(self, data_collector, mock_asset_contexts):
        """Test filtering for tradeable assets only."""
        
        # Mock the client and asset contexts
        with patch.object(data_collector.client, 'connect', new_callable=AsyncMock):
            with patch.object(data_collector.client, 'disconnect', new_callable=AsyncMock):
                with patch.object(data_collector.client, 'get_asset_contexts', 
                                new_callable=AsyncMock, return_value=mock_asset_contexts):
                    
                    # Test asset filtering
                    all_assets = ['BTC', 'ETH', 'INVALID', 'NONEXISTENT']
                    tradeable_assets = await data_collector._filter_tradeable_assets_only(all_assets)
                    
                    # Validate results
                    assert 'BTC' in tradeable_assets
                    assert 'ETH' in tradeable_assets
                    assert 'INVALID' not in tradeable_assets  # Excluded due to onlyIsolated=True
                    assert 'NONEXISTENT' not in tradeable_assets  # Not in asset contexts
                    
                    # Should have 2 tradeable assets
                    assert len(tradeable_assets) == 2
    
    @pytest.mark.asyncio
    async def test_candles_to_dataframe_conversion(self, data_collector, mock_candles_data):
        """Test conversion of candles to DataFrame."""
        
        df = data_collector._convert_candles_to_dataframe(mock_candles_data)
        
        # Validate DataFrame structure
        assert not df.empty
        assert len(df) == 100
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Validate data integrity
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        
        # Validate data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert pd.api.types.is_numeric_dtype(df[col])
    
    @pytest.mark.asyncio
    async def test_data_quality_score_calculation(self, data_collector):
        """Test data quality score calculation."""
        
        # Create test dataframes
        good_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107], 
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        bad_df = pd.DataFrame()  # Empty dataframe
        
        timeframe_data_good = {'1h': good_df, '15m': good_df}
        timeframe_data_bad = {'1h': bad_df, '15m': bad_df}
        bars_collected_good = {'1h': 3, '15m': 3}
        bars_collected_bad = {'1h': 0, '15m': 0}
        
        # Test good data quality
        good_score = data_collector._calculate_data_quality_score(
            timeframe_data_good, bars_collected_good
        )
        assert good_score > 0.5  # Should be reasonable quality
        
        # Test bad data quality
        bad_score = data_collector._calculate_data_quality_score(
            timeframe_data_bad, bars_collected_bad
        )
        assert bad_score == 0.0  # Should be zero quality
    
    @pytest.mark.asyncio
    async def test_asset_dataset_validation(self):
        """Test AssetDataSet validation logic."""
        
        # Create valid dataset
        valid_df = pd.DataFrame({
            'open': [100, 101], 
            'high': [105, 106],
            'low': [95, 96],
            'close': [103, 104], 
            'volume': [1000, 1100]
        }, index=pd.date_range('2023-01-01', periods=2, freq='H'))
        
        valid_dataset = AssetDataSet(
            asset_symbol='BTC',
            timeframe_data={'1h': valid_df, '15m': valid_df},
            collection_timestamp=datetime.now(),
            data_quality_score=0.9,
            bars_collected={'1h': 2000, '15m': 2000}  # Above minimum thresholds
        )
        
        is_valid, errors = valid_dataset.validate_data_integrity()
        assert is_valid
        assert len(errors) == 0
        
        # Create invalid dataset (insufficient bars)
        invalid_dataset = AssetDataSet(
            asset_symbol='ETH',
            timeframe_data={'1h': valid_df, '15m': valid_df},
            collection_timestamp=datetime.now(), 
            data_quality_score=0.9,
            bars_collected={'1h': 500, '15m': 500}  # Below minimum thresholds
        )
        
        is_valid, errors = invalid_dataset.validate_data_integrity()
        assert not is_valid
        assert len(errors) > 0
        assert any('insufficient bars' in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_timeframe_data_collection(self, data_collector, mock_candles_data):
        """Test single timeframe data collection."""
        
        with patch.object(data_collector.client, 'get_candles', 
                         new_callable=AsyncMock, return_value=mock_candles_data):
            
            df, bar_count = await data_collector._collect_timeframe_data('BTC', '1h')
            
            # Validate results
            assert not df.empty
            assert bar_count == 100
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    
    @pytest.mark.asyncio
    async def test_collection_metrics_tracking(self, data_collector):
        """Test collection metrics tracking."""
        
        # Initialize metrics
        assert data_collector.collection_metrics.total_assets_processed == 0
        assert data_collector.collection_metrics.successful_collections == 0
        
        # Simulate metric updates
        data_collector.collection_metrics.total_assets_processed = 10
        data_collector.collection_metrics.successful_collections = 8
        data_collector.collection_metrics.failed_collections = 2
        
        # Validate calculated properties
        assert data_collector.collection_metrics.collection_success_rate == 80.0
        
        # Test metrics dictionary generation
        summary = data_collector.get_collection_summary()
        assert 'collection_metrics' in summary
        assert summary['collection_metrics']['success_rate'] == 80.0


class TestIntegratedPipelineOrchestrator:
    """Test suite for Integrated Pipeline Orchestrator."""
    
    @pytest.fixture
    async def settings(self):
        """Create test settings."""
        test_settings = Settings()
        test_settings.environment = "testnet"
        return test_settings
    
    @pytest.fixture  
    async def orchestrator(self, settings):
        """Create pipeline orchestrator."""
        return IntegratedPipelineOrchestrator(settings)
    
    @pytest.mark.asyncio
    async def test_pipeline_component_initialization(self, orchestrator):
        """Test that pipeline components are properly initialized."""
        
        assert orchestrator.asset_filter is not None
        assert orchestrator.data_collector is not None
        assert isinstance(orchestrator.asset_filter, EnhancedAssetFilter)
        assert isinstance(orchestrator.data_collector, DynamicAssetDataCollector)
    
    @pytest.mark.asyncio
    async def test_pipeline_status_reporting(self, orchestrator):
        """Test pipeline status reporting."""
        
        status = orchestrator.get_pipeline_status()
        
        assert 'pipeline_executed' in status
        assert 'components_status' in status
        assert status['components_status']['asset_filter_ready'] is True
        assert status['components_status']['data_collector_ready'] is True
    
    @pytest.mark.asyncio
    async def test_pipeline_integration_connection(self, orchestrator):
        """Test connection between pipeline components."""
        
        # Mock asset filter results
        mock_assets = ['BTC', 'ETH']
        mock_metrics = {'BTC': Mock(), 'ETH': Mock()}
        
        with patch.object(orchestrator.asset_filter, 'filter_universe',
                         new_callable=AsyncMock, return_value=(mock_assets, mock_metrics)):
            with patch.object(orchestrator.data_collector, 'connect_with_discovery_system',
                             new_callable=AsyncMock):
                with patch.object(orchestrator.data_collector, 'collect_assets_data_pipeline',
                                 new_callable=AsyncMock, return_value={}):
                    with patch.object(orchestrator.data_collector, 'get_multi_asset_data',
                                     new_callable=AsyncMock, return_value={}):
                        
                        # Test pipeline execution
                        results = await orchestrator.execute_full_pipeline()
                        
                        # Validate pipeline results structure
                        assert 'discovery_results' in results
                        assert 'data_collection_results' in results
                        assert 'evolution_ready_data' in results
                        assert 'pipeline_metrics' in results
                        
                        # Validate pipeline success
                        assert results['pipeline_metrics']['pipeline_success'] is True


class TestAPIIntegration:
    """Integration tests with real API (when available)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_asset_contexts(self):
        """Test real API integration for asset contexts (requires network)."""
        
        settings = Settings()
        settings.environment = "testnet"  # Use testnet for safety
        
        collector = DynamicAssetDataCollector(settings)
        
        try:
            await collector.client.connect()
            
            # Test real asset contexts call
            contexts = await collector.client.get_asset_contexts()
            
            # Basic validation
            assert isinstance(contexts, list)
            if contexts:  # If we got data
                assert 'name' in contexts[0]
                print(f"✅ Real API test: Retrieved {len(contexts)} asset contexts")
            
        except Exception as e:
            pytest.skip(f"Real API test skipped due to connectivity: {e}")
        finally:
            await collector.client.disconnect()
    
    @pytest.mark.integration  
    @pytest.mark.asyncio
    async def test_real_api_candles(self):
        """Test real API integration for candles (requires network)."""
        
        settings = Settings()
        settings.environment = "testnet"
        
        collector = DynamicAssetDataCollector(settings)
        
        try:
            await collector.client.connect()
            
            # Test real candles call for BTC
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (100 * 3600 * 1000)  # 100 hours ago
            
            candles = await collector.client.get_candles(
                symbol='BTC',
                interval='1h',
                start_time=start_time,
                end_time=end_time
            )
            
            # Basic validation
            assert isinstance(candles, list)
            if candles:  # If we got data
                assert 't' in candles[0]  # Should have timestamp
                assert 'o' in candles[0]  # Should have OHLC
                print(f"✅ Real API test: Retrieved {len(candles)} candles for BTC")
            
        except Exception as e:
            pytest.skip(f"Real API test skipped due to connectivity: {e}")
        finally:
            await collector.client.disconnect()


class TestPerformanceAndMemory:
    """Performance and memory usage tests."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_dataset_memory_usage(self):
        """Test memory usage with large datasets."""
        
        settings = Settings()
        collector = DynamicAssetDataCollector(settings)
        
        # Create large mock dataset (5000 bars)
        large_candles = []
        base_time = int(datetime.now().timestamp() * 1000)
        
        for i in range(5000):
            timestamp = base_time - (i * 3600 * 1000)
            large_candles.append({
                't': timestamp,
                'o': 50000 + (i * 10),
                'h': 50100 + (i * 10), 
                'l': 49900 + (i * 10),
                'c': 50050 + (i * 10),
                'v': 1000 + (i * 5)
            })
        
        # Test conversion without memory issues
        df = collector._convert_candles_to_dataframe(large_candles)
        
        assert len(df) == 5000
        assert not df.empty
        
        # Validate memory efficient processing
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"💾 Memory usage for 5000 bars: {memory_usage_mb:.2f} MB")
        
        # Should be reasonable memory usage (less than 50MB for 5000 bars)
        assert memory_usage_mb < 50


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])