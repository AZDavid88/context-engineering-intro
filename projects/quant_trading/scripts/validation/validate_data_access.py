#!/usr/bin/env python3
"""
Historical Data Access Validation for Monte Carlo + Walk Forward Systems

This script validates that we can access sufficient historical data from Hyperliquid
for implementing the Monte Carlo and Walk Forward validation systems.

Tests:
1. Candle data availability across multiple timeframes
2. Data quality and consistency validation  
3. Historical depth assessment for validation requirements
4. Rate limiting compliance during bulk downloads
5. Data format compatibility with existing genetic seeds

Based on research from:
- /research/hyperliquid_documentation/3_info_endpoint.md
- /research/hyperliquid_documentation/8_historical_data_access.md
- Existing /src/data/hyperliquid_client.py infrastructure
"""

import asyncio
import sys
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.hyperliquid_client import HyperliquidClient
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidationResults:
    """Container for data validation results."""
    
    def __init__(self):
        self.timeframe_tests = {}
        self.quality_tests = {}
        self.volume_tests = {}
        self.success_rate = 0.0
        self.total_data_points = 0
        self.validation_passed = False


class HistoricalDataValidator:
    """Validates historical data access for Monte Carlo + Walk Forward systems."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = HyperliquidClient(self.settings)
        self.results = DataValidationResults()
        
        # Validation requirements from Monte Carlo + Walk Forward specs
        self.validation_requirements = {
            "scalping": {
                "timeframe": "1m",
                "min_days": 90,      # 3 months for Monte Carlo scenarios
                "min_samples": 129600,  # 90 days * 1440 minutes
                "use_case": "1-minute scalping strategies"
            },
            "day_trading": {
                "timeframe": "5m", 
                "min_days": 180,     # 6 months for walk forward
                "min_samples": 51840,   # 180 days * 288 5-min periods
                "use_case": "5-minute day trading strategies"
            },
            "swing_trading": {
                "timeframe": "15m",
                "min_days": 365,     # 1 year for comprehensive validation
                "min_samples": 35040,   # 365 days * 96 15-min periods  
                "use_case": "15-minute swing trading strategies"
            },
            "position_trading": {
                "timeframe": "1h",
                "min_days": 730,     # 2 years for regime analysis
                "min_samples": 17520,   # 730 days * 24 hours
                "use_case": "1-hour position trading strategies"
            }
        }
        
        # Test assets (from settings.py default target_assets)
        self.test_assets = ["BTC", "ETH", "SOL", "AVAX", "MATIC"]
    
    async def run_comprehensive_validation(self) -> DataValidationResults:
        """Run complete data access validation suite."""
        
        logger.info("🔍 STARTING COMPREHENSIVE DATA ACCESS VALIDATION")
        logger.info("=" * 70)
        
        try:
            # Connect to Hyperliquid
            await self.client.connect()
            logger.info("✅ Successfully connected to Hyperliquid API")
            
            # Test 1: Asset availability
            await self._validate_asset_availability()
            
            # Test 2: Timeframe data access
            await self._validate_timeframe_access()
            
            # Test 3: Historical depth validation
            await self._validate_historical_depth()
            
            # Test 4: Data quality assessment
            await self._validate_data_quality()
            
            # Test 5: Rate limiting compliance
            await self._validate_rate_limiting()
            
            # Calculate overall results
            self._calculate_final_results()
            
            # Display results
            self._display_validation_results()
            
        except Exception as e:
            logger.error(f"❌ Validation failed with error: {e}")
            self.results.validation_passed = False
            
        finally:
            await self.client.disconnect()
            
        return self.results
    
    async def _validate_asset_availability(self):
        """Test that target assets are available on Hyperliquid."""
        
        logger.info("\n📊 Testing Asset Availability...")
        
        try:
            # Get all available assets
            all_mids = await self.client.get_all_mids()
            available_assets = list(all_mids.keys())
            
            logger.info(f"Found {len(available_assets)} available assets")
            
            # Check our target assets
            missing_assets = []
            available_targets = []
            
            for asset in self.test_assets:
                if asset in available_assets:
                    available_targets.append(asset)
                    logger.info(f"  ✅ {asset}: Available (mid price: {all_mids[asset]})")
                else:
                    missing_assets.append(asset)
                    logger.warning(f"  ❌ {asset}: Not available")
            
            # Update test assets to only available ones
            self.test_assets = available_targets
            
            if len(self.test_assets) >= 3:
                logger.info(f"✅ Asset availability test PASSED ({len(self.test_assets)}/5 assets available)")
                self.results.volume_tests['asset_availability'] = True
            else:
                logger.error(f"❌ Asset availability test FAILED (only {len(self.test_assets)}/5 assets available)")
                self.results.volume_tests['asset_availability'] = False
                
        except Exception as e:
            logger.error(f"❌ Asset availability test failed: {e}")
            self.results.volume_tests['asset_availability'] = False
    
    async def _validate_timeframe_access(self):
        """Test access to different timeframes for each validation requirement."""
        
        logger.info("\n⏰ Testing Timeframe Data Access...")
        
        for strategy_type, config in self.validation_requirements.items():
            timeframe = config['timeframe']
            
            logger.info(f"\n--- Testing {timeframe} data for {config['use_case']} ---")
            
            success_count = 0
            total_tests = len(self.test_assets)
            
            for asset in self.test_assets:
                try:
                    # Test recent data access (last 24 hours)
                    end_time = int(datetime.now().timestamp() * 1000)
                    start_time = end_time - (24 * 60 * 60 * 1000)  # 24 hours ago
                    
                    candles = await self.client.get_candles(
                        symbol=asset,
                        interval=timeframe,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if candles and len(candles) > 0:
                        success_count += 1
                        sample_candle = candles[0]
                        
                        logger.info(f"  ✅ {asset} {timeframe}: {len(candles)} candles retrieved")
                        logger.debug(f"    Sample candle: {sample_candle}")
                    else:
                        logger.warning(f"  ❌ {asset} {timeframe}: No data retrieved")
                        
                except Exception as e:
                    logger.error(f"  ❌ {asset} {timeframe}: Error - {e}")
            
            # Record results
            success_rate = success_count / total_tests if total_tests > 0 else 0
            self.results.timeframe_tests[strategy_type] = {
                'timeframe': timeframe,
                'success_count': success_count,
                'total_tests': total_tests,
                'success_rate': success_rate,
                'passed': success_rate >= 0.8  # 80% success rate required
            }
            
            if success_rate >= 0.8:
                logger.info(f"✅ {timeframe} timeframe test PASSED ({success_count}/{total_tests} assets)")
            else:
                logger.error(f"❌ {timeframe} timeframe test FAILED ({success_count}/{total_tests} assets)")
    
    async def _validate_historical_depth(self):
        """Test historical data depth for validation requirements."""
        
        logger.info("\n📈 Testing Historical Data Depth...")
        
        # Test with most liquid asset (BTC) to get best data coverage
        test_asset = "BTC" if "BTC" in self.test_assets else self.test_assets[0]
        
        for strategy_type, config in self.validation_requirements.items():
            timeframe = config['timeframe']
            min_days = config['min_days']
            min_samples = config['min_samples']
            
            logger.info(f"\n--- Testing {min_days}-day history for {timeframe} ({config['use_case']}) ---")
            
            try:
                # Calculate time range
                end_time = int(datetime.now().timestamp() * 1000)
                start_time = end_time - (min_days * 24 * 60 * 60 * 1000)
                
                # Request historical data
                logger.info(f"Requesting {min_days} days of {timeframe} data for {test_asset}...")
                
                candles = await self.client.get_candles(
                    symbol=test_asset,
                    interval=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if candles:
                    actual_samples = len(candles)
                    coverage_ratio = actual_samples / min_samples
                    
                    # Calculate actual time span
                    if len(candles) >= 2:
                        first_time = candles[0].get('t', 0) / 1000  # Convert to seconds
                        last_time = candles[-1].get('t', 0) / 1000
                        actual_days = (last_time - first_time) / (24 * 60 * 60)
                    else:
                        actual_days = 0
                    
                    logger.info(f"  📊 Retrieved {actual_samples} samples over {actual_days:.1f} days")
                    logger.info(f"  📊 Required: {min_samples} samples over {min_days} days")
                    logger.info(f"  📊 Coverage: {coverage_ratio:.2%}")
                    
                    # Record results
                    passed = coverage_ratio >= 0.8  # 80% coverage required
                    self.results.quality_tests[f'{strategy_type}_depth'] = {
                        'required_samples': min_samples,
                        'actual_samples': actual_samples,
                        'required_days': min_days,
                        'actual_days': actual_days,
                        'coverage_ratio': coverage_ratio,
                        'passed': passed
                    }
                    
                    if passed:
                        logger.info(f"  ✅ Historical depth test PASSED for {strategy_type}")
                    else:
                        logger.warning(f"  ⚠️  Historical depth test PARTIAL for {strategy_type} (only {coverage_ratio:.1%} coverage)")
                        
                else:
                    logger.error(f"  ❌ No historical data available for {test_asset} {timeframe}")
                    self.results.quality_tests[f'{strategy_type}_depth'] = {
                        'passed': False,
                        'error': 'No data available'
                    }
                    
            except Exception as e:
                logger.error(f"  ❌ Historical depth test failed for {strategy_type}: {e}")
                self.results.quality_tests[f'{strategy_type}_depth'] = {
                    'passed': False,
                    'error': str(e)
                }
    
    async def _validate_data_quality(self):
        """Validate data quality and format compatibility."""
        
        logger.info("\n🔍 Testing Data Quality and Format...")
        
        test_asset = "BTC" if "BTC" in self.test_assets else self.test_assets[0]
        
        try:
            # Get sample data
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (7 * 24 * 60 * 60 * 1000)  # 1 week
            
            candles = await self.client.get_candles(
                symbol=test_asset,
                interval="1h",  # Use 1-hour for quality testing
                start_time=start_time,
                end_time=end_time
            )
            
            if not candles:
                logger.error("❌ No data available for quality testing")
                self.results.quality_tests['data_quality'] = {'passed': False}
                return
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(candles)
            
            logger.info(f"📊 Analyzing {len(df)} candles for data quality...")
            
            # Test 1: Required fields
            required_fields = ['t', 'o', 'h', 'l', 'c', 'v']  # Time, OHLCV
            missing_fields = [field for field in required_fields if field not in df.columns]
            
            if missing_fields:
                logger.error(f"❌ Missing required fields: {missing_fields}")
                self.results.quality_tests['data_quality'] = {'passed': False, 'error': f'Missing fields: {missing_fields}'}
                return
            else:
                logger.info("✅ All required OHLCV fields present")
            
            # Test 2: Data continuity (no large gaps)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.sort_values('timestamp')
            
            time_diffs = df['timestamp'].diff().dt.total_seconds().fillna(0)
            expected_interval = 3600  # 1 hour in seconds
            large_gaps = (time_diffs > expected_interval * 2).sum()  # Gaps > 2x expected
            
            gap_ratio = large_gaps / len(df) if len(df) > 0 else 1.0
            
            logger.info(f"📊 Found {large_gaps} large gaps out of {len(df)} intervals ({gap_ratio:.2%})")
            
            # Test 3: Price sanity checks
            df[['o', 'h', 'l', 'c']] = df[['o', 'h', 'l', 'c']].astype(float)
            
            # Check OHLC relationships
            invalid_ohlc = (
                (df['h'] < df['o']) | (df['h'] < df['c']) |  # High should be >= Open, Close
                (df['l'] > df['o']) | (df['l'] > df['c']) |  # Low should be <= Open, Close
                (df['h'] < df['l'])                          # High should be >= Low
            ).sum()
            
            ohlc_error_rate = invalid_ohlc / len(df) if len(df) > 0 else 1.0
            
            logger.info(f"📊 Found {invalid_ohlc} invalid OHLC relationships ({ohlc_error_rate:.2%})")
            
            # Test 4: Volume sanity
            df['v'] = df['v'].astype(float)
            zero_volume = (df['v'] <= 0).sum()
            volume_error_rate = zero_volume / len(df) if len(df) > 0 else 1.0
            
            logger.info(f"📊 Found {zero_volume} zero/negative volume periods ({volume_error_rate:.2%})")
            
            # Overall quality assessment
            quality_passed = (
                gap_ratio < 0.05 and         # Less than 5% gaps
                ohlc_error_rate < 0.01 and   # Less than 1% OHLC errors
                volume_error_rate < 0.1      # Less than 10% volume errors
            )
            
            self.results.quality_tests['data_quality'] = {
                'passed': quality_passed,
                'gap_ratio': gap_ratio,
                'ohlc_error_rate': ohlc_error_rate,
                'volume_error_rate': volume_error_rate,
                'total_samples': len(df)
            }
            
            if quality_passed:
                logger.info("✅ Data quality test PASSED")
            else:
                logger.warning("⚠️  Data quality test PARTIAL (some issues found)")
                
        except Exception as e:
            logger.error(f"❌ Data quality test failed: {e}")
            self.results.quality_tests['data_quality'] = {'passed': False, 'error': str(e)}
    
    async def _validate_rate_limiting(self):
        """Test rate limiting compliance during bulk requests."""
        
        logger.info("\n⚡ Testing Rate Limiting Compliance...")
        
        start_time = datetime.now()
        request_count = 0
        errors = 0
        
        try:
            # Make rapid requests to test rate limiting
            test_requests = 20  # Test with 20 requests
            
            for i in range(test_requests):
                try:
                    await self.client.get_all_mids()
                    request_count += 1
                    
                    if i % 5 == 0:
                        logger.info(f"  📊 Completed {i + 1}/{test_requests} requests")
                        
                except Exception as e:
                    errors += 1
                    logger.warning(f"  ⚠️  Request {i + 1} failed: {e}")
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            actual_rate = request_count / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"📊 Rate limiting test results:")
            logger.info(f"  - Successful requests: {request_count}/{test_requests}")
            logger.info(f"  - Failed requests: {errors}")
            logger.info(f"  - Elapsed time: {elapsed_time:.2f} seconds")
            logger.info(f"  - Actual rate: {actual_rate:.2f} req/sec")
            logger.info(f"  - Configured limit: {self.settings.hyperliquid.max_requests_per_second} req/sec")
            
            # Rate limiting is working if we stay under limit and have minimal errors
            rate_limit_passed = (
                actual_rate <= self.settings.hyperliquid.max_requests_per_second * 1.1 and  # Allow 10% margin
                errors < test_requests * 0.2  # Less than 20% error rate
            )
            
            self.results.quality_tests['rate_limiting'] = {
                'passed': rate_limit_passed,
                'actual_rate': actual_rate,
                'configured_limit': self.settings.hyperliquid.max_requests_per_second,
                'error_rate': errors / test_requests if test_requests > 0 else 1.0
            }
            
            if rate_limit_passed:
                logger.info("✅ Rate limiting test PASSED")
            else:
                logger.warning("⚠️  Rate limiting test PARTIAL")
                
        except Exception as e:
            logger.error(f"❌ Rate limiting test failed: {e}")
            self.results.quality_tests['rate_limiting'] = {'passed': False, 'error': str(e)}
    
    def _calculate_final_results(self):
        """Calculate overall validation results."""
        
        # Count passed tests
        passed_timeframe_tests = sum(1 for test in self.results.timeframe_tests.values() if test.get('passed', False))
        total_timeframe_tests = len(self.results.timeframe_tests)
        
        passed_quality_tests = sum(1 for test in self.results.quality_tests.values() if test.get('passed', False))
        total_quality_tests = len(self.results.quality_tests)
        
        passed_volume_tests = sum(1 for test in self.results.volume_tests.values() if test)
        total_volume_tests = len(self.results.volume_tests)
        
        total_passed = passed_timeframe_tests + passed_quality_tests + passed_volume_tests
        total_tests = total_timeframe_tests + total_quality_tests + total_volume_tests
        
        self.results.success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Overall validation passes if 80% of tests pass
        self.results.validation_passed = self.results.success_rate >= 0.8
        
        # Count total data points available
        for test in self.results.quality_tests.values():
            if 'actual_samples' in test:
                self.results.total_data_points += test['actual_samples']
    
    def _display_validation_results(self):
        """Display comprehensive validation results."""
        
        logger.info("\n" + "=" * 70)
        logger.info("📋 COMPREHENSIVE DATA ACCESS VALIDATION RESULTS")
        logger.info("=" * 70)
        
        # Overall status
        status_emoji = "✅" if self.results.validation_passed else "❌"
        logger.info(f"\n{status_emoji} OVERALL STATUS: {'PASSED' if self.results.validation_passed else 'FAILED'}")
        logger.info(f"📊 Success Rate: {self.results.success_rate:.1%}")
        logger.info(f"📊 Total Data Points Validated: {self.results.total_data_points:,}")
        
        # Timeframe test results
        if self.results.timeframe_tests:
            logger.info(f"\n⏰ TIMEFRAME ACCESS TESTS:")
            for strategy_type, test in self.results.timeframe_tests.items():
                status = "✅" if test['passed'] else "❌"
                logger.info(f"  {status} {strategy_type.upper():<15} ({test['timeframe']}): {test['success_count']}/{test['total_tests']} assets ({test['success_rate']:.1%})")
        
        # Historical depth results
        depth_tests = {k: v for k, v in self.results.quality_tests.items() if k.endswith('_depth')}
        if depth_tests:
            logger.info(f"\n📈 HISTORICAL DEPTH TESTS:")
            for test_name, test in depth_tests.items():
                strategy_type = test_name.replace('_depth', '')
                status = "✅" if test.get('passed', False) else "❌"
                if 'coverage_ratio' in test:
                    logger.info(f"  {status} {strategy_type.upper():<15}: {test['actual_samples']:,} samples, {test['actual_days']:.0f} days ({test['coverage_ratio']:.1%} coverage)")
                else:
                    logger.info(f"  {status} {strategy_type.upper():<15}: {test.get('error', 'Failed')}")
        
        # Quality test results
        quality_tests = {k: v for k, v in self.results.quality_tests.items() if not k.endswith('_depth')}
        if quality_tests:
            logger.info(f"\n🔍 DATA QUALITY TESTS:")
            for test_name, test in quality_tests.items():
                status = "✅" if test.get('passed', False) else "❌"
                logger.info(f"  {status} {test_name.replace('_', ' ').title()}")
        
        # Volume/Asset tests
        if self.results.volume_tests:
            logger.info(f"\n📊 ASSET AVAILABILITY TESTS:")
            for test_name, passed in self.results.volume_tests.items():
                status = "✅" if passed else "❌"
                logger.info(f"  {status} {test_name.replace('_', ' ').title()}")
        
        # Final assessment
        logger.info(f"\n" + "=" * 70)
        if self.results.validation_passed:
            logger.info("🎉 DATA ACCESS VALIDATION: READY FOR MONTE CARLO + WALK FORWARD IMPLEMENTATION")
            logger.info("All critical data requirements satisfied for validation systems.")
        else:
            logger.info("⚠️  DATA ACCESS VALIDATION: PARTIAL SUCCESS - REVIEW REQUIRED")
            logger.info("Some data limitations found. Consider adjusting validation parameters.")
        logger.info("=" * 70)


async def main():
    """Run the comprehensive data validation."""
    
    print("🔍 HYPERLIQUID HISTORICAL DATA ACCESS VALIDATION")
    print("Testing data availability for Monte Carlo + Walk Forward validation systems")
    print("=" * 70)
    
    validator = HistoricalDataValidator()
    results = await validator.run_comprehensive_validation()
    
    return results.validation_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)