#!/usr/bin/env python3
"""
HYPERLIQUID DATA INGESTION VALIDATION SCRIPT

This script validates the historical data download component of the genetic algorithm 
trading flow. It tests the ability to retrieve OHLCV candlestick data from Hyperliquid
for specific assets and timeframes.

Based on research documentation:
- /research/hyperliquid_documentation/3_info_endpoint.md (candleSnapshot endpoint)
- /research/hyperliquid_documentation/5_websocket_api.md (candle data formats)

Usage: python test_data_ingestion.py

Expected Output:
- If successful: OHLCV data for multiple assets and timeframes
- If failed: Specific error details and troubleshooting guidance
"""

import asyncio
import sys
import traceback
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Import our existing infrastructure (research-validated)
try:
    from src.data.hyperliquid_client import HyperliquidClient
    from src.config.settings import get_settings
except ImportError as e:
    print(f"❌ IMPORT ERROR: Failed to import required modules: {e}")
    print("Ensure you're running from the project root directory.")
    sys.exit(1)


def format_candle_data(candles: List[Dict[str, Any]], symbol: str, interval: str) -> str:
    """Format candlestick data for display.
    
    Args:
        candles: List of candle dictionaries from Hyperliquid API
        symbol: Trading symbol
        interval: Time interval
        
    Returns:
        Formatted string with OHLCV data
    """
    if not candles:
        return f"   ⚠️  No candle data available for {symbol} ({interval})"
    
    # Sort candles by timestamp (research shows data may not be ordered)
    sorted_candles = sorted(candles, key=lambda x: x.get('t', 0))
    
    output = [f"\n📊 {symbol} - {interval} interval ({len(sorted_candles)} candles):"]
    output.append("   Timestamp           | Open      | High      | Low       | Close     | Volume")
    output.append("   " + "-" * 78)
    
    # Show first 3 and last 3 candles if more than 6 total
    if len(sorted_candles) > 6:
        display_candles = sorted_candles[:3] + ["..."] + sorted_candles[-3:]
    else:
        display_candles = sorted_candles
    
    for candle in display_candles:
        if candle == "...":
            output.append("   " + "." * 20 + " ... " + "." * 50)
            continue
            
        # Extract OHLCV data (research-documented format)
        timestamp = candle.get('t', 0)
        open_price = candle.get('o', 0)
        high_price = candle.get('h', 0) 
        low_price = candle.get('l', 0)
        close_price = candle.get('c', 0)
        volume = candle.get('v', 0)
        
        # Format timestamp
        dt = datetime.fromtimestamp(timestamp / 1000) if timestamp else datetime.now()
        time_str = dt.strftime("%Y-%m-%d %H:%M")
        
        output.append(f"   {time_str} | {float(open_price):>9.2f} | {float(high_price):>9.2f} | "
                     f"{float(low_price):>9.2f} | {float(close_price):>9.2f} | {float(volume):>9.2f}")
    
    return "\n".join(output)


async def test_data_ingestion() -> Dict[str, Any]:
    """Test historical data ingestion using Hyperliquid client.
    
    Returns:
        Dictionary with test results and metrics
    """
    results = {
        'connection_success': False,
        'total_requests': 0,
        'successful_requests': 0,
        'data_points_retrieved': 0,
        'test_results': {},
        'errors': [],
        'execution_time_ms': 0
    }
    
    # Test configuration based on research documentation
    test_cases = [
        {'symbol': 'BTC', 'interval': '1h'},    # Most liquid market
        {'symbol': 'ETH', 'interval': '15m'},   # Second most liquid
        {'symbol': 'SOL', 'interval': '1d'},    # Alternative asset
    ]
    
    start_time = datetime.now()
    
    try:
        print("🔧 Initializing Hyperliquid client for data ingestion testing...")
        settings = get_settings()
        
        print(f"   Environment: {settings.environment}")
        print(f"   API URL: {settings.hyperliquid_api_url}")
        
        # Connect to Hyperliquid
        print("\n🌐 Connecting to Hyperliquid exchange...")
        async with HyperliquidClient(settings) as client:
            results['connection_success'] = True
            print("   ✅ Connection established successfully")
            
            # Calculate time range for data request (last 7 days)
            end_time = datetime.now()
            start_time_data = end_time - timedelta(days=7)
            start_timestamp = int(start_time_data.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            print(f"\n📅 Requesting data for time range:")
            print(f"   Start: {start_time_data.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Test data ingestion for each symbol/interval combination
            print("\n📊 Testing historical data retrieval...")
            print("   Using endpoint: POST /info with type='candleSnapshot'")  # Research reference
            
            for test_case in test_cases:
                symbol = test_case['symbol']
                interval = test_case['interval']
                results['total_requests'] += 1
                
                try:
                    print(f"\n   Fetching {symbol} data ({interval} intervals)...")
                    
                    # Use documented API method from research
                    candles = await client.get_candles(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_timestamp,
                        end_time=end_timestamp
                    )
                    
                    # Validate response format (based on research documentation)
                    if not isinstance(candles, list):
                        raise ValueError(f"Expected list response, got {type(candles)}")
                    
                    # Store results
                    results['successful_requests'] += 1
                    results['data_points_retrieved'] += len(candles)
                    results['test_results'][f"{symbol}_{interval}"] = {
                        'candle_count': len(candles),
                        'data': candles[:5] if candles else []  # Store sample data
                    }
                    
                    print(f"      ✅ Retrieved {len(candles)} candles")
                    
                    # Display formatted data
                    formatted_output = format_candle_data(candles, symbol, interval)
                    print(formatted_output)
                    
                except Exception as e:
                    error_msg = f"Failed to fetch {symbol} {interval} data: {e}"
                    results['errors'].append(error_msg)
                    results['test_results'][f"{symbol}_{interval}"] = {
                        'error': str(e),
                        'candle_count': 0
                    }
                    print(f"      ❌ {error_msg}")
            
            # Summary of data ingestion test
            print("\n" + "=" * 60)
            print("📈 DATA INGESTION SUMMARY")
            print("=" * 60)
            
            total_candles = results['data_points_retrieved']
            success_rate = (results['successful_requests'] / results['total_requests']) * 100
            
            print(f"   Requests Made: {results['total_requests']}")
            print(f"   Successful: {results['successful_requests']}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Total Data Points: {total_candles}")
            
            if total_candles > 0:
                avg_candles = total_candles / results['successful_requests'] if results['successful_requests'] > 0 else 0
                print(f"   Average Candles/Request: {avg_candles:.1f}")
                
                # Validate data quality
                print("\n🔍 Data Quality Validation:")
                quality_issues = []
                
                for test_key, test_data in results['test_results'].items():
                    if 'data' in test_data and test_data['data']:
                        sample_candle = test_data['data'][0]
                        required_fields = ['t', 'o', 'h', 'l', 'c', 'v']  # Research-documented fields
                        missing_fields = [field for field in required_fields if field not in sample_candle]
                        
                        if missing_fields:
                            quality_issues.append(f"{test_key}: Missing fields {missing_fields}")
                
                if quality_issues:
                    print("   ⚠️  Data quality issues found:")
                    for issue in quality_issues:
                        print(f"      - {issue}")
                    results['errors'].extend(quality_issues)
                else:
                    print("   ✅ All data contains required OHLCV fields")
    
    except ConnectionError as e:
        error_msg = f"Connection failed: {e}"
        results['errors'].append(error_msg)
        print(f"\n❌ CONNECTION ERROR: {error_msg}")
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        results['errors'].append(error_msg)
        print(f"\n❌ UNEXPECTED ERROR: {error_msg}")
        traceback.print_exc()
    
    finally:
        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        results['execution_time_ms'] = execution_time
    
    return results


def main():
    """Main test execution function."""
    print("📈 HYPERLIQUID DATA INGESTION VALIDATION")
    print("=" * 60)
    print("Testing asset discovery → data ingestion → GA flow (Step 2/3)")
    print("Research-based validation using documented candleSnapshot API")
    print("=" * 60)
    
    # Setup logging for debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the async test
    test_results = asyncio.run(test_data_ingestion())
    
    # Final results summary
    print("\n" + "=" * 60)
    print("📋 DATA INGESTION TEST RESULTS")
    print("=" * 60)
    
    success = (test_results['connection_success'] and 
              test_results['successful_requests'] > 0 and 
              test_results['data_points_retrieved'] > 0 and
              len([e for e in test_results['errors'] if 'Failed to fetch' in e]) == 0)
    
    if success:
        print("🎉 RESULT: DATA INGESTION SUCCESSFUL")
        print(f"   ✅ Connected to Hyperliquid exchange")
        print(f"   ✅ Successfully retrieved data from {test_results['successful_requests']} sources")
        print(f"   ✅ Total data points: {test_results['data_points_retrieved']}")
        print(f"   ✅ Execution time: {test_results['execution_time_ms']:.1f}ms")
        
        # Data quality summary
        if test_results['data_points_retrieved'] > 100:
            print("   ✅ Sufficient data volume for genetic algorithm training")
        else:
            print("   ⚠️  Limited data volume - may need longer time range")
            
        print("\n✨ Data ingestion component is working correctly!")
        print("   Ready to proceed with end-to-end GA flow testing.")
    else:
        print("❌ RESULT: DATA INGESTION FAILED")
        if not test_results['connection_success']:
            print("   ❌ Failed to connect to Hyperliquid exchange")
        if test_results['data_points_retrieved'] == 0:
            print("   ❌ No historical data retrieved")
        for error in test_results['errors']:
            print(f"   ❌ {error}")
        
        print(f"\n📊 Execution time: {test_results['execution_time_ms']:.1f}ms")
        print("\n💡 Next Steps:")
        print("   1. Verify asset symbols are valid on Hyperliquid")
        print("   2. Check time range parameters")
        print("   3. Review API rate limiting and quotas")
    
    # Return exit code for automation
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)