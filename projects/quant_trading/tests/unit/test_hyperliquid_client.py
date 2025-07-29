#!/usr/bin/env python3
"""
HYPERLIQUID ASSET DISCOVERY VALIDATION SCRIPT

This script validates the asset discovery component of the genetic algorithm trading flow.
It uses the existing HyperliquidClient infrastructure to test real connectivity and 
asset enumeration from Hyperliquid exchange.

Based on research documentation:
- /research/hyperliquid_documentation/3_info_endpoint.md (assetCtxs endpoint)
- /research/hyperliquid_documentation/research_summary.md (API patterns)

Usage: python test_hyperliquid_assets.py

Expected Output:
- If successful: List of all tradeable assets with constraints
- If failed: Specific error details and troubleshooting guidance
"""

import asyncio
import sys
import traceback
import logging
from typing import Dict, List, Any
from datetime import datetime

# Import our existing infrastructure (research-validated)
try:
    from src.data.hyperliquid_client import HyperliquidClient
    from src.config.settings import get_settings
except ImportError as e:
    print(f"❌ IMPORT ERROR: Failed to import required modules: {e}")
    print("Ensure you're running from the project root directory.")
    sys.exit(1)


def format_asset_info(asset_ctx: Dict[str, Any]) -> str:
    """Format asset context information for display.
    
    Args:
        asset_ctx: Asset context dictionary from Hyperliquid API
        
    Returns:
        Formatted string with key asset information
    """
    name = asset_ctx.get('name', 'UNKNOWN')
    max_leverage = asset_ctx.get('maxLeverage', 'N/A')
    size_decimals = asset_ctx.get('szDecimals', 'N/A')
    margin_only = asset_ctx.get('marginOnly', False)
    
    # Format margin status
    margin_status = "Margin Only" if margin_only else "Cross/Isolated"
    
    return (f"  📈 {name:<12} | Max Leverage: {max_leverage:<3} | "
           f"Size Decimals: {size_decimals:<2} | Mode: {margin_status}")


async def test_asset_discovery() -> Dict[str, Any]:
    """Test asset discovery using Hyperliquid client.
    
    Returns:
        Dictionary with test results and metrics
    """
    results = {
        'connection_success': False,
        'asset_count': 0,
        'assets_discovered': [],
        'errors': [],
        'execution_time_ms': 0
    }
    
    start_time = datetime.now()
    
    try:
        # Initialize settings and client (research-validated approach)
        print("🔧 Initializing Hyperliquid client with settings...")
        settings = get_settings()
        
        print(f"   Environment: {settings.environment}")
        print(f"   API URL: {settings.hyperliquid_api_url}")
        print(f"   Rate Limit: {settings.hyperliquid.max_requests_per_second} req/sec")
        
        # Connect to Hyperliquid (async context manager pattern)
        print("\n🌐 Connecting to Hyperliquid exchange...")
        async with HyperliquidClient(settings) as client:
            results['connection_success'] = True
            print("   ✅ Connection established successfully")
            
            # Test asset discovery using documented API endpoint
            print("\n📊 Fetching asset contexts from Hyperliquid...")
            print("   Using endpoint: POST /info with type='meta'")  # Corrected from testing
            
            asset_contexts = await client.get_asset_contexts()
            
            # Process and validate response
            if not isinstance(asset_contexts, list):
                raise ValueError(f"Expected list response, got {type(asset_contexts)}")
            
            results['asset_count'] = len(asset_contexts)
            results['assets_discovered'] = asset_contexts
            
            print(f"   ✅ Successfully retrieved {len(asset_contexts)} asset contexts")
            
            # Display detailed asset information
            print("\n📋 DISCOVERED TRADEABLE ASSETS:")
            print("=" * 80)
            
            if not asset_contexts:
                print("   ⚠️  No assets found - this may indicate connection or permission issues")
                results['errors'].append("No assets returned from API")
            else:
                # Sort assets by name for consistent display
                sorted_assets = sorted(asset_contexts, key=lambda x: x.get('name', ''))
                
                for i, asset in enumerate(sorted_assets, 1):
                    print(f"{i:2d}. {format_asset_info(asset)}")
                
                # Summary statistics
                leveraged_assets = [a for a in asset_contexts if a.get('maxLeverage', 1) > 1]
                margin_only_assets = [a for a in asset_contexts if a.get('marginOnly', False)]
                
                print("\n📊 ASSET DISCOVERY SUMMARY:")
                print(f"   Total Assets: {len(asset_contexts)}")
                print(f"   Leveraged Assets: {len(leveraged_assets)}")
                print(f"   Margin-Only Assets: {len(margin_only_assets)}")
                print(f"   Cross/Isolated Assets: {len(asset_contexts) - len(margin_only_assets)}")
            
    except ConnectionError as e:
        error_msg = f"Connection failed: {e}"
        results['errors'].append(error_msg)
        print(f"\n❌ CONNECTION ERROR: {error_msg}")
        print("💡 Troubleshooting:")
        print("   - Check internet connection")
        print("   - Verify VPN connection (required for Hyperliquid)")
        print("   - Check API URL in settings")
        
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
    print("🔍 HYPERLIQUID ASSET DISCOVERY VALIDATION")
    print("=" * 60)
    print("Testing asset discovery → data ingestion → GA flow (Step 1/3)")
    print("Research-based validation using documented API endpoints")
    print("=" * 60)
    
    # Setup logging for debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the async test
    test_results = asyncio.run(test_asset_discovery())
    
    # Final results summary
    print("\n" + "=" * 60)
    print("📋 ASSET DISCOVERY TEST RESULTS")
    print("=" * 60)
    
    success = (test_results['connection_success'] and 
              test_results['asset_count'] > 0 and 
              len(test_results['errors']) == 0)
    
    if success:
        print("🎉 RESULT: ASSET DISCOVERY SUCCESSFUL")
        print(f"   ✅ Connected to Hyperliquid exchange")
        print(f"   ✅ Discovered {test_results['asset_count']} tradeable assets")
        print(f"   ✅ Execution time: {test_results['execution_time_ms']:.1f}ms")
        print("\n✨ Asset discovery component is working correctly!")
        print("   Ready to proceed with historical data ingestion testing.")
    else:
        print("❌ RESULT: ASSET DISCOVERY FAILED")
        if not test_results['connection_success']:
            print("   ❌ Failed to connect to Hyperliquid exchange")
        if test_results['asset_count'] == 0:
            print("   ❌ No assets discovered")
        for error in test_results['errors']:
            print(f"   ❌ {error}")
        
        print(f"\n📊 Execution time: {test_results['execution_time_ms']:.1f}ms")
        print("\n💡 Next Steps:")
        print("   1. Check network connectivity and VPN status")
        print("   2. Verify Hyperliquid API credentials in settings")
        print("   3. Review logs above for specific error details")
    
    # Return exit code for automation
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)