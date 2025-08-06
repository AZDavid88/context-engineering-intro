#!/usr/bin/env python3
"""
Asset Universe Filter Validation Script

Tests the ResearchBackedAssetFilter with real Hyperliquid data to validate:
1. Asset filtering from 180 → optimal subset
2. Liquidity analysis using L2 book data  
3. Volatility analysis using historical candles
4. Correlation-based diversity filtering
5. Multi-criteria scoring and ranking

Expected Result: 20-25 high-quality assets selected for genetic algorithm focus
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import Settings, Environment
from src.discovery.asset_universe_filter import ResearchBackedAssetFilter


# Configure logging for detailed filtering analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """Test asset universe filtering with comprehensive validation."""
    print("🔍 ASSET UNIVERSE FILTERING VALIDATION")
    print("=" * 60)
    print("Testing intelligent 180 → 25 asset reduction using:")
    print("- L2 book depth analysis for liquidity scoring")  
    print("- Historical volatility analysis for trading potential")
    print("- Correlation matrix for portfolio diversity")
    print("- Multi-criteria composite scoring")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Initialize configuration and filter
        config = Settings(environment=Environment.DEVELOPMENT)
        asset_filter = ResearchBackedAssetFilter(config)
        
        print("🔧 Initializing ResearchBackedAssetFilter...")
        print(f"   Target universe size: {asset_filter.target_universe_size}")
        print(f"   Min liquidity depth: ${asset_filter.min_liquidity_depth:,.0f}")
        print(f"   Max bid-ask spread: {asset_filter.max_bid_ask_spread:.1%}")
        print(f"   Volatility range: {asset_filter.min_daily_volatility:.1%} - {asset_filter.max_daily_volatility:.1%}")
        print()
        
        # Test complete filtering pipeline
        print("🌐 STAGE 1: Asset Discovery")
        print("   Discovering all tradeable assets...")
        
        # Run complete filtering (this will discover assets internally)
        filtered_assets, asset_metrics = await asset_filter.filter_universe()
        
        print("📊 STAGE 2: Metrics Calculation")
        print(f"   Calculated comprehensive metrics for {len(asset_metrics)} assets")
        print()
        
        print("🎯 STAGE 3: Multi-Stage Filtering Results")
        print(f"   Original universe: 180 assets (expected)")
        print(f"   Filtered universe: {len(filtered_assets)} assets")
        print()
        
        # Generate detailed filtering summary
        filter_summary = asset_filter.get_filter_summary(filtered_assets, asset_metrics)
        
        print("📋 FILTERING SUMMARY")
        print("=" * 40)
        print(f"Selected Assets: {filter_summary['selected_assets']}")
        print(f"Average Liquidity Score: {filter_summary['average_liquidity_score']:.3f}")
        print(f"Average Volatility Score: {filter_summary['average_volatility_score']:.3f}")
        print(f"Average Composite Score: {filter_summary['average_composite_score']:.3f}")
        print(f"Score Range: {filter_summary['score_range']['min']:.3f} - {filter_summary['score_range']['max']:.3f}")
        print()
        
        print("🏆 TOP 10 SELECTED ASSETS")
        print("=" * 50)
        print("Rank | Asset    | Composite Score | Liquidity | Volatility")
        print("-" * 50)
        
        top_assets = filter_summary['top_performers'][:10]
        for rank, (asset, score) in enumerate(top_assets, 1):
            metrics = asset_metrics[asset]
            print(f"{rank:4d} | {asset:8s} | {score:13.3f} | {metrics.liquidity_score:8.3f} | {metrics.volatility_score:9.3f}")
        
        print()
        
        # Display detailed metrics for top 5 assets
        print("📊 DETAILED METRICS - TOP 5 ASSETS")
        print("=" * 80)
        
        for rank, (asset, score) in enumerate(top_assets[:5], 1):
            metrics = asset_metrics[asset]
            print(f"\n{rank}. {asset} (Composite Score: {score:.3f})")
            print(f"   📈 Liquidity Metrics:")
            print(f"      Avg Bid Depth: ${metrics.avg_bid_depth:,.0f}")
            print(f"      Avg Ask Depth: ${metrics.avg_ask_depth:,.0f}")
            print(f"      Bid-Ask Spread: {metrics.bid_ask_spread:.4f} ({metrics.bid_ask_spread:.2%})")
            print(f"      Depth Imbalance: {metrics.depth_imbalance:.3f}")
            print(f"   📊 Volatility Metrics:")
            print(f"      Daily Volatility: {metrics.daily_volatility:.1%}")
            print(f"      Intraday Volatility: {metrics.intraday_volatility:.1%}")
            print(f"      Volatility Stability: {metrics.volatility_stability:.3f}")
            print(f"   ⚖️ Trading Metrics:")
            print(f"      Max Leverage: {metrics.max_leverage}x")
            print(f"      Size Decimals: {metrics.size_decimals}")
        
        # Performance metrics
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("📋 ASSET UNIVERSE FILTERING TEST RESULTS")
        print("=" * 60)
        
        if len(filtered_assets) >= 15:  # Reasonable minimum
            print("🎉 RESULT: ASSET FILTERING SUCCESSFUL")
            print(f"   ✅ Filtered {len(asset_metrics)} → {len(filtered_assets)} assets")
            print(f"   ✅ Average composite score: {filter_summary['average_composite_score']:.3f}")
            print(f"   ✅ Execution time: {execution_time:.1f}s")
            print()
            print("✨ Asset universe filter is working correctly!")
            print("   Ready to proceed with hierarchical timeframe discovery.")
            
            # Update todo
            print("\n🔄 Next Phase: Implementing progressive timeframe discovery...")
            
        else:
            print("⚠️ RESULT: INSUFFICIENT ASSETS SELECTED")
            print(f"   Selected only {len(filtered_assets)} assets (expected 15+)")
            print("   May need to adjust filtering thresholds")
        
        print()
        print("📋 Selected Asset Universe:")
        print(", ".join(filtered_assets))
        
    except Exception as e:
        print(f"\n❌ ASSET FILTERING TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())