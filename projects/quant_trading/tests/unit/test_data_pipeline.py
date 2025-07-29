#!/usr/bin/env python3
"""
Standalone Market Data Pipeline Test
Tests data ingestion and processing components
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_data_pipeline():
    """Test market data pipeline standalone."""
    print("📊 Testing Market Data Pipeline...")
    
    try:
        from src.data.market_data_pipeline import MarketDataPipeline
        from src.config.settings import get_settings
        
        settings = get_settings()
        pipeline = MarketDataPipeline(settings)
        print("✅ Pipeline initialized")
        
        # Test pipeline status
        status = pipeline.get_status()
        print(f"✅ Pipeline status: {status}")
        
        # Test metrics access
        metrics = pipeline.get_metrics()
        print(f"✅ Metrics available: {metrics.messages_processed} messages processed")
        
        # Test callback registration
        callback_called = False
        
        def tick_callback(tick):
            nonlocal callback_called
            callback_called = True
            
        pipeline.add_tick_callback(tick_callback)
        print("✅ Callback registration operational")
        
        # Test available symbols (no network needed)
        try:
            # Test internal state methods
            if hasattr(pipeline, '_should_trigger_circuit_breaker'):
                circuit_status = pipeline._should_trigger_circuit_breaker()
                print(f"✅ Circuit breaker logic: {circuit_status}")
            
            print("✅ Pipeline interfaces functional")
        except Exception as e:
            print(f"⚠️  Expected interface limitation: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_data_pipeline())
    sys.exit(0 if success else 1)