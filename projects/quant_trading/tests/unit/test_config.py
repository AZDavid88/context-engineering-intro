#!/usr/bin/env python3
"""
Standalone Configuration System Test
Tests settings.py and environment variable handling
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config_system():
    """Test configuration system standalone."""
    print("🔧 Testing Configuration System...")
    
    try:
        from src.config.settings import get_settings, Settings, Environment, HyperliquidConfig
        
        # Test default settings
        settings = get_settings()
        print(f"✅ Settings loaded: Environment = {settings.environment}")
        print(f"✅ Hyperliquid config: {settings.hyperliquid.mainnet_url}")
        print(f"✅ Genetic config: Population = {settings.genetic_algorithm.population_size}")
        print(f"✅ Trading config: Max drawdown = {settings.trading.max_daily_drawdown}")
        
        # Test environment override
        os.environ["ENVIRONMENT"] = "testnet"
        settings_testnet = get_settings()
        print(f"✅ Environment override: {settings_testnet.environment}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_config_system()
    sys.exit(0 if success else 1)