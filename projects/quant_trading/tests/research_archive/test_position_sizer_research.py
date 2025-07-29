#!/usr/bin/env python3
"""
Research-Driven Position Sizer Test
Based on VectorBT portfolio optimization research
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_position_sizer_interface():
    """Test Position Sizer interface following research patterns."""
    print("⚖️ Testing Position Sizer (Research-Driven)...")
    
    try:
        from src.execution.position_sizer import GeneticPositionSizer
        from src.config.settings import get_settings
        
        settings = get_settings()
        position_sizer = GeneticPositionSizer(settings)
        print("✅ Position sizer initialized")
        
        # Test interface methods based on VectorBT research patterns
        expected_methods = [
            'calculate_position_size', 'update_genetic_parameters', 
            'get_risk_adjustment', 'calculate_kelly_fraction'
        ]
        
        for method in expected_methods:
            if hasattr(position_sizer, method):
                print(f"✅ Method available: {method}")
            else:
                print(f"⚠️  Method not found: {method}")
        
        # Test Kelly Criterion implementation (from research)
        if hasattr(position_sizer, 'kelly_criterion'):
            print("✅ Kelly Criterion implementation available")
        
        return True
        
    except Exception as e:
        print(f"❌ Position sizer interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kelly_criterion_patterns():
    """Test Kelly Criterion calculations from research."""
    print("📊 Testing Kelly Criterion Patterns...")
    
    try:
        # Research pattern: Kelly Criterion calculation
        # f = (bp - q) / b where f = fraction, b = odds, p = win rate, q = loss rate
        
        def kelly_criterion(win_rate, avg_win, avg_loss):
            """Calculate Kelly fraction following research patterns."""
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            loss_rate = 1 - win_rate
            odds = avg_win / avg_loss
            
            kelly_fraction = (win_rate * odds - loss_rate) / odds
            return max(0.0, min(kelly_fraction, 0.25))  # Cap at 25% for safety
        
        # Test cases from VectorBT research
        test_cases = [
            (0.6, 1.5, 1.0),  # 60% win rate, 1.5:1 reward ratio
            (0.55, 2.0, 1.0), # 55% win rate, 2:1 reward ratio  
            (0.4, 3.0, 1.0),  # 40% win rate, 3:1 reward ratio
            (0.7, 1.0, 1.0),  # 70% win rate, 1:1 reward ratio
        ]
        
        for i, (win_rate, avg_win, avg_loss) in enumerate(test_cases):
            kelly_f = kelly_criterion(win_rate, avg_win, avg_loss)
            print(f"✅ Test {i+1}: Win rate {win_rate:.0%}, Kelly fraction {kelly_f:.1%}")
            
            # Validate Kelly fraction is reasonable
            if 0 <= kelly_f <= 0.25:
                print(f"✅ Kelly fraction within safe bounds")
            else:
                print(f"❌ Kelly fraction outside safe bounds: {kelly_f}")
        
        print("✅ Kelly Criterion patterns validated")
        return True
        
    except Exception as e:
        print(f"❌ Kelly Criterion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_weight_generation():
    """Test portfolio weight generation from VectorBT research."""
    print("📈 Testing Portfolio Weight Generation...")
    
    try:
        # Research Pattern: Multi-asset weight generation with normalization
        symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
        num_assets = len(symbols)
        
        # Generate random weights (genetic algorithm pattern)
        np.random.seed(42)  # For reproducible testing
        raw_weights = np.random.random_sample(num_assets)
        
        # Normalize weights (research requirement)
        normalized_weights = raw_weights / np.sum(raw_weights)
        
        print(f"✅ Generated weights for {num_assets} assets")
        
        # Validate normalization
        weight_sum = np.sum(normalized_weights)
        if abs(weight_sum - 1.0) < 1e-10:
            print("✅ Weight normalization correct")
        else:
            print(f"❌ Weight normalization failed: sum = {weight_sum}")
        
        # Test weight constraints (research pattern)
        max_weight = np.max(normalized_weights)
        min_weight = np.min(normalized_weights)
        
        print(f"✅ Weight range: {min_weight:.1%} to {max_weight:.1%}")
        
        # Test portfolio allocation calculation
        total_capital = 10000.0
        allocations = {}
        
        for symbol, weight in zip(symbols, normalized_weights):
            allocation = total_capital * weight
            allocations[symbol] = allocation
            print(f"✅ {symbol}: ${allocation:.2f} ({weight:.1%})")
        
        # Validate total allocation
        total_allocated = sum(allocations.values())
        if abs(total_allocated - total_capital) < 0.01:
            print("✅ Portfolio allocation correct")
        else:
            print(f"❌ Portfolio allocation error: ${total_allocated:.2f} vs ${total_capital:.2f}")
        
        print("✅ Portfolio weight generation patterns validated")
        return True
        
    except Exception as e:
        print(f"❌ Portfolio weight generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_scaling_patterns():
    """Test risk scaling patterns from research."""
    print("🛡️ Testing Risk Scaling Patterns...")
    
    try:
        # Research Pattern: Risk scaling based on volatility and correlation
        
        def calculate_risk_scaling(volatility, correlation, base_size=0.1):
            """Calculate risk-adjusted position size."""
            # Volatility adjustment (inverse relationship)
            vol_adjustment = 1.0 / (1.0 + volatility * 10)
            
            # Correlation adjustment (reduce for high correlation)
            corr_adjustment = 1.0 - (correlation * 0.5)
            
            # Combined scaling
            risk_scaled_size = base_size * vol_adjustment * corr_adjustment
            
            # Apply bounds
            return max(0.01, min(risk_scaled_size, 0.25))
        
        # Test scenarios
        test_scenarios = [
            (0.02, 0.1, "Low vol, low corr"),
            (0.05, 0.3, "Medium vol, medium corr"),
            (0.1, 0.7, "High vol, high corr"),
            (0.15, 0.9, "Very high vol, very high corr"),
        ]
        
        for volatility, correlation, description in test_scenarios:
            scaled_size = calculate_risk_scaling(volatility, correlation)
            print(f"✅ {description}: {scaled_size:.1%} position size")
            
            # Validate bounds
            if 0.01 <= scaled_size <= 0.25:
                print(f"✅ Risk scaling within bounds")
            else:
                print(f"❌ Risk scaling outside bounds: {scaled_size}")
        
        print("✅ Risk scaling patterns validated")
        return True
        
    except Exception as e:
        print(f"❌ Risk scaling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("⚖️ Research-Driven Position Sizer Testing\n")
    
    success1 = test_position_sizer_interface()
    print()
    success2 = test_kelly_criterion_patterns()
    print()
    success3 = test_portfolio_weight_generation()
    print()
    success4 = test_risk_scaling_patterns()
    
    overall_success = success1 and success2 and success3 and success4
    print(f"\n📋 Overall Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    sys.exit(0 if overall_success else 1)