#!/usr/bin/env python3
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from src.strategy.genetic_seeds import FundingRateCarrySeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType

def create_test_scenarios():
    scenarios = {}
    
    # Trending market (FAILING)
    dates1 = pd.date_range('2023-01-01', periods=100, freq='D')
    trend_prices = pd.Series(100 * (1 + np.linspace(0, 0.5, 100)), index=dates1)
    scenarios['trending'] = pd.DataFrame({
        'open': trend_prices * 0.99,
        'high': trend_prices * 1.02,
        'low': trend_prices * 0.98,
        'close': trend_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates1)
    
    # Breakout market (FAILING)
    dates2 = pd.date_range('2023-01-01', periods=100, freq='D')
    breakout_prices = pd.Series([100] * 50 + list(100 * (1 + np.linspace(0, 0.3, 50))), index=dates2)
    scenarios['breakout'] = pd.DataFrame({
        'open': breakout_prices * 0.99,
        'high': breakout_prices * 1.02,
        'low': breakout_prices * 0.98,
        'close': breakout_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates2)
    
    # Oscillating market (WORKING)
    dates3 = pd.date_range('2023-01-01', periods=100, freq='D')
    oscillating_prices = pd.Series(100 + 20 * np.sin(np.linspace(0, 4*np.pi, 100)), index=dates3)
    scenarios['oscillating'] = pd.DataFrame({
        'open': oscillating_prices * 0.99,
        'high': oscillating_prices * 1.02,
        'low': oscillating_prices * 0.98,
        'close': oscillating_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates3)
    
    return scenarios

scenarios = create_test_scenarios()
genes = SeedGenes(seed_id='test_funding', seed_type=SeedType.CARRY, parameters={})
funding_seed = FundingRateCarrySeed(genes)

print('=== FUNDING RATE CARRY DETAILED DEBUG ===')
print('Parameters:', funding_seed.genes.parameters)

for scenario_name, data in scenarios.items():
    print(f'\n--- {scenario_name.upper()} SCENARIO ---')
    
    # Test funding rate indicators
    indicators = funding_seed.calculate_technical_indicators(data)
    funding_rate = indicators['funding_rate']
    
    funding_threshold = funding_seed.genes.parameters['funding_threshold']
    reversal_sensitivity = funding_seed.genes.parameters['reversal_sensitivity']
    
    print(f'Funding rate range: {funding_rate.min():.6f} to {funding_rate.max():.6f}')
    print(f'Funding threshold: {funding_threshold:.6f}')
    print(f'Reversal sensitivity: {reversal_sensitivity:.2f}')
    
    # Check individual conditions
    positive_funding = (funding_rate > funding_threshold).sum()
    negative_funding = (funding_rate < -funding_threshold).sum()
    positive_persistence = (indicators['positive_persistence'] >= 2).sum()
    negative_persistence = (indicators['negative_persistence'] >= 2).sum()
    low_reversal = (indicators['reversal_strength'] < reversal_sensitivity).sum()
    
    print(f'Positive funding periods: {positive_funding}')
    print(f'Negative funding periods: {negative_funding}')
    print(f'Positive persistence periods: {positive_persistence}')
    print(f'Negative persistence periods: {negative_persistence}')
    print(f'Low reversal periods: {low_reversal}')
    
    # Check AND conditions
    positive_carry_conditions = (
        (funding_rate > funding_threshold) &
        (indicators['positive_persistence'] >= 2) &
        (indicators['reversal_strength'] < reversal_sensitivity)
    )
    negative_carry_conditions = (
        (funding_rate < -funding_threshold) &
        (indicators['negative_persistence'] >= 2) &
        (indicators['reversal_strength'] < reversal_sensitivity)
    )
    
    print(f'Positive carry AND conditions: {positive_carry_conditions.sum()}')
    print(f'Negative carry AND conditions: {negative_carry_conditions.sum()}')
    
    # Test final signals
    signals = funding_seed.generate_signals(data)
    signal_count = (signals != 0).sum()
    print(f'Final signals generated: {signal_count}')
    print(f'Signal range: {signals.min():.3f} to {signals.max():.3f}')
    
    if signal_count == 0:
        print('❌ FAILURE: No signals in', scenario_name)
        
        # Show momentum filter impact
        rate_momentum = funding_seed.genes.parameters['rate_momentum']
        if rate_momentum > 0.3:
            momentum_boost_positive = (indicators['funding_momentum'] > 0).sum()
            momentum_boost_negative = (indicators['funding_momentum'] < 0).sum()
            print(f'  Momentum filter active (weight: {rate_momentum:.2f})')
            print(f'  Positive momentum periods: {momentum_boost_positive}')
            print(f'  Negative momentum periods: {momentum_boost_negative}')
            
            # Final AND with momentum
            final_positive = (positive_carry_conditions & (indicators['funding_momentum'] > 0)).sum()
            final_negative = (negative_carry_conditions & (indicators['funding_momentum'] < 0)).sum()
            print(f'  Final positive carry (with momentum): {final_positive}')
            print(f'  Final negative carry (with momentum): {final_negative}')
    else:
        print('✅ SUCCESS:', signal_count, 'signals in', scenario_name)