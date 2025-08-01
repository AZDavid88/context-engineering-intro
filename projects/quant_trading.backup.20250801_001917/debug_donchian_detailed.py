#!/usr/bin/env python3
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from src.strategy.genetic_seeds import DonchianBreakoutSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType

# Test scenarios from comprehensive validation
def create_test_scenarios():
    scenarios = {}
    
    # Trending market (FAILING scenario)
    dates1 = pd.date_range('2023-01-01', periods=100, freq='D')
    trend_prices = pd.Series(100 * (1 + np.linspace(0, 0.5, 100)), index=dates1)
    scenarios['trending'] = pd.DataFrame({
        'open': trend_prices * 0.99,
        'high': trend_prices * 1.02,
        'low': trend_prices * 0.98,
        'close': trend_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates1)
    
    # Breakout pattern (WORKING scenario)
    dates4 = pd.date_range('2023-01-01', periods=100, freq='D')
    breakout_prices = []
    for i in range(100):
        if i < 50:
            breakout_prices.append(100 + np.random.normal(0, 1))  # Sideways
        else:
            breakout_prices.append(100 + (i-50) * 0.5 + np.random.normal(0, 1))  # Breakout
    breakout_prices = pd.Series(breakout_prices, index=dates4)
    scenarios['breakout'] = pd.DataFrame({
        'open': breakout_prices * 0.99,
        'high': breakout_prices * 1.02,
        'low': breakout_prices * 0.98,
        'close': breakout_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates4)
    
    return scenarios

# Test Donchian seed
scenarios = create_test_scenarios()
genes = SeedGenes(seed_id='test_donchian', seed_type=SeedType.BREAKOUT, parameters={})
donchian_seed = DonchianBreakoutSeed(genes)

print('=== DONCHIAN BREAKOUT DETAILED DEBUG ===')
print('Parameters:', donchian_seed.genes.parameters)

for scenario_name, data in scenarios.items():
    print(f'\n--- {scenario_name.upper()} SCENARIO ---')
    
    # Test indicators
    indicators = donchian_seed.calculate_technical_indicators(data)
    high_channel = indicators['donchian_high']
    low_channel = indicators['donchian_low']
    close = data['close']
    
    print(f'Close range: {close.min():.2f} to {close.max():.2f}')
    print(f'High channel range: {high_channel.min():.2f} to {high_channel.max():.2f}')
    print(f'Low channel range: {low_channel.min():.2f} to {low_channel.max():.2f}')
    
    # Check breakout conditions
    above_high = (close > high_channel).sum()
    below_low = (close < low_channel).sum()
    print(f'Periods above high channel: {above_high}')
    print(f'Periods below low channel: {below_low}')
    
    # Test signals
    signals = donchian_seed.generate_signals(data)
    signal_count = (signals != 0).sum()
    print(f'Signals generated: {signal_count}')
    print(f'Signal range: {signals.min():.3f} to {signals.max():.3f}')
    
    if signal_count == 0:
        print('❌ FAILURE: No signals in', scenario_name)
    else:
        print('✅ SUCCESS:', signal_count, 'signals in', scenario_name)