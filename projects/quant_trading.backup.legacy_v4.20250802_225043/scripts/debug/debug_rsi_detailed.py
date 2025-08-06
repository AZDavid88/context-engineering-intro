#!/usr/bin/env python3
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from src.strategy.genetic_seeds import RSIFilterSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType

# Test failing scenarios
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
    
    # Oscillating market (FAILING)
    dates2 = pd.date_range('2023-01-01', periods=100, freq='D')
    oscillating_prices = pd.Series(100 + 20 * np.sin(np.linspace(0, 4*np.pi, 100)), index=dates2)
    scenarios['oscillating'] = pd.DataFrame({
        'open': oscillating_prices * 0.99,
        'high': oscillating_prices * 1.02,
        'low': oscillating_prices * 0.98,
        'close': oscillating_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates2)
    
    return scenarios

scenarios = create_test_scenarios()
genes = SeedGenes(seed_id='test_rsi', seed_type=SeedType.MEAN_REVERSION, parameters={})
rsi_seed = RSIFilterSeed(genes)

print('=== RSI FILTER DETAILED DEBUG ===')
print('Parameters:', rsi_seed.genes.parameters)

for scenario_name, data in scenarios.items():
    print(f'\n--- {scenario_name.upper()} SCENARIO ---')
    
    # Test RSI calculation
    indicators = rsi_seed.calculate_technical_indicators(data)
    rsi = indicators['rsi']
    
    oversold_threshold = rsi_seed.genes.parameters['oversold_threshold']
    overbought_threshold = rsi_seed.genes.parameters['overbought_threshold']
    
    print(f'RSI range: {rsi.min():.1f} to {rsi.max():.1f}')
    print(f'Oversold threshold: {oversold_threshold:.1f}')
    print(f'Overbought threshold: {overbought_threshold:.1f}')
    
    # Check conditions
    oversold_periods = (rsi <= oversold_threshold).sum()
    overbought_periods = (rsi >= overbought_threshold).sum()
    print(f'Oversold periods: {oversold_periods}')
    print(f'Overbought periods: {overbought_periods}')
    
    # Test signals
    signals = rsi_seed.generate_signals(data)
    signal_count = (signals != 0).sum()
    print(f'Signals generated: {signal_count}')
    print(f'Signal range: {signals.min():.3f} to {signals.max():.3f}')
    
    if signal_count == 0:
        print('❌ FAILURE: No signals in', scenario_name)
    else:
        print('✅ SUCCESS:', signal_count, 'signals in', scenario_name)