#!/usr/bin/env python3
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from src.strategy.genetic_seeds import StochasticOscillatorSeed
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
    
    # Oscillating market (WORKING)
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
genes = SeedGenes(seed_id='test_stoch', seed_type=SeedType.MOMENTUM, parameters={})
stoch_seed = StochasticOscillatorSeed(genes)

print('=== STOCHASTIC OSCILLATOR DETAILED DEBUG ===')
print('Parameters:', stoch_seed.genes.parameters)

for scenario_name, data in scenarios.items():
    print(f'\n--- {scenario_name.upper()} SCENARIO ---')
    
    # Test stochastic calculation
    indicators = stoch_seed.calculate_technical_indicators(data)
    k_percent = indicators['k_percent']
    d_percent = indicators['d_percent']
    
    overbought_level = stoch_seed.genes.parameters['overbought_level']
    oversold_level = stoch_seed.genes.parameters['oversold_level']
    
    print(f'%K range: {k_percent.min():.1f} to {k_percent.max():.1f}')
    print(f'%D range: {d_percent.min():.1f} to {d_percent.max():.1f}')
    print(f'Oversold threshold: {oversold_level:.1f}')
    print(f'Overbought threshold: {overbought_level:.1f}')
    
    # Check conditions for each signal type
    oversold_periods = (k_percent <= oversold_level).sum()
    overbought_periods = (k_percent >= overbought_level).sum()
    crossovers_up = indicators['k_crosses_above_d'].sum()
    crossovers_down = indicators['k_crosses_below_d'].sum()
    
    print(f'Oversold periods: {oversold_periods}')
    print(f'Overbought periods: {overbought_periods}')
    print(f'K crosses above D: {crossovers_up}')
    print(f'K crosses below D: {crossovers_down}')
    
    # Test individual signal components
    crossover_signals = stoch_seed._generate_crossover_signals(indicators)
    zone_signals = stoch_seed._generate_zone_signals(indicators)
    divergence_signals = stoch_seed._generate_divergence_signals(indicators)
    
    print(f'Crossover signals: {(crossover_signals != 0).sum()} / max: {crossover_signals.abs().max():.3f}')
    print(f'Zone signals: {(zone_signals != 0).sum()} / max: {zone_signals.abs().max():.3f}')
    print(f'Divergence signals: {(divergence_signals != 0).sum()} / max: {divergence_signals.abs().max():.3f}')
    
    # Test final signals
    signals = stoch_seed.generate_signals(data)
    signal_count = (signals != 0).sum()
    print(f'Final signals generated: {signal_count}')
    print(f'Signal range: {signals.min():.3f} to {signals.max():.3f}')
    
    if signal_count == 0:
        print('❌ FAILURE: No signals in', scenario_name)
        
        # Debug why crossover signals fail
        bullish_crossover = (
            indicators['k_crosses_above_d'] &
            indicators['oversold_zone'] &
            (indicators['k_momentum'] > 0)
        )
        bearish_crossover = (
            indicators['k_crosses_below_d'] &
            indicators['overbought_zone'] &
            (indicators['k_momentum'] < 0)
        )
        volume_confirmed = indicators['volume_ratio'] >= 1.0
        momentum_up = indicators['price_momentum'] > 0
        momentum_down = indicators['price_momentum'] < 0
        
        print(f'  Bullish crossover conditions: {bullish_crossover.sum()}')
        print(f'  Bearish crossover conditions: {bearish_crossover.sum()}')
        print(f'  Volume confirmed periods: {volume_confirmed.sum()}')
        print(f'  Momentum up periods: {momentum_up.sum()}')
        print(f'  Momentum down periods: {momentum_down.sum()}')
        
        # Show the AND logic breakdown
        buy_conditions = (
            bullish_crossover &
            volume_confirmed &
            momentum_up
        )
        sell_conditions = (
            bearish_crossover &
            volume_confirmed &
            momentum_down
        )
        print(f'  Final buy conditions (ALL must be true): {buy_conditions.sum()}')
        print(f'  Final sell conditions (ALL must be true): {sell_conditions.sum()}')
    else:
        print('✅ SUCCESS:', signal_count, 'signals in', scenario_name)