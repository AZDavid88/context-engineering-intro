#!/usr/bin/env python3
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from src.strategy.genetic_seeds import EMACrossoverSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType

# Create test data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
trend_prices = pd.Series(100 * (1 + np.linspace(0, 0.5, 100)), index=dates)
data = pd.DataFrame({
    'open': trend_prices * 0.99,
    'high': trend_prices * 1.02,
    'low': trend_prices * 0.98,
    'close': trend_prices,
    'volume': np.random.randint(1000, 10000, 100)
}, index=dates)

# Test EMA seed
genes = SeedGenes(seed_id='test_ema', seed_type=SeedType.MOMENTUM, parameters={})
ema_seed = EMACrossoverSeed(genes)

print('=== EMA CROSSOVER DEBUG ===')
print('Parameters:', ema_seed.genes.parameters)

# Check indicators
indicators = ema_seed.calculate_technical_indicators(data)
print('Fast EMA range:', indicators['fast_ema'].min(), 'to', indicators['fast_ema'].max())
print('Slow EMA range:', indicators['slow_ema'].min(), 'to', indicators['slow_ema'].max())
print('EMA spread range:', indicators['ema_spread'].min(), 'to', indicators['ema_spread'].max())

# Check signal conditions
fast_ema = indicators['fast_ema']
slow_ema = indicators['slow_ema']
crossover_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
crossover_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

print('Crossover up periods:', crossover_up.sum())
print('Crossover down periods:', crossover_down.sum())
print('Fast > Slow periods:', (fast_ema > slow_ema).sum())

# Test signal generation
signals = ema_seed.generate_signals(data)
print('Signals generated:', (signals != 0).sum())
print('Signal range:', signals.min(), 'to', signals.max())