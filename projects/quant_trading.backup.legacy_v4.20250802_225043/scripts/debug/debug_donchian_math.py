#!/usr/bin/env python3
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from src.strategy.genetic_seeds import DonchianBreakoutSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType

# Create trending data
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

genes = SeedGenes(seed_id='test_donchian', seed_type=SeedType.BREAKOUT, parameters={})
donchian_seed = DonchianBreakoutSeed(genes)

print('=== DONCHIAN MATH DEBUG ===')
print('Parameters:', donchian_seed.genes.parameters)

indicators = donchian_seed.calculate_technical_indicators(data)
close = data['close']
high_channel = indicators['donchian_high']
low_channel = indicators['donchian_low']

# Examine the math
breakout_threshold = donchian_seed.genes.parameters['breakout_threshold']
breakout_factor = 1.0 + breakout_threshold
print(f'Breakout threshold: {breakout_threshold}')
print(f'Breakout factor: {breakout_factor}')

# Check the conditions
print(f'\nLast 10 periods:')
for i in range(-10, 0):
    idx = i
    c = close.iloc[idx]
    hc = high_channel.iloc[idx]
    hc_factor = hc * breakout_factor
    above_high = c > hc
    above_breakout = c > hc_factor
    print(f'Period {idx:2d}: Close={c:6.2f}, High={hc:6.2f}, Factor={hc_factor:6.2f}, Above_High={above_high}, Above_Breakout={above_breakout}')

# Count actual breakouts
upper_breakout = close > (high_channel * breakout_factor)
lower_breakout = close < (low_channel * (2.0 - breakout_factor))

print(f'\nBreakout analysis:')
print(f'Periods above high channel: {(close > high_channel).sum()}')
print(f'Periods above breakout threshold: {upper_breakout.sum()}')
print(f'Periods below low channel: {(close < low_channel).sum()}')
print(f'Periods below breakout threshold: {lower_breakout.sum()}')