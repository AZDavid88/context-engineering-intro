#!/usr/bin/env python3
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from src.strategy.genetic_seeds import LinearSVCClassifierSeed
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

# Test SVC seed
genes = SeedGenes(seed_id='test_svc', seed_type=SeedType.ML_CLASSIFIER, parameters={})
svc_seed = LinearSVCClassifierSeed(genes)

print('=== LINEAR SVC DEBUG ===')
print('Parameters:', svc_seed.genes.parameters)

try:
    # Test signal generation
    signals = svc_seed.generate_signals(data)
    print('Signals generated:', (signals != 0).sum())
    print('Signal range:', signals.min(), 'to', signals.max())
except Exception as e:
    print('Error:', str(e))
    import traceback
    traceback.print_exc()