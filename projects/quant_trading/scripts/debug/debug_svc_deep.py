#!/usr/bin/env python3
import sys
sys.path.append('.')
import pandas as pd
import numpy as np

# Test sklearn availability
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    print("✅ sklearn available")
    sklearn_available = True
except ImportError as e:
    print(f"❌ sklearn not available: {e}")
    sklearn_available = False

# Test original LinearSVCClassifierSeed
from src.strategy.genetic_seeds import LinearSVCClassifierSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType

# Create test data with more history for ML
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=200, freq='D')  # More data for ML
trend_prices = pd.Series(100 * (1 + np.linspace(0, 0.5, 200)), index=dates)
data = pd.DataFrame({
    'open': trend_prices * 0.99,
    'high': trend_prices * 1.02,
    'low': trend_prices * 0.98,
    'close': trend_prices,
    'volume': np.random.randint(1000, 10000, 200)
}, index=dates)

# Test with minimal parameters to reduce ML complexity
genes = SeedGenes(
    seed_id='test_svc', 
    seed_type=SeedType.ML_CLASSIFIER, 
    parameters={
        'lookback_window': 50.0,     # Reduce from 100
        'feature_count': 3.0,        # Reduce from 8
        'regularization': 1.0,
        'ensemble_size': 1.0,        # Single model
        'cross_validation': 3.0      # Minimal CV
    }
)

svc_seed = LinearSVCClassifierSeed(genes)

print('=== LINEAR SVC DEEP DEBUG ===')
print('Parameters:', svc_seed.genes.parameters)
print('Data length:', len(data))

try:
    # Test each component separately
    print('\n1. Testing technical indicators...')
    indicators = svc_seed.calculate_technical_indicators(data)
    print(f'   Indicators calculated: {len(indicators)} features')
    
    print('\n2. Testing feature engineering...')
    features = svc_seed._engineer_features(indicators)
    print(f'   Features engineered: {features.shape}')
    print(f'   Features: {list(features.columns)}')
    
    print('\n3. Testing signal generation...')
    signals = svc_seed.generate_signals(data)
    print(f'   Signals generated: {(signals != 0).sum()}')
    print(f'   Signal range: {signals.min()} to {signals.max()}')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()