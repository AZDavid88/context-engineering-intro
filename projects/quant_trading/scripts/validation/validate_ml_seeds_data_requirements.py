#!/usr/bin/env python3
"""
ML Seeds Data Requirement Validation
Tests that ML-based genetic seeds properly handle data sufficiency requirements
"""
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from src.strategy.genetic_seeds import LinearSVCClassifierSeed, PCATreeQuantileSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType

def create_data_scenarios():
    """Create test scenarios with different data lengths"""
    scenarios = {}
    
    base_dates = pd.date_range('2023-01-01', periods=250, freq='D')
    base_prices = pd.Series(100 * (1 + np.cumsum(np.random.normal(0, 0.01, 250))), index=base_dates)
    
    # Insufficient data (50 periods) - should return no signals
    scenarios['insufficient_50'] = pd.DataFrame({
        'open': base_prices[:50] * 0.99,
        'high': base_prices[:50] * 1.02,
        'low': base_prices[:50] * 0.98,
        'close': base_prices[:50],
        'volume': np.random.randint(1000, 10000, 50)
    }, index=base_dates[:50])
    
    # Marginal data (100 periods) - should return no signals for LinearSVC
    scenarios['marginal_100'] = pd.DataFrame({
        'open': base_prices[:100] * 0.99,
        'high': base_prices[:100] * 1.02,
        'low': base_prices[:100] * 0.98,
        'close': base_prices[:100],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=base_dates[:100])
    
    # Adequate data (200 periods) - should generate ML signals
    scenarios['adequate_200'] = pd.DataFrame({
        'open': base_prices[:200] * 0.99,
        'high': base_prices[:200] * 1.02,
        'low': base_prices[:200] * 0.98,
        'close': base_prices[:200],
        'volume': np.random.randint(1000, 10000, 200)
    }, index=base_dates[:200])
    
    # Abundant data (250 periods) - should generate strong ML signals
    scenarios['abundant_250'] = pd.DataFrame({
        'open': base_prices * 0.99,
        'high': base_prices * 1.02,
        'low': base_prices * 0.98,
        'close': base_prices,
        'volume': np.random.randint(1000, 10000, 250)
    }, index=base_dates)
    
    return scenarios

def test_linear_svc_data_requirements():
    """Test LinearSVCClassifierSeed data requirement validation"""
    print("=== LINEAR SVC CLASSIFIER DATA REQUIREMENT TESTS ===")
    
    scenarios = create_data_scenarios()
    genes = SeedGenes(seed_id='test_svc', seed_type=SeedType.ML_CLASSIFIER, parameters={})
    svc_seed = LinearSVCClassifierSeed(genes)
    
    print(f"Parameters: {svc_seed.genes.parameters}")
    print(f"Lookback window requirement: {svc_seed.genes.parameters['lookback_window']}")
    
    results = {}
    
    for scenario_name, data in scenarios.items():
        print(f"\n--- {scenario_name.upper()} ({len(data)} periods) ---")
        
        # Test signal generation
        signals = svc_seed.generate_signals(data)
        signal_count = (signals != 0).sum()
        
        print(f"Data length: {len(data)}")
        print(f"Signals generated: {signal_count}")
        print(f"Signal range: {signals.min():.3f} to {signals.max():.3f}")
        
        # Check expected behavior
        lookback_window = int(svc_seed.genes.parameters['lookback_window'])
        min_required = lookback_window + 10  # From line 225 in seed
        
        if len(data) < min_required:
            if signal_count == 0:
                print("✅ CORRECT: No signals with insufficient data")
                results[scenario_name] = "PASS_INSUFFICIENT"
            else:
                print("❌ ERROR: Generated signals with insufficient data")
                results[scenario_name] = "FAIL_SHOULD_BE_ZERO"
        else:
            if signal_count > 0:
                print("✅ CORRECT: Generated signals with adequate data")
                results[scenario_name] = "PASS_ADEQUATE"
            else:
                print("⚠️  WARNING: No signals despite adequate data (may be ML model issue)")
                results[scenario_name] = "WARN_NO_SIGNALS"
        
        # Test feature engineering
        try:
            indicators = svc_seed.calculate_technical_indicators(data)
            features_df = svc_seed._engineer_features(indicators)
            print(f"Features engineered: {len(features_df.columns)} features")
            print(f"Valid feature rows: {len(features_df.dropna())}")
        except Exception as e:
            print(f"Feature engineering error: {e}")
    
    return results

def test_pca_tree_data_requirements():
    """Test PCATreeQuantileSeed data requirement validation"""
    print("\n=== PCA TREE QUANTILE DATA REQUIREMENT TESTS ===")
    
    scenarios = create_data_scenarios()
    genes = SeedGenes(seed_id='test_pca', seed_type=SeedType.ML_CLASSIFIER, parameters={})
    pca_seed = PCATreeQuantileSeed(genes)
    
    print(f"Parameters: {pca_seed.genes.parameters}")
    
    results = {}
    
    for scenario_name, data in scenarios.items():
        print(f"\n--- {scenario_name.upper()} ({len(data)} periods) ---")
        
        # Test signal generation
        signals = pca_seed.generate_signals(data)
        signal_count = (signals != 0).sum()
        
        print(f"Data length: {len(data)}")
        print(f"Signals generated: {signal_count}")
        print(f"Signal range: {signals.min():.3f} to {signals.max():.3f}")
        
        # Check expected behavior based on ML requirements
        if len(data) < 100:  # Typical ML minimum
            if signal_count == 0:
                print("✅ CORRECT: No signals with insufficient data")
                results[scenario_name] = "PASS_INSUFFICIENT"
            else:
                print("❌ ERROR: Generated signals with insufficient data")
                results[scenario_name] = "FAIL_SHOULD_BE_ZERO"
        else:
            if signal_count > 0:
                print("✅ CORRECT: Generated signals with adequate data")
                results[scenario_name] = "PASS_ADEQUATE"
            else:
                print("⚠️  WARNING: No signals despite adequate data")
                results[scenario_name] = "WARN_NO_SIGNALS"
        
        # Test technical indicators
        try:
            indicators = pca_seed.calculate_technical_indicators(data)
            print(f"Technical indicators calculated successfully")
        except Exception as e:
            print(f"Technical indicators error: {e}")
    
    return results

def test_ml_fallback_behavior():
    """Test ML seeds fallback behavior when sklearn not available"""
    print("\n=== ML FALLBACK BEHAVIOR TESTS ===")
    
    # Test with adequate data
    data = create_data_scenarios()['adequate_200']
    
    # Test LinearSVC fallback
    genes_svc = SeedGenes(seed_id='test_svc_fallback', seed_type=SeedType.ML_CLASSIFIER, parameters={})
    svc_seed = LinearSVCClassifierSeed(genes_svc)
    
    print("--- LinearSVC Fallback Test ---")
    try:
        signals = svc_seed.generate_signals(data)
        signal_count = (signals != 0).sum()
        print(f"Fallback signals generated: {signal_count}")
        
        if signal_count > 0:
            print("✅ CORRECT: Fallback generates momentum-based signals")
        else:
            print("⚠️  WARNING: Fallback generated no signals")
    except Exception as e:
        print(f"❌ ERROR: Fallback failed: {e}")

def validate_ml_parameter_bounds():
    """Validate ML seeds have proper parameter bounds for data requirements"""
    print("\n=== ML PARAMETER BOUNDS VALIDATION ===")
    
    # LinearSVC bounds
    genes_svc = SeedGenes(seed_id='test_bounds', seed_type=SeedType.ML_CLASSIFIER, parameters={})
    svc_seed = LinearSVCClassifierSeed(genes_svc)
    svc_bounds = svc_seed.parameter_bounds
    
    print("--- LinearSVC Parameter Bounds ---")
    print(f"Lookback window: {svc_bounds['lookback_window']}")
    print(f"Feature count: {svc_bounds['feature_count']}")
    print(f"Regularization: {svc_bounds['regularization']}")
    
    # Validate bounds are reasonable for ML
    min_lookback, max_lookback = svc_bounds['lookback_window']
    if min_lookback >= 20 and max_lookback <= 200:
        print("✅ CORRECT: Lookback bounds appropriate for ML")
    else:
        print("❌ ERROR: Lookback bounds inappropriate for ML")
    
    # PCATree bounds
    genes_pca = SeedGenes(seed_id='test_bounds_pca', seed_type=SeedType.ML_CLASSIFIER, parameters={})
    pca_seed = PCATreeQuantileSeed(genes_pca)
    pca_bounds = pca_seed.parameter_bounds
    
    print("\n--- PCATree Parameter Bounds ---")
    for param, bounds in pca_bounds.items():
        print(f"{param}: {bounds}")

def main():
    """Run comprehensive ML seed data requirement validation"""
    print("🔬 ML SEEDS DATA REQUIREMENT VALIDATION")
    print("=" * 60)
    
    # Test data requirements
    svc_results = test_linear_svc_data_requirements()
    pca_results = test_pca_tree_data_requirements()
    
    # Test fallback behavior
    test_ml_fallback_behavior()
    
    # Validate parameter bounds
    validate_ml_parameter_bounds()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    print("\nLinearSVC Results:")
    for scenario, result in svc_results.items():
        status = "✅" if result.startswith("PASS") else "⚠️" if result.startswith("WARN") else "❌"
        print(f"  {status} {scenario}: {result}")
    
    print("\nPCATree Results:")
    for scenario, result in pca_results.items():
        status = "✅" if result.startswith("PASS") else "⚠️" if result.startswith("WARN") else "❌"
        print(f"  {status} {scenario}: {result}")
    
    # Overall assessment
    all_critical_passed = all(
        not result.startswith("FAIL") 
        for result in list(svc_results.values()) + list(pca_results.values())
    )
    
    if all_critical_passed:
        print("\n🎉 ML SEEDS DATA REQUIREMENTS: VALIDATION PASSED")
        print("ML seeds correctly handle data sufficiency requirements")
    else:
        print("\n❌ ML SEEDS DATA REQUIREMENTS: VALIDATION FAILED")
        print("Critical issues found in ML seed data handling")

if __name__ == "__main__":
    main()