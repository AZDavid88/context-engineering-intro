#!/usr/bin/env python3
"""
Test Research-Compliant Genetic Engine Following DEAP Patterns Exactly

This follows the research pattern from /research/deap/4_multiprocessing_distributed_evaluation.md
exactly, with the __main__ protection and simple Pool management.
"""

import multiprocessing
import pandas as pd
import numpy as np
from src.strategy.genetic_engine_research_compliant import create_research_compliant_engine, ResearchCompliantConfig

def test_single_threaded():
    """Test single-threaded evolution."""
    print('🧬 Testing single-threaded evolution...')
    
    # Create synthetic data
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 200),
        'high': np.random.uniform(105, 115, 200),
        'low': np.random.uniform(95, 105, 200),
        'close': np.random.uniform(100, 110, 200),
        'volume': np.random.uniform(1000, 5000, 200)
    }, index=dates)
    
    # Single-threaded config
    config = ResearchCompliantConfig(
        population_size=10,
        n_generations=3,
        use_multiprocessing=False  # KEY: No multiprocessing
    )
    
    engine = create_research_compliant_engine()
    engine.config = config
    
    results = engine.evolve(data)
    
    print(f'✅ Single-threaded: {results.status}, Best: {results.best_individual.seed_name}')
    return True

def main():
    """Main function with __main__ protection as per research patterns."""
    try:
        test_single_threaded()
        print('✅ ALL TESTS PASSED')
    except Exception as e:
        print(f'❌ TEST FAILED: {e}')
        import traceback
        traceback.print_exc()

# CRITICAL: Research requires __main__ protection
if __name__ == "__main__":
    main()