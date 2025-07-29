#!/usr/bin/env python3
"""
Quick validation of genetic algorithm runs with proper parameters.
This replaces documentation - we TEST the actual behavior.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
from src.execution.retail_connection_optimizer import RetailConnectionOptimizer
from src.config.settings import get_settings

async def test_genetic_run_quality():
    """Test actual genetic algorithm run with research-driven parameters."""
    print("🧬 Testing Genetic Algorithm with Research-Driven Parameters...")
    
    # Create sample market data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 1000),
        'high': np.random.uniform(100, 200, 1000),
        'low': np.random.uniform(100, 200, 1000),
        'close': np.random.uniform(100, 200, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    sample_data.set_index('timestamp', inplace=True)
    
    # Create genetic pool with small population for testing
    mock_optimizer = RetailConnectionOptimizer(get_settings())
    
    pool = GeneticStrategyPool(
        connection_optimizer=mock_optimizer,
        evolution_config=EvolutionConfig(population_size=5, generations=2),
        use_ray=False
    )
    
    # Test population initialization
    print("📊 Initializing population with seed-specific parameters...")
    pop_size = await pool.initialize_population()
    print(f"✅ Population initialized: {pop_size} individuals")
    
    # Analyze parameter quality
    print("\n🔍 Parameter Analysis:")
    for i, individual in enumerate(pool.population[:3]):  # Check first 3
        print(f"\n  Individual {i+1} ({individual.seed_type.value}):")
        for param, value in individual.genes.parameters.items():
            print(f"    {param}: {value:.3f}")
    
    # Test evolution cycle
    print("\n🔄 Running evolution cycle...")
    try:
        best_individuals = await pool.evolve_strategies(
            market_data=sample_data,
            generations=2
        )
        
        print(f"✅ Evolution completed successfully!")
        print(f"📈 Best individuals found: {len(best_individuals)}")
        print(f"🏥 Final health score: {pool.evolution_history[-1].health_score:.1f}/100")
        
        # Show evolution progress
        print("\n📊 Evolution History:")
        for i, metrics in enumerate(pool.evolution_history):
            print(f"  Gen {i}: Health={metrics.health_score:.1f}, Best={metrics.best_fitness:.3f}, Avg={metrics.average_fitness:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evolution failed: {e}")
        return False

async def validate_core_functionality():
    """Quick validation that replaces documentation claims."""
    print("🔍 CORE FUNCTIONALITY VALIDATION")
    print("=" * 50)
    
    # Test genetic algorithm
    ga_success = await test_genetic_run_quality()
    
    # Test research context exists
    research_paths = [
        "research/prometheus_python_official/research_summary.md",
        "research/grafana_pyroscope_official/research_summary.md",
    ]
    
    research_ok = all(os.path.exists(path) for path in research_paths)
    print(f"\n📚 Research Context: {'✅ Available' if research_ok else '❌ Missing'}")
    
    # Overall status
    print(f"\n🚀 DEPLOYMENT READINESS:")
    print(f"  Genetic Algorithm: {'✅ READY' if ga_success else '❌ NOT READY'}")
    print(f"  Research Context: {'✅ READY' if research_ok else '❌ NOT READY'}")
    
    return ga_success and research_ok

if __name__ == "__main__":
    print(f"🔬 GENETIC ALGORITHM VALIDATION - {datetime.now()}")
    print("This replaces documentation - we test actual behavior!\n")
    
    result = asyncio.run(validate_core_functionality())
    
    if result:
        print("\n🎉 ALL SYSTEMS GO - READY FOR PRODUCTION!")
    else:
        print("\n🚨 VALIDATION FAILED - NOT READY FOR DEPLOYMENT!")
    
    sys.exit(0 if result else 1)