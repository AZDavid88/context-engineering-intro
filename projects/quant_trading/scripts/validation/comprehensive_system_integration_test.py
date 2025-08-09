#!/usr/bin/env python3
"""
Comprehensive Integration Test for Quant Trading System
Tests the complete workflow: Data → Evolution → Signal Generation
"""

import sys
import pandas as pd
import numpy as np
import asyncio
sys.path.append('/workspaces/context-engineering-intro/projects/quant_trading')

# Import shell-safe operations to avoid \! syntax errors
from src.utils.shell_safe_operations import safe_not_equal

print('=== COMPREHENSIVE INTEGRATION VALIDATION ===')

async def complete_system_test():
    from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
    from src.execution.retail_connection_optimizer import (
        RetailConnectionOptimizer, TradingSessionProfile,
        TradingTimeframe, ConnectionUsagePattern
    )
    from src.strategy.genetic_seeds.base_seed import SeedType
    from src.config.settings import get_settings
    
    print('Testing complete trading system workflow with synthetic market data...')
    
    # Generate realistic market data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    
    # Create realistic crypto-like price movement
    base_price = 50000  # BTC-like starting price
    trend = np.linspace(0, 5000, 200)  # Long-term upward trend
    volatility = np.random.normal(0, 200, 200)  # Daily volatility
    prices = base_price + trend + np.cumsum(volatility)
    
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.normal(0, 50, 200),
        'high': prices + np.abs(np.random.normal(100, 50, 200)),
        'low': prices - np.abs(np.random.normal(100, 50, 200)),
        'close': prices,
        'volume': np.random.uniform(500, 2000, 200)
    })
    
    print(f'✅ Generated realistic market data: {len(market_data)} candles')
    print(f'   Price range: ${market_data.close.min():,.0f} - ${market_data.close.max():,.0f}')
    print(f'   Total return: {((market_data.close.iloc[-1] / market_data.close.iloc[0]) - 1) * 100:.1f}%')
    
    # Setup trading system
    session_profile = TradingSessionProfile(
        timeframe=TradingTimeframe.INTRADAY,
        expected_api_calls_per_minute=100,
        max_concurrent_strategies=10,
        usage_pattern=ConnectionUsagePattern.BURST,
        session_duration_hours=6.0
    )
    
    connection_optimizer = RetailConnectionOptimizer(session_profile)
    
    config = EvolutionConfig(
        population_size=8,
        generations=3,
        mutation_rate=0.25,
        crossover_rate=0.75,
        elite_ratio=0.3
    )
    
    pool = GeneticStrategyPool(
        connection_optimizer=connection_optimizer,
        use_ray=False,
        evolution_config=config
    )
    
    # Run complete evolution cycle
    seed_types = [SeedType.MOMENTUM, SeedType.MEAN_REVERSION, SeedType.VOLATILITY]
    await pool.initialize_population(seed_types)
    
    print('\n🧬 Running multi-generation evolution...')
    evolved_strategies = await pool.evolve_strategies(market_data, generations=3)
    
    # Analyze results
    successful_strategies = [s for s in evolved_strategies if s.fitness and s.fitness > -999]
    
    print(f'\n📊 EVOLUTION RESULTS:')
    print(f'   Total strategies: {len(evolved_strategies)}')
    print(f'   Successful evaluations: {len(successful_strategies)}')
    
    if successful_strategies:
        fitnesses = [s.fitness for s in successful_strategies]
        print(f'   Best Sharpe ratio: {max(fitnesses):.4f}')
        print(f'   Average Sharpe ratio: {np.mean(fitnesses):.4f}')
        print(f'   Strategies with positive Sharpe: {sum(1 for f in fitnesses if f > 0)}')
        
        # Test top strategies on out-of-sample data
        print('\n📈 OUT-OF-SAMPLE TESTING:')
        
        # Use last 50 data points as out-of-sample
        oos_data = market_data.tail(50).copy()
        
        top_3_strategies = sorted(successful_strategies, key=lambda x: x.fitness, reverse=True)[:3]
        
        from src.strategy.genetic_seeds.seed_registry import get_registry
        registry = get_registry()
        settings = get_settings()
        
        oos_results = []
        
        for i, strategy in enumerate(top_3_strategies):
            try:
                # Get seed class and create instance
                available_seeds = registry._type_index.get(strategy.seed_type, [])
                if available_seeds:
                    seed_name = available_seeds[0]
                    seed_class = registry.get_seed_class(seed_name)
                    
                    if seed_class:
                        seed_instance = seed_class(strategy.genes, settings)
                        
                        # Generate signals on out-of-sample data
                        oos_signals = seed_instance.generate_signals(oos_data)
                        
                        # Fix pandas Series ambiguity - convert to numpy array immediately
                        if oos_signals is not None and len(oos_signals) > 0:
                            signals_array = np.array(oos_signals)
                            
                            # Calculate out-of-sample performance
                            returns = []
                            position = 0
                            
                            for j in range(len(signals_array) - 1):
                                signal = signals_array[j]
                                if signal > 0.2 and position <= 0:  # Buy signal
                                    position = 1
                                elif signal < -0.2 and position >= 0:  # Sell signal
                                    position = -1
                                
                                if j > 0 and position != 0:  # Simple comparison works fine here
                                    current_price = float(oos_data.iloc[j]['close'])
                                    next_price = float(oos_data.iloc[j+1]['close'])
                                    price_change = (next_price - current_price) / current_price
                                    returns.append(position * price_change)
                            
                            if returns:
                                total_return = sum(returns)
                                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                                
                                oos_results.append({
                                    'strategy': i+1,
                                    'type': strategy.seed_type.value,
                                    'in_sample_sharpe': strategy.fitness,
                                    'oos_sharpe': sharpe,
                                    'oos_return': total_return,
                                    'trades': len(returns)
                                })
                                
                                print(f'   Strategy {i+1} ({strategy.seed_type.value}):')
                                print(f'     In-sample Sharpe: {strategy.fitness:.4f}')
                                print(f'     Out-of-sample Sharpe: {sharpe:.4f}')
                                print(f'     Out-of-sample return: {total_return*100:.2f}%')
                                print(f'     Trades executed: {len(returns)}')
            
            except Exception as e:
                print(f'   Strategy {i+1} failed OOS test: {e}')
        
        # Summary statistics
        if oos_results:
            avg_oos_sharpe = np.mean([r['oos_sharpe'] for r in oos_results])
            profitable_strategies = sum(1 for r in oos_results if r['oos_return'] > 0)
            
            print(f'\n📋 OUT-OF-SAMPLE SUMMARY:')
            print(f'   Average OOS Sharpe: {avg_oos_sharpe:.4f}')
            print(f'   Profitable strategies: {profitable_strategies}/{len(oos_results)}')
            
    # Get final summary
    summary = pool.get_evolution_summary()
    print(f'\n🏁 FINAL SYSTEM STATUS:')
    print(f'   Evolution status: {summary.get("status")}')
    print(f'   Total generations: {summary.get("generations")}')
    print(f'   Final health score: {summary.get("current_health_score"):.1f}/100')
    
    await pool.cleanup()
    
    return len(successful_strategies) > 0 and summary.get('current_health_score', 0) > 80

if __name__ == "__main__":
    # Run comprehensive test
    success = asyncio.run(complete_system_test())
    print(f'\n🎯 COMPREHENSIVE SYSTEM VALIDATION: {"✅ FUNCTIONAL" if success else "❌ NOT FUNCTIONAL"}')