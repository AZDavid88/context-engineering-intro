#!/usr/bin/env python3
"""
Complete Pipeline Integration Test - Discovery → Data → Strategy

Tests the complete end-to-end pipeline following the anti-hallucination protocol:
CONTEXT AT /workspaces/context-engineering-intro/projects/quant_trading/research

Pipeline Flow:
1. Discovery: enhanced_asset_filter.py -> Filter tradeable assets 
2. Data Collection: dynamic_asset_data_collector.py -> Download 1h/15m OHLCV
3. Strategy Evolution: genetic_engine_research_compliant.py -> Discover high-alpha strategies

Validation Requirements:
- Real asset discovery and filtering
- Multi-timeframe data collection with proper OHLCV + parameters
- Genetic seeds integration with collected data
- End-to-end strategy evolution
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Pipeline imports
from src.discovery.enhanced_asset_filter import EnhancedAssetFilter
from src.data.dynamic_asset_data_collector import DynamicAssetDataCollector
from src.strategy.genetic_engine_research_compliant import create_research_compliant_engine, ResearchCompliantConfig
from src.config.settings import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineIntegrationTester:
    """Complete pipeline integration tester."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.settings = get_settings()
        self.asset_filter = None
        self.data_collector = None
        self.genetic_engine = None
        
        # Pipeline results
        self.filtered_assets = []
        self.collected_datasets = {}
        self.evolution_results = None
        
    async def test_stage_1_discovery(self) -> List[str]:
        """Test Stage 1: Asset Discovery and Filtering."""
        logger.info("🔍 STAGE 1: ASSET DISCOVERY AND FILTERING")
        
        try:
            # Initialize enhanced asset filter
            self.asset_filter = EnhancedAssetFilter(self.settings)
            
            # Apply enhanced filtering (this discovers and filters in one step)
            logger.info("⚡ Applying enhanced filtering...")
            filtered_assets, asset_metrics = await self.asset_filter.filter_universe(
                assets=None,  # Discover all assets
                refresh_cache=True,
                enable_optimizations=True
            )
            
            # Limit for testing
            self.filtered_assets = filtered_assets[:10] if len(filtered_assets) > 10 else filtered_assets
            
            logger.info(f"   Found {len(filtered_assets)} total filtered assets")
            logger.info(f"   Using {len(self.filtered_assets)} for testing")
            
            logger.info(f"✅ Stage 1 Complete: {len(self.filtered_assets)} tradeable assets filtered")
            logger.info(f"   Assets: {self.filtered_assets[:5]}...")  # Show first 5
            # Get timing from enhanced metrics if available
            if hasattr(self.asset_filter, 'enhanced_metrics'):
                filter_time = self.asset_filter.enhanced_metrics.filter_duration_seconds
                logger.info(f"   Filter time: {filter_time:.2f}s")
            
            if not self.filtered_assets:
                raise ValueError("No assets passed filtering criteria")
                
            return self.filtered_assets
            
        except Exception as e:
            logger.error(f"❌ Stage 1 Failed: {e}")
            raise
    
    async def test_stage_2_data_collection(self, assets: List[str]) -> Dict:
        """Test Stage 2: Multi-timeframe Data Collection."""
        logger.info("📈 STAGE 2: MULTI-TIMEFRAME DATA COLLECTION")
        
        try:
            # Initialize data collector
            self.data_collector = DynamicAssetDataCollector(self.settings)
            
            # Collect multi-timeframe data  
            test_assets = assets[:3]  # Limit to 3 assets for testing
            logger.info(f"🔄 Collecting data for {len(test_assets)} assets...")
            collection_results = await self.data_collector.collect_assets_data_pipeline(
                discovered_assets=test_assets,
                include_enhanced_data=True
            )
            
            self.collected_datasets = collection_results.get('datasets', {})
            collection_metrics = collection_results.get('metrics', {})
            
            # Validate collected data
            total_bars = 0
            for asset, dataset in self.collected_datasets.items():
                if hasattr(dataset, 'data_1h'):
                    total_bars += len(dataset.data_1h)
                if hasattr(dataset, 'data_15m'):
                    total_bars += len(dataset.data_15m)
            
            logger.info(f"✅ Stage 2 Complete: {len(self.collected_datasets)} assets with data")
            logger.info(f"   Total bars collected: {total_bars}")
            logger.info(f"   Collection time: {collection_metrics.get('total_collection_time', 0):.2f}s")
            
            # Validate data quality for genetic seeds
            for asset, dataset in self.collected_datasets.items():
                if hasattr(dataset, 'data_1h') and len(dataset.data_1h) > 0:
                    data_1h = dataset.data_1h
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in data_1h.columns]
                    if missing_cols:
                        logger.warning(f"⚠️  Asset {asset} missing columns: {missing_cols}")
                    else:
                        logger.info(f"✅ Asset {asset} has all required OHLCV columns")
            
            if not self.collected_datasets:
                raise ValueError("No data collected for any assets")
                
            return self.collected_datasets
            
        except Exception as e:
            logger.error(f"❌ Stage 2 Failed: {e}")
            raise
    
    def test_stage_3_strategy_evolution(self, datasets: Dict) -> Dict:
        """Test Stage 3: Genetic Strategy Evolution."""
        logger.info("🧬 STAGE 3: GENETIC STRATEGY EVOLUTION")
        
        try:
            # Select best dataset for evolution (most data)
            best_asset = None
            best_data = None
            max_bars = 0
            
            for asset, dataset in datasets.items():
                if hasattr(dataset, 'data_1h') and len(dataset.data_1h) > max_bars:
                    max_bars = len(dataset.data_1h)
                    best_asset = asset
                    best_data = dataset.data_1h
            
            if best_data is None or len(best_data) < 50:
                raise ValueError("Insufficient data for genetic evolution")
            
            logger.info(f"🎯 Using {best_asset} with {len(best_data)} bars for evolution")
            
            # Configure genetic engine for testing
            config = ResearchCompliantConfig(
                population_size=20,        # Small population for testing
                n_generations=5,           # Few generations for testing  
                use_multiprocessing=False, # Single-threaded for stability
                fitness_weights=(1.0, -1.0, 1.0)  # Sharpe, drawdown, consistency
            )
            
            # Create and configure engine
            self.genetic_engine = create_research_compliant_engine()
            self.genetic_engine.config = config
            
            logger.info(f"🏗️  Genetic engine configured with {len(self.genetic_engine.genetic_seeds)} seeds")
            
            # Run evolution
            logger.info("🚀 Running genetic evolution...")
            evolution_start = time.time()
            
            self.evolution_results = self.genetic_engine.evolve(best_data)
            
            evolution_time = time.time() - evolution_start
            
            # Analyze results
            logger.info(f"✅ Stage 3 Complete: Evolution successful!")
            logger.info(f"   Status: {self.evolution_results.status}")
            logger.info(f"   Best Strategy: {self.evolution_results.best_individual.seed_name}")
            logger.info(f"   Evolution time: {evolution_time:.2f}s")
            logger.info(f"   Population evolved: {len(self.evolution_results.population)} individuals")
            logger.info(f"   Generations completed: {len(self.evolution_results.generation_stats)}")
            
            # Validate strategy fitness
            if self.evolution_results.generation_stats:
                final_stats = self.evolution_results.generation_stats[-1]
                best_fitness = final_stats.get('max', [0, 0, 0])
                logger.info(f"   Final best fitness: {best_fitness}")
                
                # Check for high-alpha potential (Sharpe > 1.0)
                if isinstance(best_fitness, (list, tuple)) and len(best_fitness) > 0:
                    sharpe_ratio = best_fitness[0]
                    if sharpe_ratio > 1.0:
                        logger.info(f"🎯 HIGH-ALPHA STRATEGY DISCOVERED! Sharpe: {sharpe_ratio:.3f}")
                    else:
                        logger.info(f"📊 Strategy evolved with Sharpe: {sharpe_ratio:.3f}")
            
            return {
                'status': self.evolution_results.status,
                'best_strategy': self.evolution_results.best_individual.seed_name,
                'evolution_time': evolution_time,
                'population_size': len(self.evolution_results.population),
                'generations': len(self.evolution_results.generation_stats)
            }
            
        except Exception as e:
            logger.error(f"❌ Stage 3 Failed: {e}")
            raise
    
    def generate_pipeline_summary(self) -> Dict:
        """Generate comprehensive pipeline test summary."""
        return {
            'pipeline_status': 'SUCCESS' if self.evolution_results else 'FAILED',
            'stage_1_assets_filtered': len(self.filtered_assets),
            'stage_2_datasets_collected': len(self.collected_datasets),
            'stage_3_evolution_completed': self.evolution_results is not None,
            'best_strategy_discovered': (
                self.evolution_results.best_individual.seed_name 
                if self.evolution_results else None
            ),
            'total_genetic_seeds_available': (
                len(self.genetic_engine.genetic_seeds) 
                if self.genetic_engine else 0
            )
        }

async def main():
    """Main pipeline integration test."""
    logger.info("🚀 COMPLETE PIPELINE INTEGRATION TEST")
    logger.info("=" * 60)
    
    tester = PipelineIntegrationTester()
    
    try:
        # Stage 1: Discovery
        filtered_assets = await tester.test_stage_1_discovery()
        
        # Stage 2: Data Collection  
        collected_datasets = await tester.test_stage_2_data_collection(filtered_assets)
        
        # Stage 3: Strategy Evolution
        evolution_results = tester.test_stage_3_strategy_evolution(collected_datasets)
        
        # Generate summary
        summary = tester.generate_pipeline_summary()
        
        logger.info("=" * 60)
        logger.info("🎯 PIPELINE INTEGRATION TEST COMPLETE")
        logger.info(f"   Status: {summary['pipeline_status']}")
        logger.info(f"   Assets Filtered: {summary['stage_1_assets_filtered']}")
        logger.info(f"   Datasets Collected: {summary['stage_2_datasets_collected']}")
        logger.info(f"   Best Strategy: {summary['best_strategy_discovered']}")
        logger.info(f"   Genetic Seeds Used: {summary['total_genetic_seeds_available']}")
        logger.info("✅ ALL PIPELINE STAGES INTEGRATED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"❌ PIPELINE INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Generate failure summary
        summary = tester.generate_pipeline_summary()
        logger.error(f"Failure Summary: {summary}")

# CRITICAL: Research requires __main__ protection for multiprocessing
if __name__ == "__main__":
    asyncio.run(main())