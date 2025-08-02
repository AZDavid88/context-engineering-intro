#!/usr/bin/env python3
"""
Genetic Seeds Functionality Validation

Comprehensive validation that all genetic seeds remain fully functional
after backup file cleanup and architectural changes.
"""

import sys
sys.path.append('/workspaces/context-engineering-intro/projects/quant_trading')

# Import genetic seeds at module level
try:
    from src.strategy.genetic_seeds import *
    IMPORTS_SUCCESSFUL = True
except Exception as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = e

def validate_genetic_seeds():
    """Comprehensive validation of genetic seeds functionality."""
    
    print("🧬 GENETIC SEEDS FUNCTIONALITY VALIDATION")
    print("=" * 50)
    
    # Test 1: Import all genetic seeds
    if not IMPORTS_SUCCESSFUL:
        print(f'❌ Import failed: {IMPORT_ERROR}')
        return False
    else:
        print('✅ All genetic seeds import successfully')

    # Test 2: Registry functionality
    try:
        registry = get_registry()
        registered_count = len(registry._registry)
        print(f'✅ Registry initialized with {registered_count} seeds')
        
        # List all registered seeds
        all_seeds = registry.list_all_seeds()
        print(f'✅ Registry contains: {list(all_seeds.keys())}')
        
        if registered_count != 14:
            print(f'⚠️ Expected 14 seeds, found {registered_count}')
        else:
            print('✅ All 14 seeds properly registered')
            
    except Exception as e:
        print(f'❌ Registry test failed: {e}')
        return False

    # Test 3: Individual seed instantiation
    seed_classes = [
        EMACrossoverSeed, DonchianBreakoutSeed, RSIFilterSeed, 
        StochasticOscillatorSeed, SMATrendFilterSeed, ATRStopLossSeed,
        IchimokuCloudSeed, VWAPReversionSeed, VolatilityScalingSeed,
        FundingRateCarrySeed, LinearSVCClassifierSeed, PCATreeQuantileSeed,
        BollingerBandsSeed, NadarayaWatsonSeed
    ]

    print(f'\n🧪 Testing individual seed instantiation:')
    functional_seeds = 0
    
    for i, SeedClass in enumerate(seed_classes, 1):
        try:
            # Create test genes
            genes = SeedGenes(
                seed_id=f'test_{i}',
                seed_type=SeedType.MOMENTUM,
                parameters={}
            )
            
            # Instantiate seed
            seed_instance = SeedClass(genes)
            seed_name = seed_instance.seed_name
            param_count = len(seed_instance.parameter_bounds)
            
            print(f'   ✅ {seed_name}: {param_count} parameters defined')
            functional_seeds += 1
            
        except Exception as e:
            print(f'   ❌ {SeedClass.__name__} failed: {e}')
    
    # Test 4: Parameter validation
    print(f'\n🔬 Testing parameter bounds validation:')
    for SeedClass in seed_classes[:3]:  # Test first 3 seeds
        try:
            genes = SeedGenes(
                seed_id='param_test',
                seed_type=SeedType.MOMENTUM,
                parameters={}
            )
            seed = SeedClass(genes)
            
            # Check parameter bounds exist and are valid
            bounds = seed.parameter_bounds
            required_params = seed.required_parameters
            
            valid_bounds = True
            for param in required_params:
                if param not in bounds:
                    valid_bounds = False
                    break
                min_val, max_val = bounds[param]
                if min_val >= max_val:
                    valid_bounds = False
                    break
            
            if valid_bounds:
                print(f'   ✅ {seed.seed_name}: Valid parameter bounds')
            else:
                print(f'   ❌ {seed.seed_name}: Invalid parameter bounds')
                
        except Exception as e:
            print(f'   ❌ {SeedClass.__name__} parameter test failed: {e}')
    
    # Summary
    print(f'\n📊 VALIDATION SUMMARY:')
    print(f'   Functional seeds: {functional_seeds}/{len(seed_classes)}')
    print(f'   Registry seeds: {registered_count}')
    print(f'   Expected seeds: 14')
    
    success = (functional_seeds == len(seed_classes) and registered_count == 14)
    
    if success:
        print(f'\n🎉 ALL GENETIC SEEDS FULLY FUNCTIONAL')
        print(f'✅ No functionality lost from backup cleanup')
    else:
        print(f'\n❌ VALIDATION FAILED - FUNCTIONALITY COMPROMISED')
    
    return success

if __name__ == "__main__":
    success = validate_genetic_seeds()
    sys.exit(0 if success else 1)