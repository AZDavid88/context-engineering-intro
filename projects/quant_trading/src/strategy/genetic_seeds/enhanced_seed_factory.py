"""
Enhanced Seed Factory - Universal Correlation Enhancement Registry

This module provides automatic registration of correlation-enhanced versions
of all genetic seeds using the Universal Correlation Enhancer.

Key Features:
- Automatic discovery and enhancement of all genetic seeds
- Registry integration for seamless genetic algorithm compatibility
- Factory methods for creating enhanced seed instances
- Backward compatibility with existing seed selection logic

Usage:
    # Register all enhanced seeds
    from src.strategy.genetic_seeds.enhanced_seed_factory import register_all_enhanced_seeds
    register_all_enhanced_seeds()
    
    # Create enhanced seed instances
    enhanced_rsi = create_enhanced_seed("RSIFilterSeed", genes)
    enhanced_bb = create_enhanced_seed("BollingerBandsSeed", genes)
"""

import logging
from typing import Dict, List, Type, Optional, Any
import importlib
import inspect

from .universal_correlation_enhancer import UniversalCorrelationEnhancer, enhance_seed_with_correlation
from .base_seed import BaseSeed, SeedGenes, SeedType
from .seed_registry import get_registry, genetic_seed
from src.config.settings import get_settings, Settings

logger = logging.getLogger(__name__)


# Registry of all base seed classes and their enhanced counterparts
ENHANCED_SEED_REGISTRY: Dict[str, Type[BaseSeed]] = {}


def discover_all_genetic_seeds() -> Dict[str, Type[BaseSeed]]:
    """
    Discover all genetic seed classes in the genetic_seeds package.
    
    Returns:
        Dictionary mapping seed class names to their classes
    """
    discovered_seeds = {}
    
    # List of known seed modules (can be extended)
    seed_modules = [
        'ema_crossover_seed',
        'rsi_filter_seed', 
        'atr_stop_loss_seed',
        'volatility_scaling_seed',
        'pca_tree_quantile_seed',
        'linear_svc_classifier_seed',
        'bollinger_bands_seed',
        'donchian_breakout_seed',
        'stochastic_oscillator_seed',
        'ichimoku_cloud_seed',
        'nadaraya_watson_seed',
        'sma_trend_filter_seed',
        'vwap_reversion_seed',
        'funding_rate_carry_seed'
    ]
    
    for module_name in seed_modules:
        try:
            # Import the module
            module = importlib.import_module(f'.{module_name}', 'src.strategy.genetic_seeds')
            
            # Find seed classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's a BaseSeed subclass (but not BaseSeed itself)
                if (issubclass(obj, BaseSeed) and 
                    obj != BaseSeed and 
                    obj.__module__ == module.__name__ and
                    not name.startswith('Correlation') and  # Skip already enhanced seeds
                    not name.startswith('Universal')):     # Skip universal enhancer
                    
                    discovered_seeds[name] = obj
                    logger.debug(f"Discovered genetic seed: {name}")
                    
        except ImportError as e:
            logger.warning(f"Could not import seed module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error discovering seeds in {module_name}: {e}")
    
    logger.info(f"Discovered {len(discovered_seeds)} genetic seed classes")
    return discovered_seeds


def create_enhanced_seed_class(base_seed_class: Type[BaseSeed]) -> Type[BaseSeed]:
    """
    Create a correlation-enhanced version of a base seed class.
    
    Args:
        base_seed_class: The base seed class to enhance
        
    Returns:
        Enhanced seed class that wraps the base seed with correlation capabilities
    """
    
    class CorrelationEnhanced(BaseSeed):
        """Dynamically created correlation-enhanced seed class."""
        
        def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
            """Initialize enhanced seed by wrapping base seed with universal enhancer."""
            # Create base seed instance
            self._base_seed_instance = base_seed_class(genes, settings)
            
            # Wrap with universal correlation enhancer
            self._enhanced_seed = UniversalCorrelationEnhancer(self._base_seed_instance, settings)
            
            # Initialize as BaseSeed with enhanced genes
            super().__init__(self._enhanced_seed.genes, settings)
        
        @property
        def seed_name(self) -> str:
            return self._enhanced_seed.seed_name
        
        @property 
        def seed_description(self) -> str:
            return self._enhanced_seed.seed_description
        
        @property
        def required_parameters(self) -> List[str]:
            return self._enhanced_seed.required_parameters
        
        @property
        def parameter_bounds(self) -> Dict[str, tuple]:
            return self._enhanced_seed.parameter_bounds
        
        def calculate_technical_indicators(self, data) -> Dict[str, Any]:
            return self._enhanced_seed.calculate_technical_indicators(data)
        
        def generate_signals(self, data, filtered_assets=None, current_asset=None, timeframe='1h'):
            return self._enhanced_seed.generate_signals(data, filtered_assets, current_asset, timeframe)
        
        def clone_with_mutations(self, mutations: Dict[str, Any]):
            # Create new instance with mutations applied to base seed
            mutated_base = self._base_seed_instance.clone_with_mutations(mutations)
            return create_enhanced_seed_instance(base_seed_class.__name__, mutated_base.genes, self.settings)
        
        def should_enter_position(self, data, signal):
            return self._enhanced_seed.should_enter_position(data, signal)
        
        def calculate_stop_loss_level(self, data, entry_price, position_direction):
            return self._enhanced_seed.calculate_stop_loss_level(data, entry_price, position_direction)
        
        def calculate_position_size(self, data, signal):
            return self._enhanced_seed.calculate_position_size(data, signal)
        
        def get_exit_conditions(self, data, position_direction):
            return self._enhanced_seed.get_exit_conditions(data, position_direction)
        
        def __str__(self) -> str:
            return str(self._enhanced_seed)
        
        def __repr__(self) -> str:
            return f"Enhanced({base_seed_class.__name__})"
    
    # Set class attributes for registry compatibility
    CorrelationEnhanced.__name__ = f"Correlation_Enhanced_{base_seed_class.__name__}"
    CorrelationEnhanced.__qualname__ = f"Correlation_Enhanced_{base_seed_class.__name__}"
    CorrelationEnhanced.__doc__ = f"Correlation-enhanced version of {base_seed_class.__name__}"
    
    return CorrelationEnhanced


def register_enhanced_seed(base_seed_class: Type[BaseSeed]) -> Type[BaseSeed]:
    """
    Register a correlation-enhanced version of a base seed class.
    
    Args:
        base_seed_class: Base seed class to enhance and register
        
    Returns:
        Enhanced seed class
    """
    enhanced_class = create_enhanced_seed_class(base_seed_class)
    enhanced_name = f"Correlation_Enhanced_{base_seed_class.__name__}"
    
    # Store in our registry
    ENHANCED_SEED_REGISTRY[enhanced_name] = enhanced_class
    ENHANCED_SEED_REGISTRY[base_seed_class.__name__ + "_Enhanced"] = enhanced_class  # Alternative naming
    
    # Register with main seed registry using genetic_seed decorator
    try:
        registry = get_registry()
        registry.register_seed(enhanced_class)
        logger.info(f"Registered enhanced seed: {enhanced_name}")
        return enhanced_class
    except Exception as e:
        logger.error(f"Failed to register enhanced seed {enhanced_name}: {e}")
        return enhanced_class


def register_all_enhanced_seeds() -> int:
    """
    Discover and register correlation-enhanced versions of all genetic seeds.
    
    Returns:
        Number of enhanced seeds registered
    """
    logger.info("Starting automatic registration of all correlation-enhanced genetic seeds...")
    
    # Discover all base seed classes
    base_seeds = discover_all_genetic_seeds()
    
    registered_count = 0
    failed_count = 0
    
    for seed_name, seed_class in base_seeds.items():
        try:
            # Skip if already enhanced
            if 'Correlation' in seed_name or 'Enhanced' in seed_name:
                continue
                
            # Register enhanced version
            enhanced_class = register_enhanced_seed(seed_class)
            
            if enhanced_class:
                registered_count += 1
                logger.debug(f"✅ Registered: {seed_name} -> Correlation_Enhanced_{seed_name}")
            else:
                failed_count += 1
                logger.warning(f"❌ Failed: {seed_name}")
                
        except Exception as e:
            failed_count += 1
            logger.error(f"❌ Error registering {seed_name}: {e}")
    
    logger.info(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                ENHANCED SEED REGISTRATION COMPLETE           ║
    ║                                                              ║
    ║  ✅ Successfully Registered: {registered_count:2d} enhanced seeds               ║
    ║  ❌ Registration Failures:   {failed_count:2d} seeds                      ║
    ║  📊 Total Discovery:         {len(base_seeds):2d} base seeds                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    return registered_count


def create_enhanced_seed_instance(
    base_seed_name: str, 
    genes: SeedGenes, 
    settings: Optional[Settings] = None
) -> Optional[BaseSeed]:
    """
    Factory method to create an enhanced seed instance.
    
    Args:
        base_seed_name: Name of the base seed class
        genes: Genetic parameters
        settings: Optional settings
        
    Returns:
        Enhanced seed instance or None if creation fails
    """
    enhanced_name = f"Correlation_Enhanced_{base_seed_name}"
    
    if enhanced_name in ENHANCED_SEED_REGISTRY:
        try:
            enhanced_class = ENHANCED_SEED_REGISTRY[enhanced_name]
            return enhanced_class(genes, settings)
        except Exception as e:
            logger.error(f"Failed to create enhanced seed instance {enhanced_name}: {e}")
            return None
    
    # Fallback: try to create using universal enhancer directly
    try:
        registry = get_registry()
        base_class = registry.get_seed_class(base_seed_name)
        
        if base_class:
            base_instance = base_class(genes, settings)
            return UniversalCorrelationEnhancer(base_instance, settings)
        else:
            logger.error(f"Base seed class {base_seed_name} not found in registry")
            return None
            
    except Exception as e:
        logger.error(f"Fallback creation failed for {base_seed_name}: {e}")
        return None


def get_all_enhanced_seed_names() -> List[str]:
    """
    Get names of all registered enhanced seeds.
    
    Returns:
        List of enhanced seed names
    """
    return list(ENHANCED_SEED_REGISTRY.keys())


def is_seed_enhanced(seed_name: str) -> bool:
    """
    Check if a seed has a correlation-enhanced version available.
    
    Args:
        seed_name: Name of the seed to check
        
    Returns:
        True if enhanced version exists
    """
    enhanced_name = f"Correlation_Enhanced_{seed_name}"
    return enhanced_name in ENHANCED_SEED_REGISTRY


def get_enhancement_statistics() -> Dict[str, Any]:
    """
    Get statistics about enhanced seed registration.
    
    Returns:
        Dictionary with enhancement statistics
    """
    registry = get_registry()
    all_seeds = registry.get_all_seed_names()
    
    enhanced_seeds = [name for name in all_seeds if 'Correlation_Enhanced' in name]
    base_seeds = [name for name in all_seeds if 'Correlation_Enhanced' not in name]
    
    return {
        'total_seeds': len(all_seeds),
        'enhanced_seeds': len(enhanced_seeds),
        'base_seeds': len(base_seeds),
        'enhancement_coverage': len(enhanced_seeds) / max(len(base_seeds), 1),
        'enhanced_seed_names': enhanced_seeds,
        'factory_registry_size': len(ENHANCED_SEED_REGISTRY)
    }


# Auto-registration hook for when module is imported
def _auto_register_on_import():
    """Automatically register enhanced seeds when module is imported."""
    try:
        settings = get_settings()
        if getattr(settings, 'auto_register_enhanced_seeds', True):
            register_all_enhanced_seeds()
    except Exception as e:
        logger.warning(f"Auto-registration failed: {e}")


# Execute auto-registration
if __name__ != "__main__":  # Only when imported, not when run directly
    _auto_register_on_import()


# CLI interface for manual registration
if __name__ == "__main__":
    print("🚀 Manually registering all correlation-enhanced genetic seeds...")
    count = register_all_enhanced_seeds()
    stats = get_enhancement_statistics()
    print(f"\n📊 Enhancement Statistics:")
    for key, value in stats.items():
        if key != 'enhanced_seed_names':  # Skip long list
            print(f"   {key}: {value}")
    print(f"\n✅ Registration complete! {count} enhanced seeds ready for genetic evolution.")