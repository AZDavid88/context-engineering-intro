"""
Genetic Seed Registry System - FIXED FOR MULTIPROCESSING

This module provides a centralized registry for all genetic seeds, enabling
dynamic seed discovery, validation, and management for the genetic algorithm.

CRITICAL FIX: All validator functions are now module-level (picklable) instead of closures.

Key Features:
- Central seed registration and discovery
- Seed validation and compatibility checking
- Dynamic seed instantiation
- Population management utilities
- MULTIPROCESSING COMPATIBLE (no local closures)
"""

from typing import Dict, List, Type, Optional, Any, Callable
from collections import defaultdict
import logging
from dataclasses import dataclass
from enum import Enum

from .base_seed import BaseSeed, SeedType, SeedGenes
from src.config.settings import get_settings, Settings


# Module-level validator functions (picklable for multiprocessing)
def validate_base_interface(seed_class: Type[BaseSeed]) -> List[str]:
    """Validate that seed implements BaseSeed interface correctly."""
    errors = []
    
    # Check required methods
    required_methods = [
        'generate_signals', 'calculate_technical_indicators',
        'seed_name', 'seed_description', 'required_parameters', 'parameter_bounds'
    ]
    
    for method in required_methods:
        if not hasattr(seed_class, method):
            errors.append(f"Missing required method: {method}")
        elif method in ['seed_name', 'seed_description', 'required_parameters', 'parameter_bounds']:
            # Check if it's a property
            if not isinstance(getattr(seed_class, method), property):
                errors.append(f"Method {method} should be a property")
    
    return errors


def validate_parameter_bounds(seed_class: Type[BaseSeed]) -> List[str]:
    """Validate parameter bounds are reasonable."""
    errors = []
    
    try:
        # Create dummy instance to test parameter bounds
        dummy_genes = SeedGenes(
            seed_id="test",
            seed_type=SeedType.MOMENTUM,
            parameters={}
        )
        dummy_seed = seed_class(dummy_genes)
        
        bounds = dummy_seed.parameter_bounds
        required_params = dummy_seed.required_parameters
        
        # Check that all required parameters have bounds
        for param in required_params:
            if param not in bounds:
                errors.append(f"Required parameter '{param}' missing from parameter_bounds")
        
        # Check bounds validity
        for param, (min_val, max_val) in bounds.items():
            if min_val >= max_val:
                errors.append(f"Invalid bounds for '{param}': min ({min_val}) >= max ({max_val})")
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                errors.append(f"Bounds for '{param}' must be numeric")
                
    except Exception as e:
        errors.append(f"Error validating parameter bounds: {str(e)}")
    
    return errors


def validate_signal_generation(seed_class: Type[BaseSeed]) -> List[str]:
    """Validate seed can generate signals with synthetic data."""
    errors = []
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create synthetic OHLCV data
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        synthetic_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Create test seed
        test_genes = SeedGenes(
            seed_id="validation_test",
            seed_type=SeedType.MOMENTUM,
            parameters={}
        )
        
        # Add default parameters for testing
        seed_instance = seed_class(test_genes)
        
        # Test signal generation
        signals = seed_instance.generate_signals(synthetic_data)
        
        if signals is None:
            errors.append("generate_signals returned None")
        elif not isinstance(signals, pd.Series):
            errors.append("generate_signals must return a pandas Series")
        elif len(signals) == 0:
            errors.append("generate_signals returned empty Series")
            
    except Exception as e:
        errors.append(f"Error in signal generation validation: {str(e)}")
    
    return errors


class RegistrationStatus(str, Enum):
    """Seed registration status."""
    REGISTERED = "registered"
    VALIDATION_FAILED = "validation_failed"
    DUPLICATE = "duplicate"
    INCOMPATIBLE = "incompatible"


@dataclass
class SeedRegistration:
    """Seed registration information."""
    seed_class: Type[BaseSeed]
    seed_name: str
    seed_type: SeedType
    status: RegistrationStatus
    validation_errors: List[str]
    registration_timestamp: float


class SeedRegistry:
    """Central registry for genetic trading seeds - MULTIPROCESSING COMPATIBLE."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize seed registry.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.Registry")
        
        # Registry storage
        self._registry: Dict[str, SeedRegistration] = {}
        self._type_index: Dict[SeedType, List[str]] = defaultdict(list)
        self._validation_functions: List[Callable[[Type[BaseSeed]], List[str]]] = []
        
        # Registration statistics
        self._registration_count = 0
        self._validation_failures = 0
        
        # Setup default validators (now using module-level functions)
        self._setup_default_validators()
    
    def _setup_default_validators(self) -> None:
        """Set up default seed validation functions using module-level functions."""
        # Register module-level validators (picklable for multiprocessing)
        self._validation_functions.extend([
            validate_base_interface,
            validate_parameter_bounds,
            validate_signal_generation
        ])
    
    def register_seed(self, seed_class: Type[BaseSeed], 
                     force_reregister: bool = False) -> RegistrationStatus:
        """Register a genetic seed class.
        
        Args:
            seed_class: Seed class to register
            force_reregister: Force re-registration if already exists
            
        Returns:
            Registration status
        """
        seed_name = seed_class.__name__
        
        # Check for duplicate registration
        if seed_name in self._registry and not force_reregister:
            self.logger.warning(f"Seed {seed_name} already registered")
            return RegistrationStatus.DUPLICATE
        
        # Run validation
        validation_errors = []
        for validator in self._validation_functions:
            try:
                errors = validator(seed_class)
                validation_errors.extend(errors)
            except Exception as e:
                validation_errors.append(f"Validator error: {str(e)}")
        
        # Determine registration status
        if validation_errors:
            status = RegistrationStatus.VALIDATION_FAILED
            self._validation_failures += 1
            self.logger.error(f"Seed {seed_name} validation failed: {validation_errors}")
        else:
            status = RegistrationStatus.REGISTERED
            self._registration_count += 1
            self.logger.info(f"Seed {seed_name} registered successfully")
        
        # Create registration record
        import time
        registration = SeedRegistration(
            seed_class=seed_class,
            seed_name=seed_name,
            seed_type=getattr(seed_class, '_seed_type', SeedType.MOMENTUM),
            status=status,
            validation_errors=validation_errors,
            registration_timestamp=time.time()
        )
        
        # Store registration
        self._registry[seed_name] = registration
        
        # Update type index
        if status == RegistrationStatus.REGISTERED:
            self._type_index[registration.seed_type].append(seed_name)
        
        return status
    
    def get_seed_class(self, seed_name: str) -> Optional[Type[BaseSeed]]:
        """Get seed class by name.
        
        Args:
            seed_name: Name of the seed
            
        Returns:
            Seed class or None if not found
        """
        registration = self._registry.get(seed_name)
        if registration and registration.status == RegistrationStatus.REGISTERED:
            return registration.seed_class
        return None
    
    def get_all_seed_names(self) -> List[str]:
        """Get all registered seed names.
        
        Returns:
            List of seed names
        """
        return [
            name for name, reg in self._registry.items()
            if reg.status == RegistrationStatus.REGISTERED
        ]
    
    def get_all_seed_classes(self) -> List[Type[BaseSeed]]:
        """Get all registered seed classes.
        
        Returns:
            List of seed classes
        """
        return [
            reg.seed_class for reg in self._registry.values()
            if reg.status == RegistrationStatus.REGISTERED
        ]
    
    def get_seeds_by_type(self, seed_type: SeedType) -> List[Type[BaseSeed]]:
        """Get seeds by type.
        
        Args:
            seed_type: Type of seeds to retrieve
            
        Returns:
            List of seed classes
        """
        seed_names = self._type_index.get(seed_type, [])
        return [
            self._registry[name].seed_class
            for name in seed_names
            if self._registry[name].status == RegistrationStatus.REGISTERED
        ]
    
    def create_random_individual(self) -> Optional[Any]:
        """Create a random individual from registered seeds.
        
        Returns:
            Random seed instance or None
        """
        import random
        
        registered_classes = self.get_all_seed_classes()
        if not registered_classes:
            return None
        
        # Select random seed class
        seed_class = random.choice(registered_classes)
        
        # Create random genes
        genes = SeedGenes(
            seed_id=seed_class.__name__,
            seed_type=getattr(seed_class, '_seed_type', SeedType.MOMENTUM),
            parameters={}
        )
        
        # Create seed instance
        return seed_class(genes)
    
    def add_validator(self, validator_func: Callable[[Type[BaseSeed]], List[str]]) -> None:
        """Add custom validator function.
        
        Args:
            validator_func: Validator function
        """
        self._validation_functions.append(validator_func)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Registry statistics
        """
        return {
            "total_registrations": len(self._registry),
            "successful_registrations": self._registration_count,
            "validation_failures": self._validation_failures,
            "registered_seeds": self.get_all_seed_names(),
            "seeds_by_type": {
                seed_type.value: len(seed_names)
                for seed_type, seed_names in self._type_index.items()
            }
        }


# Global registry instance
_global_registry: Optional[SeedRegistry] = None


def get_registry() -> SeedRegistry:
    """Get global seed registry instance.
    
    Returns:
        Global seed registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SeedRegistry()
    return _global_registry


def register_seed(seed_class: Type[BaseSeed]) -> RegistrationStatus:
    """Register a seed class with the global registry.
    
    Args:
        seed_class: Seed class to register
        
    Returns:
        Registration status
    """
    return get_registry().register_seed(seed_class)


# Decorator for automatic registration
def genetic_seed(seed_class: Type[BaseSeed]) -> Type[BaseSeed]:
    """Decorator to automatically register a genetic seed.
    
    Usage:
        @genetic_seed
        class MyCustomSeed(BaseSeed):
            ...
    """
    register_seed(seed_class)
    return seed_class