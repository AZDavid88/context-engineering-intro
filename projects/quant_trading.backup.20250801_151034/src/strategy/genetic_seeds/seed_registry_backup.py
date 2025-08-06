"""
Genetic Seed Registry System

This module provides a centralized registry for all genetic seeds, enabling
dynamic seed discovery, validation, and management for the genetic algorithm.

Key Features:
- Central seed registration and discovery
- Seed validation and compatibility checking
- Dynamic seed instantiation
- Population management utilities
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
    """Central registry for genetic trading seeds."""
    
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
        
        # Setup default validators
        self._setup_default_validators()
    
    def _setup_default_validators(self) -> None:
        """Set up default seed validation functions."""
        
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
            """Validate signal generation with synthetic data."""
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
                for param, (min_val, max_val) in seed_instance.parameter_bounds.items():
                    test_genes.parameters[param] = (min_val + max_val) / 2
                
                # Re-create with parameters
                seed_instance = seed_class(test_genes)
                
                # Test signal generation
                signals = seed_instance.generate_signals(synthetic_data)
                
                # Validate signal format
                if not isinstance(signals, pd.Series):
                    errors.append("generate_signals must return a pandas Series")
                elif len(signals) != len(synthetic_data):
                    errors.append("Signal length must match input data length")
                elif not signals.dtype in [np.int64, np.float64]:
                    errors.append("Signals must be numeric (int or float)")
                elif signals.isna().any():
                    errors.append("Signals cannot contain NaN values")
                
                # Test technical indicators
                indicators = seed_instance.calculate_technical_indicators(synthetic_data)
                if not isinstance(indicators, dict):
                    errors.append("calculate_technical_indicators must return a dictionary")
                
            except Exception as e:
                errors.append(f"Error in signal generation validation: {str(e)}")
            
            return errors
        
        # Register validators
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
            force_reregister: Whether to force re-registration of existing seeds
            
        Returns:
            Registration status
        """
        try:
            # Get seed name
            dummy_genes = SeedGenes(
                seed_id="dummy",
                seed_type=SeedType.MOMENTUM,
                parameters={}
            )
            dummy_instance = seed_class(dummy_genes)
            seed_name = dummy_instance.seed_name
            seed_type = dummy_instance.genes.seed_type
            
            # Check for duplicates
            if seed_name in self._registry and not force_reregister:
                self.logger.warning(f"Seed '{seed_name}' already registered")
                return RegistrationStatus.DUPLICATE
            
            # Validate seed
            validation_errors = self._validate_seed(seed_class)
            if validation_errors:
                self.logger.error(f"Validation failed for seed '{seed_name}': {validation_errors}")
                status = RegistrationStatus.VALIDATION_FAILED
                self._validation_failures += 1
            else:
                status = RegistrationStatus.REGISTERED
                self._registration_count += 1
                
                # Add to type index
                if seed_name not in self._type_index[seed_type]:
                    self._type_index[seed_type].append(seed_name)
            
            # Create registration record
            import time
            registration = SeedRegistration(
                seed_class=seed_class,
                seed_name=seed_name,
                seed_type=seed_type,
                status=status,
                validation_errors=validation_errors,
                registration_timestamp=time.time()
            )
            
            self._registry[seed_name] = registration
            
            self.logger.info(f"Seed '{seed_name}' registration: {status.value}")
            return status
            
        except Exception as e:
            self.logger.error(f"Error registering seed: {e}")
            return RegistrationStatus.INCOMPATIBLE
    
    def _validate_seed(self, seed_class: Type[BaseSeed]) -> List[str]:
        """Validate a seed class using all registered validators.
        
        Args:
            seed_class: Seed class to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        all_errors = []
        
        for validator in self._validation_functions:
            try:
                errors = validator(seed_class)
                all_errors.extend(errors)
            except Exception as e:
                all_errors.append(f"Validator error: {str(e)}")
        
        return all_errors
    
    def get_seed_class(self, seed_name: str) -> Optional[Type[BaseSeed]]:
        """Get seed class by name.
        
        Args:
            seed_name: Name of the seed
            
        Returns:
            Seed class if found and valid, None otherwise
        """
        if seed_name not in self._registry:
            return None
        
        registration = self._registry[seed_name]
        if registration.status == RegistrationStatus.REGISTERED:
            return registration.seed_class
        
        return None
    
    def get_seeds_by_type(self, seed_type: SeedType) -> List[Type[BaseSeed]]:
        """Get all registered seeds of a specific type.
        
        Args:
            seed_type: Type of seeds to retrieve
            
        Returns:
            List of seed classes
        """
        seed_classes = []
        for seed_name in self._type_index[seed_type]:
            seed_class = self.get_seed_class(seed_name)
            if seed_class:
                seed_classes.append(seed_class)
        
        return seed_classes
    
    def list_all_seeds(self) -> Dict[str, Dict[str, Any]]:
        """List all registered seeds with their information.
        
        Returns:
            Dictionary mapping seed names to their information
        """
        seed_info = {}
        
        for seed_name, registration in self._registry.items():
            if registration.status == RegistrationStatus.REGISTERED:
                # Create dummy instance to get description
                try:
                    dummy_genes = SeedGenes(
                        seed_id="info",
                        seed_type=registration.seed_type,
                        parameters={}
                    )
                    dummy_instance = registration.seed_class(dummy_genes)
                    
                    seed_info[seed_name] = {
                        'type': registration.seed_type.value,
                        'description': dummy_instance.seed_description,
                        'required_parameters': dummy_instance.required_parameters,
                        'parameter_bounds': dummy_instance.parameter_bounds,
                        'registration_time': registration.registration_timestamp
                    }
                except Exception as e:
                    seed_info[seed_name] = {
                        'type': registration.seed_type.value,
                        'description': f"Error loading: {e}",
                        'registration_time': registration.registration_timestamp
                    }
        
        return seed_info
    
    def create_seed_instance(self, seed_name: str, genes: SeedGenes) -> Optional[BaseSeed]:
        """Create an instance of a registered seed.
        
        Args:
            seed_name: Name of the seed to instantiate
            genes: Genetic parameters for the seed
            
        Returns:
            Seed instance if successful, None otherwise
        """
        seed_class = self.get_seed_class(seed_name)
        if not seed_class:
            self.logger.error(f"Seed '{seed_name}' not found in registry")
            return None
        
        try:
            return seed_class(genes, self.settings)
        except Exception as e:
            self.logger.error(f"Error creating seed instance '{seed_name}': {e}")
            return None
    
    def create_random_population(self, population_size: int = 100) -> List[BaseSeed]:
        """Create a random population of seeds for genetic algorithm.
        
        Args:
            population_size: Number of seeds to create
            
        Returns:
            List of random seed instances
        """
        import random
        
        population = []
        registered_seeds = [name for name, reg in self._registry.items() 
                          if reg.status == RegistrationStatus.REGISTERED]
        
        if not registered_seeds:
            self.logger.error("No registered seeds available for population creation")
            return population
        
        for i in range(population_size):
            # Select random seed type
            seed_name = random.choice(registered_seeds)
            seed_class = self.get_seed_class(seed_name)
            
            if seed_class:
                # Create dummy instance to get parameter bounds
                dummy_genes = SeedGenes(
                    seed_id=f"pop_{i}",
                    seed_type=SeedType.MOMENTUM,  # Will be updated
                    generation=0,
                    parameters={}
                )
                
                try:
                    dummy_instance = seed_class(dummy_genes)
                    dummy_genes.seed_type = dummy_instance.genes.seed_type
                    
                    # Generate random parameters within bounds
                    for param, (min_val, max_val) in dummy_instance.parameter_bounds.items():
                        dummy_genes.parameters[param] = random.uniform(min_val, max_val)
                    
                    # Create final instance
                    seed_instance = seed_class(dummy_genes, self.settings)
                    population.append(seed_instance)
                    
                except Exception as e:
                    self.logger.error(f"Error creating random seed '{seed_name}': {e}")
        
        self.logger.info(f"Created random population of {len(population)} seeds")
        return population
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        by_type = {}
        for seed_type in SeedType:
            by_type[seed_type.value] = len(self._type_index[seed_type])
        
        return {
            'total_registered': self._registration_count,
            'validation_failures': self._validation_failures,
            'by_type': by_type,
            'total_in_registry': len(self._registry)
        }
    
    def add_validator(self, validator_func: Callable[[Type[BaseSeed]], List[str]]) -> None:
        """Add a custom validation function.
        
        Args:
            validator_func: Function that takes a seed class and returns list of errors
        """
        self._validation_functions.append(validator_func)
        self.logger.info("Added custom validator to registry")


# Global registry instance
_global_registry: Optional[SeedRegistry] = None


def get_registry() -> SeedRegistry:
    """Get the global seed registry instance."""
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