"""
Quantitative Trading System - Core Package

This package provides a comprehensive quantitative trading system with:
- Asset discovery and filtering (discovery/)
- Genetic algorithm strategy evolution (strategy/)  
- VectorBT backtesting engine (backtesting/)
- Real-time execution and monitoring (execution/)
- Data management and APIs (data/)
- System configuration (config/)
- Utilities and compatibility layers (utils/)

Architecture follows 4-phase pipeline:
Phase 1: Asset Discovery → Phase 2: Strategy Evolution → Phase 3: Backtesting → Phase 4: Execution
"""

__version__ = "2.0.0"
__author__ = "Quantitative Trading System"

# Core module imports for convenience
from . import config
from . import data
from . import discovery
from . import strategy
from . import backtesting  
from . import execution
from . import utils

__all__ = [
    'config',
    'data', 
    'discovery',
    'strategy',
    'backtesting',
    'execution',
    'utils'
]