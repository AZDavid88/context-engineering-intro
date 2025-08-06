"""
Trading Execution Package

Handles live trading execution and monitoring:
- Genetic strategy pool management with Ray integration
- Order management and execution
- Position sizing and risk management  
- Paper trading simulation
- Real-time monitoring and alerting
- Trading system session management
"""

from .genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
from .trading_system_manager import TradingSystemManager
from .order_management import OrderManager
from .position_sizer import GeneticPositionSizer
from .risk_management import GeneticRiskManager
from .paper_trading import PaperTradingEngine
from .monitoring import RealTimeMonitoringSystem

__all__ = [
    'GeneticStrategyPool',
    'EvolutionConfig',
    'TradingSystemManager',
    'OrderManager',
    'GeneticPositionSizer', 
    'GeneticRiskManager',
    'PaperTradingEngine',
    'RealTimeMonitoringSystem'
]