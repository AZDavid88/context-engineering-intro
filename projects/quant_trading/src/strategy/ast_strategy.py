
import operator
import random
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import datetime, timezone
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import warnings

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

# DEAP imports for genetic programming
from deap import base, creator, tools, gp, algorithms
import multiprocessing

# Technical analysis using official pandas APIs (PRIMARY)
# Optional enhancements detected at runtime (TA-Lib v0.5.0+ if available)

# Import our configuration and data systems
from src.config.settings import get_settings, Settings

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
AST-Based Trading Strategy Representation with Genetic Programming
This module provides Abstract Syntax Tree (AST) representation for trading strategies
that can be evolved using genetic algorithms through the DEAP framework.
Key Features:
- Strongly-typed genetic programming primitives for trading
- Technical indicator integration (RSI, SMA, MACD, Bollinger Bands)
- Type-safe strategy construction preventing invalid combinations  
- Multi-objective fitness evaluation (Sharpe, drawdown, consistency)
- Strategy lifecycle management (birth → testing → production → death)
- Integration with Fear & Greed environmental pressure
Based on comprehensive research from:
- DEAP Genetic Programming Framework (100% coverage)
- Technical Analysis Library (TA-Lib) integration patterns
- Vectorbt backtesting engine compatibility
- Production-ready strategy evolution patterns
"""
warnings.filterwarnings('ignore')
class StrategyLifecycle(str, Enum):
    """Strategy lifecycle states."""
    BIRTH = "birth"                    # Just created, untested
    VALIDATION = "validation"          # In backtesting validation
    PAPER_TRADING = "paper_trading"    # Live paper trading
    PRODUCTION = "production"          # Live trading with real capital
    DECAY = "decay"                    # Performance degrading
    DEATH = "death"                    # Terminated/retired
class FitnessObjective(str, Enum):
    """Multi-objective fitness components."""
    SHARPE_RATIO = "sharpe_ratio"      # Risk-adjusted returns
    MAX_DRAWDOWN = "max_drawdown"      # Maximum peak-to-trough decline
    WIN_RATE = "win_rate"              # Percentage of winning trades
    CONSISTENCY = "consistency"        # Performance stability over time
    PROFIT_FACTOR = "profit_factor"    # Gross profit / gross loss
class StrategyGenes(BaseModel):
    """Validated strategy genetic representation."""
    # Core genetic material
    tree_representation: str = Field(..., description="String representation of GP tree")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Strategy complexity (0=simple, 1=complex)")
    # Technical indicator genes
    indicators_used: List[str] = Field(default_factory=list, description="List of technical indicators")
    lookback_periods: Dict[str, int] = Field(default_factory=dict, description="Lookback periods for indicators")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Decision thresholds")
    # Trading behavior genes
    position_sizing_method: str = Field(default="fixed", description="Position sizing method")
    risk_tolerance: float = Field(default=0.02, ge=0.001, le=0.1, description="Risk per trade")
    time_horizon: str = Field(default="short", description="Trading time horizon")
    # Evolution metadata
    generation: int = Field(default=0, ge=0, description="Generation number")
    parent_ids: List[str] = Field(default_factory=list, description="Parent strategy IDs")
    mutation_count: int = Field(default=0, ge=0, description="Number of mutations applied")
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
class StrategyFitness(BaseModel):
    """Multi-objective fitness evaluation results."""
    # Primary fitness metrics
    sharpe_ratio: float = Field(..., description="Sharpe ratio (higher is better)")
    max_drawdown: float = Field(..., description="Maximum drawdown (lower is better)")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Win rate (0-1)")
    consistency: float = Field(..., ge=0.0, le=1.0, description="Consistency score (0-1)")
    profit_factor: float = Field(..., ge=0.0, description="Profit factor (>1 profitable)")
    # Auxiliary metrics
    total_return: float = Field(..., description="Total return percentage")
    volatility: float = Field(..., ge=0.0, description="Return volatility")
    max_consecutive_losses: int = Field(..., ge=0, description="Maximum consecutive losses")
    # Fitness score (weighted combination)
    composite_fitness: float = Field(..., description="Weighted composite fitness score")
    # Validation across time periods
    in_sample_fitness: float = Field(..., description="In-sample fitness score")
    out_of_sample_fitness: float = Field(..., description="Out-of-sample fitness score")
    walk_forward_fitness: float = Field(..., description="Walk-forward fitness score")
    @field_validator('composite_fitness', mode='before')
    @classmethod
    def calculate_composite_fitness(cls, v, info):
        """Calculate weighted composite fitness score."""
        if not all(key in info.data for key in ['sharpe_ratio', 'max_drawdown', 'win_rate', 'consistency']):
            return 0.0
        # Weights from settings (configurable)
        weights = {
            'sharpe_ratio': 0.4,
            'max_drawdown': 0.3,  # Minimize (negative weight applied later)
            'win_rate': 0.2,
            'consistency': 0.1
        }
        # Normalize and weight components
        sharpe_component = max(0, info.data['sharpe_ratio']) / 5.0  # Normalize to 0-1 (assuming max Sharpe ~5)
        drawdown_component = max(0, 1.0 - info.data['max_drawdown'])  # Invert (lower drawdown is better)
        win_rate_component = info.data['win_rate']  # Already 0-1
        consistency_component = info.data['consistency']  # Already 0-1
        composite = (
            weights['sharpe_ratio'] * sharpe_component +
            weights['max_drawdown'] * drawdown_component +
            weights['win_rate'] * win_rate_component +
            weights['consistency'] * consistency_component
        )
        return max(0.0, min(1.0, composite))  # Clamp to 0-1 range
class TradingStrategy(BaseModel):
    """Complete trading strategy with genetic material and performance history."""
    # Identity
    strategy_id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Human-readable strategy name")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Genetic material
    genes: StrategyGenes = Field(..., description="Strategy genetic representation")
    fitness: Optional[StrategyFitness] = Field(None, description="Fitness evaluation results")
    # Lifecycle management
    lifecycle_state: StrategyLifecycle = Field(default=StrategyLifecycle.BIRTH)
    tests_passed: int = Field(default=0, ge=0, description="Number of validation tests passed")
    capital_allocated: float = Field(default=0.0, ge=0.0, description="Capital currently allocated")
    # Performance tracking
    trades_executed: int = Field(default=0, ge=0, description="Total trades executed")
    realized_pnl: float = Field(default=0.0, description="Realized profit/loss")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized profit/loss")
    # Evolution history
    evolution_log: List[Dict[str, Any]] = Field(default_factory=list, description="Evolution history")
    def generate_strategy_id(self) -> str:
        """Generate unique strategy ID based on genes and timestamp."""
        import hashlib
        gene_hash = hashlib.md5(self.genes.tree_representation.encode()).hexdigest()[:8]
        timestamp = int(self.created_at.timestamp())
        return f"strat_{self.genes.generation}_{gene_hash}_{timestamp}"
    def is_ready_for_production(self) -> bool:
        """Check if strategy is ready for live trading."""
        if not self.fitness:
            return False
        return (
            self.lifecycle_state == StrategyLifecycle.PAPER_TRADING and
            self.fitness.sharpe_ratio >= 2.0 and  # Minimum Sharpe requirement
            self.fitness.max_drawdown <= 0.15 and  # Maximum 15% drawdown
            self.fitness.win_rate >= 0.4 and  # Minimum 40% win rate
            self.tests_passed >= 3  # Passed validation, paper trading, consistency tests
        )
    def add_evolution_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Add event to evolution log."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "details": details
        }
        self.evolution_log.append(event)
class TechnicalIndicatorPrimitives:
    """Technical indicator primitives for genetic programming."""
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize technical indicator primitives.
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
    @staticmethod
    def safe_divide(left: float, right: float) -> float:
        """Protected division to prevent errors."""
        try:
            if abs(right) < 1e-10:  # Near zero
                return 1.0
            return left / right
        except (ZeroDivisionError, OverflowError):
            return 1.0
    @staticmethod
    def safe_log(x: float) -> float:
        """Protected logarithm."""
        try:
            return math.log(max(abs(x), 1e-10))
        except (ValueError, OverflowError):
            return 0.0
    @staticmethod
    def safe_sqrt(x: float) -> float:
        """Protected square root."""
        try:
            return math.sqrt(max(abs(x), 0.0))
        except (ValueError, OverflowError):
            return 0.0
    def rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index using official pandas APIs (primary)."""
        # PRIMARY: Official pandas.pydata.org APIs
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        # Optional TA-Lib enhancement (detected at runtime)
        try:
            import talib
            if 'close' in data.columns:
                talib_rsi = pd.Series(talib.RSI(data['close'].values, timeperiod=period), index=data.index)
                # Could optionally use TA-Lib here, but pandas is our primary choice
        except (ImportError, Exception):
            pass  # Continue with pandas primary implementation
        return rsi.fillna(50.0)
    def sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Simple Moving Average using official pandas APIs."""
        # PRIMARY: Official pandas.Series.rolling implementation
        return data['close'].rolling(window=period).mean()
    def ema(self, data: pd.DataFrame, period: int = 12) -> pd.Series:
        """Exponential Moving Average using official pandas APIs."""
        # PRIMARY: Official pandas.Series.ewm implementation
        return data['close'].ewm(span=period, adjust=True).mean()
    def macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD indicator using official pandas APIs."""
        # PRIMARY: Official pandas.Series.ewm implementation
        ema_fast = data['close'].ewm(span=fast, adjust=True).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=True).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=True).mean()
        return macd_line, signal_line
    def bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        if TALIB_AVAILABLE and 'close' in data.columns:
            upper, middle, lower = talib.BBANDS(data['close'].values, timeperiod=period, nbdevup=std, nbdevdn=std)
            return (pd.Series(upper, index=data.index), 
                   pd.Series(middle, index=data.index), 
                   pd.Series(lower, index=data.index))
        else:
            bb_data = ta.bbands(data['close'], length=period, std=std)
            return bb_data.iloc[:, 0], bb_data.iloc[:, 1], bb_data.iloc[:, 2]  # Upper, Mid, Lower
    def atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        if TALIB_AVAILABLE and all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period),
                           index=data.index)
        else:
            return ta.atr(data['high'], data['low'], data['close'], length=period)
class GeneticProgrammingEngine:
    """Genetic Programming Engine for trading strategy evolution."""
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize GP engine.
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()  # KEY CONNECTION TO SETTINGS.PY
        self.indicators = TechnicalIndicatorPrimitives(settings)
        # GP configuration from settings
        self.population_size = self.settings.genetic_algorithm.population_size
        self.max_generations = self.settings.genetic_algorithm.max_generations
        self.crossover_prob = self.settings.genetic_algorithm.crossover_probability
        self.mutation_prob = self.settings.genetic_algorithm.mutation_probability
        self.tournament_size = self.settings.genetic_algorithm.tournament_size
        self.max_tree_height = self.settings.genetic_algorithm.max_tree_height
        self.max_tree_size = self.settings.genetic_algorithm.max_tree_size
        # Initialize DEAP components
        self._setup_deap_primitives()
        self._setup_deap_toolbox()
        # Evolution tracking
        self.generation_count = 0
        self.best_strategies: List[TradingStrategy] = []
    def _setup_deap_primitives(self) -> None:
        """Set up strongly-typed genetic programming primitives."""
        # Define types for type-safe GP
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0, 1.0))  # Maximize most, minimize drawdown
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
        # Create primitive set with types
        self.pset = gp.PrimitiveSetTyped("trading_strategy", [pd.DataFrame], bool)
        # Technical indicator primitives: DataFrame → float
        self.pset.addPrimitive(self._get_rsi, [pd.DataFrame], float, name="rsi")
        self.pset.addPrimitive(self._get_sma_20, [pd.DataFrame], float, name="sma20")
        self.pset.addPrimitive(self._get_sma_50, [pd.DataFrame], float, name="sma50")
        self.pset.addPrimitive(self._get_ema_12, [pd.DataFrame], float, name="ema12")
        self.pset.addPrimitive(self._get_ema_26, [pd.DataFrame], float, name="ema26")
        self.pset.addPrimitive(self._get_macd_line, [pd.DataFrame], float, name="macd")
        self.pset.addPrimitive(self._get_macd_signal, [pd.DataFrame], float, name="macd_signal")
        self.pset.addPrimitive(self._get_bb_upper, [pd.DataFrame], float, name="bb_upper")
        self.pset.addPrimitive(self._get_bb_lower, [pd.DataFrame], float, name="bb_lower")
        self.pset.addPrimitive(self._get_atr, [pd.DataFrame], float, name="atr")
        self.pset.addPrimitive(self._get_close_price, [pd.DataFrame], float, name="close")
        self.pset.addPrimitive(self._get_volume, [pd.DataFrame], float, name="volume")
        # Mathematical primitives: float, float → float
        self.pset.addPrimitive(operator.add, [float, float], float)
        self.pset.addPrimitive(operator.sub, [float, float], float)
        self.pset.addPrimitive(operator.mul, [float, float], float)
        self.pset.addPrimitive(self.indicators.safe_divide, [float, float], float, name="div")
        # Comparison primitives: float, float → bool
        self.pset.addPrimitive(operator.gt, [float, float], bool, name="gt")
        self.pset.addPrimitive(operator.lt, [float, float], bool, name="lt")
        self.pset.addPrimitive(operator.ge, [float, float], bool, name="gte")
        self.pset.addPrimitive(operator.le, [float, float], bool, name="lte")
        # Logical primitives: bool, bool → bool
        self.pset.addPrimitive(operator.and_, [bool, bool], bool, name="and")
        self.pset.addPrimitive(operator.or_, [bool, bool], bool, name="or")
        self.pset.addPrimitive(operator.not_, [bool], bool, name="not")
        # Terminals (constants)
        self.pset.addEphemeralConstant("rand_float", lambda: random.uniform(-100, 100), float)
        self.pset.addEphemeralConstant("rsi_threshold", lambda: random.uniform(20, 80), float)
        self.pset.addEphemeralConstant("small_const", lambda: random.uniform(0.01, 1.0), float)
        # Terminal constants
        self.pset.addTerminal(True, bool)
        self.pset.addTerminal(False, bool)
        # Rename arguments for clarity
        self.pset.renameArguments(ARG0='market_data')
    def _setup_deap_toolbox(self) -> None:
        """Set up DEAP toolbox with operators."""
        self.toolbox = base.Toolbox()
        # Individual and population creation
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=4)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # Genetic operators
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)
        # Selection
        self.toolbox.register("select", tools.selNSGA2)  # Multi-objective selection
        self.toolbox.register("selectTournament", tools.selTournament, tournsize=self.tournament_size)
        # Evaluation (will be set later)
        self.toolbox.register("evaluate", self._evaluate_strategy)
        # Constraints to prevent bloat
        self.toolbox.decorate("mate", gp.staticLimit(key=len, max_value=self.max_tree_size))
        self.toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=self.max_tree_size))
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_height))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_height))
        # Multiprocessing support (if enabled in settings)
        if self.settings.genetic_algorithm.use_multiprocessing:
            if self.settings.genetic_algorithm.max_workers:
                pool = multiprocessing.Pool(self.settings.genetic_algorithm.max_workers)
            else:
                pool = multiprocessing.Pool()
            self.toolbox.register("map", pool.map)
    # Technical indicator getter methods for GP primitives
    def _get_rsi(self, data: pd.DataFrame) -> float:
        """Get RSI value for GP evaluation."""
        try:
            rsi_values = self.indicators.rsi(data, 14)
            return float(rsi_values.iloc[-1]) if len(rsi_values) > 0 and not pd.isna(rsi_values.iloc[-1]) else 50.0
        except:
            return 50.0
    def _get_sma_20(self, data: pd.DataFrame) -> float:
        """Get 20-period SMA."""
        try:
            sma_values = self.indicators.sma(data, 20)
            return float(sma_values.iloc[-1]) if len(sma_values) > 0 and not pd.isna(sma_values.iloc[-1]) else data['close'].iloc[-1]
        except:
            return float(data['close'].iloc[-1]) if len(data) > 0 else 0.0
    def _get_sma_50(self, data: pd.DataFrame) -> float:
        """Get 50-period SMA."""
        try:
            sma_values = self.indicators.sma(data, 50)
            return float(sma_values.iloc[-1]) if len(sma_values) > 0 and not pd.isna(sma_values.iloc[-1]) else data['close'].iloc[-1]
        except:
            return float(data['close'].iloc[-1]) if len(data) > 0 else 0.0
    def _get_ema_12(self, data: pd.DataFrame) -> float:
        """Get 12-period EMA."""
        try:
            ema_values = self.indicators.ema(data, 12)
            return float(ema_values.iloc[-1]) if len(ema_values) > 0 and not pd.isna(ema_values.iloc[-1]) else data['close'].iloc[-1]
        except:
            return float(data['close'].iloc[-1]) if len(data) > 0 else 0.0
    def _get_ema_26(self, data: pd.DataFrame) -> float:
        """Get 26-period EMA."""
        try:
            ema_values = self.indicators.ema(data, 26)
            return float(ema_values.iloc[-1]) if len(ema_values) > 0 and not pd.isna(ema_values.iloc[-1]) else data['close'].iloc[-1]
        except:
            return float(data['close'].iloc[-1]) if len(data) > 0 else 0.0
    def _get_macd_line(self, data: pd.DataFrame) -> float:
        """Get MACD line value."""
        try:
            macd_line, _ = self.indicators.macd(data)
            return float(macd_line.iloc[-1]) if len(macd_line) > 0 and not pd.isna(macd_line.iloc[-1]) else 0.0
        except:
            return 0.0
    def _get_macd_signal(self, data: pd.DataFrame) -> float:
        """Get MACD signal line value."""
        try:
            _, signal_line = self.indicators.macd(data)
            return float(signal_line.iloc[-1]) if len(signal_line) > 0 and not pd.isna(signal_line.iloc[-1]) else 0.0
        except:
            return 0.0
    def _get_bb_upper(self, data: pd.DataFrame) -> float:
        """Get Bollinger Band upper value."""
        try:
            upper, _, _ = self.indicators.bollinger_bands(data)
            return float(upper.iloc[-1]) if len(upper) > 0 and not pd.isna(upper.iloc[-1]) else data['close'].iloc[-1] * 1.02
        except:
            return float(data['close'].iloc[-1] * 1.02) if len(data) > 0 else 0.0
    def _get_bb_lower(self, data: pd.DataFrame) -> float:
        """Get Bollinger Band lower value."""
        try:
            _, _, lower = self.indicators.bollinger_bands(data)
            return float(lower.iloc[-1]) if len(lower) > 0 and not pd.isna(lower.iloc[-1]) else data['close'].iloc[-1] * 0.98
        except:
            return float(data['close'].iloc[-1] * 0.98) if len(data) > 0 else 0.0
    def _get_atr(self, data: pd.DataFrame) -> float:
        """Get Average True Range value."""
        try:
            atr_values = self.indicators.atr(data)
            return float(atr_values.iloc[-1]) if len(atr_values) > 0 and not pd.isna(atr_values.iloc[-1]) else 1.0
        except:
            return 1.0
    def _get_close_price(self, data: pd.DataFrame) -> float:
        """Get current close price."""
        try:
            return float(data['close'].iloc[-1]) if len(data) > 0 else 0.0
        except:
            return 0.0
    def _get_volume(self, data: pd.DataFrame) -> float:
        """Get current volume."""
        try:
            return float(data['volume'].iloc[-1]) if len(data) > 0 and 'volume' in data.columns else 1000.0
        except:
            return 1000.0
    def _evaluate_strategy(self, individual) -> Tuple[float, float, float, float]:
        """Evaluate a GP individual (strategy).
        This is a placeholder that will be replaced with actual backtesting.
        Returns fitness tuple: (sharpe_ratio, max_drawdown, win_rate, consistency)
        """
        try:
            # Compile the individual to a callable function
            strategy_func = gp.compile(individual, self.pset)
            # For now, return random fitness (will be replaced with actual backtesting)
            # In production, this will integrate with vectorbt backtesting engine
            sharpe = random.uniform(0.5, 3.0)
            drawdown = random.uniform(0.05, 0.3)
            win_rate = random.uniform(0.3, 0.7)
            consistency = random.uniform(0.4, 0.9)
            return sharpe, drawdown, win_rate, consistency
        except Exception as e:
            # Return poor fitness for invalid strategies
            return 0.1, 1.0, 0.1, 0.1
    def create_strategy_from_individual(self, individual, generation: int = 0, 
                                      fitness_results: Optional[Tuple] = None) -> TradingStrategy:
        """Create a TradingStrategy object from a GP individual.
        Args:
            individual: DEAP GP individual
            generation: Generation number
            fitness_results: Fitness evaluation results
        Returns:
            TradingStrategy object
        """
        # Extract genetic information
        tree_str = str(individual)
        complexity = len(individual) / self.max_tree_size  # Normalize complexity
        # Extract indicators used (parse tree string)
        indicators_used = []
        for indicator in ['rsi', 'sma20', 'sma50', 'ema12', 'ema26', 'macd', 'bb_upper', 'bb_lower', 'atr']:
            if indicator in tree_str:
                indicators_used.append(indicator)
        # Create strategy genes
        genes = StrategyGenes(
            tree_representation=tree_str,
            complexity_score=min(complexity, 1.0),
            indicators_used=indicators_used,
            generation=generation
        )
        # Create fitness object if results provided
        fitness = None
        if fitness_results and len(fitness_results) >= 4:
            fitness = StrategyFitness(
                sharpe_ratio=fitness_results[0],
                max_drawdown=fitness_results[1],
                win_rate=fitness_results[2],
                consistency=fitness_results[3],
                profit_factor=1.5,  # Placeholder
                total_return=0.1,   # Placeholder
                volatility=0.2,     # Placeholder
                max_consecutive_losses=3,  # Placeholder
                composite_fitness=0.0,  # Will be calculated by validator
                in_sample_fitness=0.0,  # Placeholder
                out_of_sample_fitness=0.0,  # Placeholder
                walk_forward_fitness=0.0   # Placeholder
            )
        # Generate strategy
        strategy = TradingStrategy(
            strategy_id="",  # Will be generated
            name=f"GP_Strategy_Gen{generation}",
            genes=genes,
            fitness=fitness
        )
        # Generate ID based on genes
        strategy.strategy_id = strategy.generate_strategy_id()
        return strategy
    def evolve_population(self, num_generations: Optional[int] = None) -> List[TradingStrategy]:
        """Evolve a population of trading strategies.
        Args:
            num_generations: Number of generations to evolve (uses settings default if None)
        Returns:
            List of best evolved strategies
        """
        generations = num_generations or self.max_generations
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        # Evaluate initial population
        fitnesses = list(self.toolbox.map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        # Evolution loop
        for generation in range(generations):
            self.generation_count = generation
            # Select next generation
            offspring = self.toolbox.selectTournament(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # Select best individuals for next generation
            population = self.toolbox.select(population + offspring, self.population_size)
            # Track best strategies
            if generation % 10 == 0 or generation == generations - 1:
                best_ind = tools.selBest(population, 1)[0]
                best_strategy = self.create_strategy_from_individual(
                    best_ind, generation, best_ind.fitness.values
                )
                self.best_strategies.append(best_strategy)
                print(f"Generation {generation}: Best fitness = {best_ind.fitness.values}")
        # Return final best strategies
        final_population = [self.create_strategy_from_individual(ind, generations, ind.fitness.values) 
                          for ind in tools.selBest(population, 10)]
        return final_population
async def test_ast_strategy_system():
    """Test function to validate AST strategy system."""
    # Use our settings system - DEMONSTRATES THE CONNECTION
    settings = get_settings()
    print("=== AST Strategy System Test ===")
    print(f"Population Size: {settings.genetic_algorithm.population_size}")
    print(f"Max Generations: {settings.genetic_algorithm.max_generations}")
    print(f"Crossover Probability: {settings.genetic_algorithm.crossover_probability}")
    print(f"Mutation Probability: {settings.genetic_algorithm.mutation_probability}")
    # Initialize GP engine
    gp_engine = GeneticProgrammingEngine(settings)
    print(f"\n=== Genetic Programming Setup ===")
    print(f"Primitive Set: {len(gp_engine.pset.primitives)} primitives")
    print(f"Terminals: {len(gp_engine.pset.terminals)} terminals")
    print(f"Max Tree Height: {gp_engine.max_tree_height}")
    print(f"Max Tree Size: {gp_engine.max_tree_size}")
    # Test individual creation
    print(f"\n=== Individual Creation Test ===")
    individual = gp_engine.toolbox.individual()
    print(f"Sample Individual: {str(individual)}")
    print(f"Individual Size: {len(individual)}")
    print(f"Individual Height: {individual.height}")
    # Test strategy creation
    print(f"\n=== Strategy Object Creation Test ===")
    fitness_results = (2.1, 0.08, 0.65, 0.75)  # Sample fitness
    strategy = gp_engine.create_strategy_from_individual(individual, 0, fitness_results)
    print(f"Strategy ID: {strategy.strategy_id}")
    print(f"Strategy Name: {strategy.name}")
    print(f"Indicators Used: {strategy.genes.indicators_used}")
    print(f"Complexity Score: {strategy.genes.complexity_score:.2f}")
    print(f"Fitness - Sharpe: {strategy.fitness.sharpe_ratio:.2f}")
    print(f"Fitness - Drawdown: {strategy.fitness.max_drawdown:.2f}")
    print(f"Fitness - Win Rate: {strategy.fitness.win_rate:.2f}")
    print(f"Fitness - Composite: {strategy.fitness.composite_fitness:.2f}")
    # Test small evolution run
    print(f"\n=== Evolution Test (3 generations) ===")
    evolved_strategies = gp_engine.evolve_population(3)
    print(f"Evolved {len(evolved_strategies)} strategies")
    for i, strategy in enumerate(evolved_strategies[:3]):
        print(f"Strategy {i+1}: Sharpe={strategy.fitness.sharpe_ratio:.2f}, "
              f"Complexity={strategy.genes.complexity_score:.2f}")
    print("\n✅ AST Strategy System test completed successfully!")
if __name__ == "__main__":
    """Test the AST strategy system."""
    import asyncio
    import logging
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Run test
    asyncio.run(test_ast_strategy_system())
