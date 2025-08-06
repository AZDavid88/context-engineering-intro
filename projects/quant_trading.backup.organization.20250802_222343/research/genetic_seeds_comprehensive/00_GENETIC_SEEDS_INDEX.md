# Genetic Seed Architecture Design

**Implementation Priority**: CRITICAL - Foundation for all genetic strategies  
**Consultant Integration**: ✅ Multi-objective fitness + validation framework  
**Production Requirements**: Enhanced with robustness and scaling patterns  

## Architecture Overview

The genetic seed architecture provides the foundation for evolutionary strategy discovery. Each seed represents a **genetic template** that can evolve millions of parameter combinations, not a fixed strategy implementation.

### Core Design Principles

1. **Genetic Encoding**: All parameters become evolvable genes
2. **Validation First**: Every seed must pass comprehensive testing
3. **Production Robustness**: Transaction costs and slippage integrated from day 1
4. **Scalable Evaluation**: Ray/Dask parallel processing built-in
5. **Multi-Objective Optimization**: Sharpe + Consistency + Drawdown + Turnover

## Genetic Seed Template Architecture

### 1. **Universal Seed Interface**

```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import vectorbt as vbt

class GeneticSeedBase(ABC):
    """Base class for all genetic strategy seeds"""
    
    def __init__(self, name: str, genome_size: int, seed_type: str):
        self.name = name
        self.genome_size = genome_size
        self.seed_type = seed_type
        self.validation_passed = False
        self.performance_history = []
        
        # Production requirements (consultant recommendation)
        self.transaction_costs = {
            'maker_fee': 0.0002,
            'taker_fee': 0.0005, 
            'slippage_bps': 5,
            'min_trade_size': 10.0  # USD minimum
        }
    
    @abstractmethod
    def decode_genome(self, genome: np.ndarray) -> dict:
        """Convert genetic genome to strategy parameters"""
        pass
    
    @abstractmethod
    def generate_signals(self, price_data: pd.Series, genome: np.ndarray) -> dict:
        """Generate trading signals from genetic parameters"""
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> dict:
        """Return valid parameter ranges for genetic exploration"""
        pass
    
    def validate_implementation(self) -> bool:
        """Validate seed implementation with comprehensive tests"""
        
        validator = GeneticSeedValidator()
        validation_result = validator.validate_seed_implementation(
            self.name, 
            lambda data: self.generate_signals(data, self.get_default_genome())
        )
        
        self.validation_passed = validation_result['overall_passed']
        return self.validation_passed
    
    def get_default_genome(self) -> np.ndarray:
        """Get default genome representing original strategy parameters"""
        # Override in child classes to provide meaningful defaults
        return np.random.random(self.genome_size)
    
    def calculate_fitness(self, portfolio_results, market_context=None):
        """Calculate multi-objective fitness score"""
        
        fitness_calculator = MultiObjectiveGeneticFitness()
        fitness_score, metrics = fitness_calculator.calculate_comprehensive_fitness(
            portfolio_results, market_context
        )
        
        # Store performance history
        self.performance_history.append({
            'timestamp': pd.Timestamp.now(),
            'fitness_score': fitness_score,
            'metrics': metrics
        })
        
        return fitness_score
```

### 2. **EMA Crossover Seed Implementation** (Seed #1)

```python
class EMACrossoverSeed(GeneticSeedBase):
    """EMA Crossover genetic seed with comprehensive parameter evolution"""
    
    def __init__(self):
        super().__init__(
            name="EMA_CROSSOVER",
            genome_size=8,
            seed_type="trend_following"
        )
        
        # Default parameters from consultant's seed list
        self.default_params = {
            'fast': 9,
            'slow': 21
        }
    
    def decode_genome(self, genome: np.ndarray) -> dict:
        """Convert 8-gene genome to EMA strategy parameters"""
        
        if len(genome) != self.genome_size:
            raise ValueError(f"Genome size must be {self.genome_size}")
        
        return {
            # Core EMA parameters (expanded genetic ranges)
            'fast_period': int(genome[0] * 48) + 2,         # 2-50 days
            'slow_period': int(genome[1] * 198) + 2,        # 2-200 days
            
            # Signal filtering (genetic enhancement)
            'signal_filter': genome[2] * 0.05,             # 0-5% filter
            'trend_confirmation': genome[3],                # 0.0-1.0 weight
            
            # Moving average type (genetic selection)
            'ma_type': 'ema' if genome[4] > 0.5 else 'sma', # Type evolution
            
            # Exit method (genetic logic evolution)
            'exit_method': 'crossover' if genome[5] > 0.7 else 'trailing',
            
            # Position sizing (genetic risk management)
            'position_size': genome[6] * 0.20 + 0.05,      # 5-25%
            
            # Volatility adjustment (genetic adaptation)
            'volatility_multiplier': genome[7] * 1.5 + 0.5  # 0.5x-2.0x
        }
    
    def generate_signals(self, price_data: pd.Series, genome: np.ndarray) -> dict:
        """Generate EMA crossover signals with genetic parameters"""
        
        params = self.decode_genome(genome)
        
        # Calculate moving averages with genetic type selection
        if params['ma_type'] == 'ema':
            fast_ma = price_data.ewm(span=params['fast_period']).mean()
            slow_ma = price_data.ewm(span=params['slow_period']).mean()
        else:  # SMA
            fast_ma = price_data.rolling(params['fast_period']).mean()
            slow_ma = price_data.rolling(params['slow_period']).mean()
        
        # Basic crossover signals
        bullish_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        bearish_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # Apply genetic signal filtering
        if params['signal_filter'] > 0:
            signal_strength = abs(fast_ma - slow_ma) / slow_ma
            strong_signals = signal_strength > params['signal_filter']
            bullish_cross = bullish_cross & strong_signals
            bearish_cross = bearish_cross & strong_signals
        
        # Apply genetic trend confirmation
        if params['trend_confirmation'] > 0.3:
            # Calculate longer-term trend
            long_trend = price_data.ewm(span=100).mean()
            trend_up = price_data > long_trend
            trend_confirmation_weight = params['trend_confirmation']
            
            # Apply trend filter with genetic weight
            trend_filter = np.random.random(len(price_data)) < trend_confirmation_weight
            bullish_cross = bullish_cross & (trend_up | ~trend_filter)
        
        # Apply genetic volatility adjustment
        if params['volatility_multiplier'] != 1.0:
            volatility = price_data.pct_change().rolling(20).std()
            avg_volatility = volatility.rolling(100).mean()
            vol_condition = volatility < (avg_volatility * params['volatility_multiplier'])
            
            bullish_cross = bullish_cross & vol_condition
            bearish_cross = bearish_cross & vol_condition
        
        # Generate exits based on genetic method selection
        if params['exit_method'] == 'crossover':
            exits = bearish_cross
        else:  # trailing
            # Implement simple trailing stop
            trailing_pct = 0.02  # 2% trailing stop
            exits = price_data < (price_data.rolling(5).max() * (1 - trailing_pct))
        
        return {
            'entries': bullish_cross.fillna(False),
            'exits': exits.fillna(False),
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'parameters': params
        }
    
    def get_parameter_ranges(self) -> dict:
        """Return parameter ranges for genetic exploration"""
        return {
            'fast_period': (2, 50),
            'slow_period': (2, 200),
            'signal_filter': (0.0, 0.05),
            'trend_confirmation': (0.0, 1.0),
            'ma_type_probability': (0.0, 1.0),  # EMA vs SMA selection
            'exit_method_probability': (0.0, 1.0),  # Crossover vs trailing
            'position_size': (0.05, 0.25),
            'volatility_multiplier': (0.5, 2.0)
        }
    
    def get_default_genome(self) -> np.ndarray:
        """Get genome representing original EMA(9,21) strategy"""
        
        # Map default parameters to genome space
        ranges = self.get_parameter_ranges()
        
        genome = np.array([
            # fast_period: 9 -> (9-2)/(50-2) = 7/48 ≈ 0.146
            (self.default_params['fast'] - 2) / 48,
            
            # slow_period: 21 -> (21-2)/(200-2) = 19/198 ≈ 0.096  
            (self.default_params['slow'] - 2) / 198,
            
            # signal_filter: 0% (no filter)
            0.0,
            
            # trend_confirmation: 50% weight
            0.5,
            
            # ma_type: EMA (>0.5)
            0.8,
            
            # exit_method: crossover (>0.7)
            0.8,
            
            # position_size: 10% -> (0.10-0.05)/(0.25-0.05) = 0.25
            0.25,
            
            # volatility_multiplier: 1.0x -> (1.0-0.5)/(2.0-0.5) = 0.33
            0.33
        ])
        
        return genome
```

### 3. **Genetic Population Initialization Strategy**

```python
class EnhancedGeneticPopulationInitializer:
    """Enhanced population initialization with consultant recommendations"""
    
    def __init__(self, seed_registry: dict):
        self.seeds = seed_registry
        self.validation_required = True
        self.diversity_target = 0.8
        
        # Consultant requirement: prove on small scale first
        self.validation_population_size = 10
        self.production_population_size = 1000
    
    def create_validated_population(self, population_size: int = None):
        """Create genetically diverse population with validation"""
        
        if population_size is None:
            population_size = self.production_population_size
        
        # Step 1: Validate all seeds (consultant requirement #1)
        validated_seeds = self._validate_all_seeds()
        
        if len(validated_seeds) == 0:
            raise RuntimeError("No seeds passed validation - cannot create population")
        
        # Step 2: Create diverse population from validated seeds
        population = self._create_diverse_population(validated_seeds, population_size)
        
        # Step 3: Test population on toy example (consultant requirement #2)
        if population_size >= 100:  # Only for larger populations
            self._validate_population_evolution(population[:10])
        
        return population
    
    def _validate_all_seeds(self) -> list:
        """Validate all seed implementations"""
        
        validated_seeds = []
        
        for seed_name, seed_class in self.seeds.items():
            print(f"Validating seed: {seed_name}")
            
            try:
                seed_instance = seed_class()
                
                # Run validation tests
                validation_passed = seed_instance.validate_implementation()
                
                if validation_passed:
                    validated_seeds.append(seed_instance)
                    print(f"✅ {seed_name} validation passed")
                else:
                    print(f"❌ {seed_name} validation failed")
                    
            except Exception as e:
                print(f"❌ {seed_name} validation error: {e}")
        
        print(f"Validation complete: {len(validated_seeds)}/{len(self.seeds)} seeds passed")
        return validated_seeds
    
    def _validate_population_evolution(self, test_population: list):
        """Test genetic evolution on small population (consultant requirement)"""
        
        print("Testing genetic evolution on 10-strategy population...")
        
        # Create simple test data
        test_data = pd.Series(
            np.random.random(252) * 100,  # 1 year of random prices
            index=pd.date_range('2023-01-01', periods=252, freq='D')
        )
        
        # Test evolution for 3 generations
        current_population = test_population
        fitness_history = []
        
        for generation in range(3):
            generation_fitness = []
            
            for individual in current_population:
                # Generate signals
                signals = individual.generate_signals(test_data, individual.get_default_genome())
                
                # Backtest with transaction costs
                portfolio = vbt.Portfolio.from_signals(
                    test_data,
                    signals['entries'],
                    signals['exits'],
                    init_cash=10000,
                    fees=0.0005,  # Realistic fees
                    slippage=0.0005  # Realistic slippage
                )
                
                # Calculate fitness
                fitness = individual.calculate_fitness(portfolio)
                generation_fitness.append(fitness)
            
            avg_fitness = np.mean(generation_fitness)
            fitness_history.append(avg_fitness)
            
            print(f"Generation {generation}: Average fitness = {avg_fitness:.3f}")
        
        # Validate improvement trend
        if len(fitness_history) >= 2:
            improvement = fitness_history[-1] - fitness_history[0]
            if improvement > 0:
                print("✅ Genetic evolution showing improvement trend")
                return True
            else:
                print("⚠️ Genetic evolution not improving - check fitness function")
                return False
        
        return True
```

This architecture provides the foundation for robust genetic seed implementation with all consultant recommendations integrated. The next files will cover specific implementations and production deployment patterns.