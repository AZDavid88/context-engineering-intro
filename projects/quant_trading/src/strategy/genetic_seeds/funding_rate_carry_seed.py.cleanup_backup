


from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional



from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

    
    
    
    
    
        
        
        
    
        
            
        
            
            
        
        
        
        
        
        
        
        
        
        
    
        
            
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
    
        
            
        
        
        
        
    
        
            
        
        
        
        
        
        
        
    
        
            
        
        
        
        
        
        
        
    
        
            
        
        
        
        
    
        
"""
Funding Rate Carry Genetic Seed - Seed #10
This seed implements crypto funding rate carry strategies specifically for
Hyperliquid perpetual futures. The genetic algorithm evolves optimal funding
rate thresholds and carry trade parameters.
Key Features:
- Funding rate threshold detection with genetic parameters
- Carry trade duration optimization
- Risk management for funding rate reversals
- Multi-asset funding rate arbitrage
"""
@genetic_seed
class FundingRateCarrySeed(BaseSeed):
    """Funding rate carry seed for crypto perpetual futures."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "Funding_Rate_Carry"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("Crypto funding rate carry strategy for perpetual futures. "
                "Exploits funding rate anomalies and carries positions based on "
                "funding rate predictions. Optimized for Hyperliquid exchange.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'funding_threshold',
            'carry_duration',
            'rate_momentum',
            'reversal_sensitivity',
            'funding_persistence'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max)."""
        return {
            'funding_threshold': (0.0001, 0.01),     # 0.01% to 1% funding threshold
            'carry_duration': (1.0, 72.0),           # 1 to 72 hours carry duration
            'rate_momentum': (0.0, 1.0),             # Funding rate momentum weight
            'reversal_sensitivity': (0.1, 1.0),      # Sensitivity to rate reversals
            'funding_persistence': (2.0, 24.0)       # Hours for funding persistence
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize funding rate carry seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.CARRY
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'funding_threshold': 0.001,    # 0.1% funding threshold
                'carry_duration': 8.0,         # 8 hours default carry
                'rate_momentum': 0.5,          # 50% momentum weight
                'reversal_sensitivity': 0.7,   # 70% reversal sensitivity
                'funding_persistence': 6.0     # 6 hours persistence
            }
        super().__init__(genes, settings)
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate funding rate indicators and carry metrics.
        Args:
            data: Market data with funding_rate column if available
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        persistence_hours = int(self.genes.parameters['funding_persistence'])
        # Simulate funding rate if not available (for backtesting)
        if 'funding_rate' not in data.columns:
            # Create realistic funding rate simulation
            returns = data['close'].pct_change()
            price_momentum = returns.rolling(window=24).mean()  # 24-hour momentum
            # Funding rates typically range from -0.5% to +0.5% (8-hour periods)
            base_funding = price_momentum * 0.1  # Scale momentum to funding
            noise = np.random.normal(0, 0.0001, len(data))  # Add realistic noise
            funding_rate = base_funding + noise
            # Clamp to realistic bounds
            funding_rate = np.clip(funding_rate, -0.005, 0.005)
        else:
            funding_rate = data['funding_rate']
        # Convert to pandas Series if needed
        if not isinstance(funding_rate, pd.Series):
            funding_rate = pd.Series(funding_rate, index=data.index)
        # Funding rate momentum and trends
        funding_ma_short = funding_rate.rolling(window=8).mean()  # 8-period MA
        funding_ma_long = funding_rate.rolling(window=24).mean()  # 24-period MA
        funding_momentum = funding_ma_short - funding_ma_long
        # Funding rate volatility
        funding_volatility = funding_rate.rolling(window=24).std()
        # Extreme funding detection
        funding_percentile = funding_rate.rolling(window=168).rank(pct=True)  # 1 week percentile
        extreme_positive = funding_percentile > 0.9  # Top 10%
        extreme_negative = funding_percentile < 0.1  # Bottom 10%
        # Funding rate persistence
        positive_funding = funding_rate > 0
        negative_funding = funding_rate < 0
        positive_persistence = positive_funding.rolling(window=persistence_hours).sum()
        negative_persistence = negative_funding.rolling(window=persistence_hours).sum()
        # Funding rate reversal detection
        funding_change = funding_rate.diff()
        reversal_strength = abs(funding_change) / funding_volatility
        # Price-funding divergence
        price_returns = data['close'].pct_change(periods=8)  # 8-period return
        funding_price_correlation = funding_rate.rolling(window=24).corr(price_returns)
        divergence = abs(funding_price_correlation) < 0.3  # Low correlation = divergence
        # Expected profit calculation
        # Funding is typically paid every 8 hours
        expected_hourly_profit = funding_rate / 8  # Convert 8-hour rate to hourly
        return {
            'funding_rate': safe_fillna_zero(funding_rate),
            'funding_ma_short': safe_fillna_zero(funding_ma_short),
            'funding_ma_long': safe_fillna_zero(funding_ma_long),
            'funding_momentum': safe_fillna_zero(funding_momentum),
            'funding_volatility': funding_volatility.fillna(0.0001),
            'extreme_positive': safe_fillna_false(extreme_positive),
            'extreme_negative': safe_fillna_false(extreme_negative),
            'positive_persistence': safe_fillna_zero(positive_persistence),
            'negative_persistence': safe_fillna_zero(negative_persistence),
            'reversal_strength': safe_fillna_zero(reversal_strength),
            'divergence': safe_fillna_false(divergence),
            'expected_hourly_profit': safe_fillna_zero(expected_hourly_profit),
            'funding_percentile': funding_percentile.fillna(0.5)
        }
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate funding rate carry trading signals.
        Args:
            data: Market data with funding rate information
        Returns:
            Series of trading signals: 1 (long carry), 0 (no position), -1 (short carry)
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        # Get genetic parameters
        funding_threshold = self.genes.parameters['funding_threshold']
        rate_momentum = self.genes.parameters['rate_momentum']
        reversal_sensitivity = self.genes.parameters['reversal_sensitivity']
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        funding_rate = indicators['funding_rate']
        funding_momentum = indicators['funding_momentum']
        reversal_strength = indicators['reversal_strength']
        # Positive funding rate carry (short perpetual, earn funding)
        positive_carry_conditions = (
            (funding_rate > funding_threshold) &  # High enough funding rate
            (indicators['positive_persistence'] >= 2) &  # Persistent positive funding
            (reversal_strength < reversal_sensitivity)  # No strong reversal signal
        )
        # Negative funding rate carry (long perpetual, pay negative funding)
        negative_carry_conditions = (
            (funding_rate < -funding_threshold) &  # High enough negative funding rate
            (indicators['negative_persistence'] >= 2) &  # Persistent negative funding
            (reversal_strength < reversal_sensitivity)  # No strong reversal signal
        )
        # Apply momentum filter if genetic parameter > 0.3
        if rate_momentum > 0.3:
            # Strengthen signals when momentum aligns
            momentum_boost_positive = funding_momentum > 0
            momentum_boost_negative = funding_momentum < 0
            positive_carry_conditions = positive_carry_conditions & momentum_boost_positive
            negative_carry_conditions = negative_carry_conditions & momentum_boost_negative
        # Extreme funding rate opportunities (higher signal strength)
        extreme_positive_carry = (
            indicators['extreme_positive'] &
            positive_carry_conditions
        )
        extreme_negative_carry = (
            indicators['extreme_negative'] &
            negative_carry_conditions
        )
        # Generate signals
        # Short perpetual when funding is positive (earn funding)
        signals[positive_carry_conditions] = -0.5  # Medium short signal
        signals[extreme_positive_carry] = -1.0     # Strong short signal
        # Long perpetual when funding is negative (pay negative funding = earn)
        signals[negative_carry_conditions] = 0.5   # Medium long signal
        signals[extreme_negative_carry] = 1.0      # Strong long signal
        # Exit signals when funding reverses
        strong_reversal = reversal_strength > reversal_sensitivity
        signals[strong_reversal] = 0
        # Exit when funding rate approaches zero
        low_funding = abs(funding_rate) < (funding_threshold * 0.3)
        signals[low_funding] = 0
        # Fill any NaN values
        signals = safe_fillna_zero(signals)
        return signals
    def calculate_expected_profit(self, data: pd.DataFrame, position_hours: int = 8) -> pd.Series:
        """Calculate expected profit from funding rate carry.
        Args:
            data: Market data with funding rate
            position_hours: Expected position holding time in hours
        Returns:
            Series of expected profit percentages
        """
        indicators = self.calculate_technical_indicators(data)
        # Expected profit based on funding rate and holding period
        hourly_profit = indicators['expected_hourly_profit']
        expected_profit = hourly_profit * position_hours
        # Adjust for funding volatility (higher vol = higher risk)
        funding_vol = indicators['funding_volatility']
        risk_adjustment = 1 - (funding_vol * 10)  # Scale volatility impact
        risk_adjustment = np.clip(risk_adjustment, 0.5, 1.0)
        adjusted_profit = expected_profit * risk_adjustment
        return adjusted_profit
    def get_optimal_carry_duration(self, data: pd.DataFrame) -> float:
        """Get optimal carry duration based on current market conditions.
        Args:
            data: Current market data
        Returns:
            Optimal carry duration in hours
        """
        indicators = self.calculate_technical_indicators(data.tail(24))  # Last 24 hours
        # Base duration from genetic parameters
        base_duration = self.genes.parameters['carry_duration']
        # Adjust based on funding persistence
        recent_persistence = indicators['positive_persistence'].iloc[-1] if len(indicators['positive_persistence']) > 0 else 0
        # If funding has been persistent, extend duration
        if recent_persistence > 12:  # 12+ hours of persistence
            duration_multiplier = 1.5
        elif recent_persistence > 6:  # 6+ hours of persistence
            duration_multiplier = 1.2
        else:
            duration_multiplier = 0.8  # Reduce duration for less persistent funding
        # Adjust based on funding volatility
        recent_vol = indicators['funding_volatility'].iloc[-1] if len(indicators['funding_volatility']) > 0 else 0.0001
        if recent_vol > 0.001:  # High volatility
            duration_multiplier *= 0.7  # Reduce duration
        elif recent_vol < 0.0003:  # Low volatility
            duration_multiplier *= 1.3  # Extend duration
        optimal_duration = base_duration * duration_multiplier
        # Clamp to reasonable bounds
        return max(1.0, min(72.0, optimal_duration))
    def should_exit_carry_position(self, data: pd.DataFrame, entry_funding_rate: float,
                                 hours_held: float) -> bool:
        """Determine if carry position should be exited.
        Args:
            data: Current market data
            entry_funding_rate: Funding rate when position was entered
            hours_held: Hours position has been held
        Returns:
            True if position should be exited
        """
        indicators = self.calculate_technical_indicators(data.tail(5))
        current_funding = indicators['funding_rate'].iloc[-1] if len(indicators['funding_rate']) > 0 else 0
        # Get genetic parameters
        reversal_sensitivity = self.genes.parameters['reversal_sensitivity']
        # Exit conditions:
        # 1. Funding rate has reversed significantly
        funding_change = abs(current_funding - entry_funding_rate)
        funding_reversal = funding_change > (abs(entry_funding_rate) * reversal_sensitivity)
        if funding_reversal:
            return True
        # 2. Funding rate is too small to be profitable
        funding_threshold = self.genes.parameters['funding_threshold']
        if abs(current_funding) < (funding_threshold * 0.3):
            return True
        # 3. Maximum carry duration reached
        max_duration = self.get_optimal_carry_duration(data)
        if hours_held >= max_duration:
            return True
        # 4. Strong reversal signal detected
        reversal_strength = indicators['reversal_strength'].iloc[-1] if len(indicators['reversal_strength']) > 0 else 0
        if reversal_strength > reversal_sensitivity:
            return True
        return False
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate position size based on funding rate strength.
        Args:
            data: Current market data
            signal: Signal strength (-1 to 1)
        Returns:
            Position size as percentage of capital
        """
        if abs(signal) < self.genes.filter_threshold:
            return 0.0
        # Calculate expected profit to scale position size
        expected_profit = self.calculate_expected_profit(data.tail(5))
        current_expected = expected_profit.iloc[-1] if len(expected_profit) > 0 else 0
        # Base position size from genes
        base_size = self.genes.position_size
        # Scale by signal strength and expected profit
        profit_multiplier = min(abs(current_expected) * 100, 2.0)  # Cap at 2x
        position_size = base_size * abs(signal) * max(profit_multiplier, 0.5)
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        threshold = self.genes.parameters.get('funding_threshold', 0.001)
        duration = self.genes.parameters.get('carry_duration', 8)
        momentum = self.genes.parameters.get('rate_momentum', 0.5)
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"FundingCarry(thr={threshold:.4f},dur={duration:.0f}h,mom={momentum:.2f}){fitness_str}"