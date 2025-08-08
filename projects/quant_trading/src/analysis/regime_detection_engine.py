"""
Composite Market Regime Detection Engine - Phase 3 Implementation

Multi-source regime detection engine combining all regime indicators following
the phase plan specifications and existing architectural patterns.

Key Features:
- Composite regime analysis from sentiment, volatility, correlation, and volume
- Regime conflict resolution and confidence scoring  
- Regime transition detection and stability assessment
- Integration with existing fear_greed_client.py and correlation_engine.py
- Genetic algorithm environmental pressure generation
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.data.fear_greed_client import FearGreedClient, MarketRegime as SentimentRegime
from src.analysis.correlation_engine import FilteredAssetCorrelationEngine
from src.analysis.regime_detectors import (
    VolatilityRegimeDetector, 
    CorrelationRegimeDetector, 
    VolumeRegimeDetector
)
from src.config.settings import get_settings


class CompositeRegime(str, Enum):
    """Composite market regime classifications."""
    RISK_ON = "risk_on"                    # Favorable market conditions
    RISK_OFF = "risk_off"                  # Defensive market conditions
    NEUTRAL = "neutral"                    # Balanced market conditions
    TRANSITIONAL = "transitional"         # Regime change period


@dataclass
class RegimeAnalysis:
    """Comprehensive market regime analysis results."""
    
    # Individual regime signals
    sentiment_regime: str = "neutral"
    volatility_regime: str = "medium_volatility"
    correlation_regime: str = "normal_correlation"
    volume_regime: str = "normal_volume"
    
    # Composite analysis
    composite_regime: CompositeRegime = CompositeRegime.NEUTRAL
    regime_confidence: float = 0.5
    regime_stability: float = 0.5
    
    # Supporting metrics
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    asset_count: int = 0
    data_quality_score: float = 0.0
    
    # Regime scoring breakdown
    regime_scores: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_favorable_conditions(self) -> bool:
        """Check if market conditions favor active trading."""
        return self.composite_regime == CompositeRegime.RISK_ON
    
    @property
    def is_defensive_conditions(self) -> bool:
        """Check if market conditions favor defensive positioning."""
        return self.composite_regime == CompositeRegime.RISK_OFF
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if regime classification has high confidence."""
        return self.regime_confidence > 0.7


class CompositeRegimeDetectionEngine:
    """
    Multi-source regime detection engine combining all regime indicators.
    Extends existing fear_greed_client.py with additional regime sources.
    """
    
    def __init__(self, 
                 fear_greed_client: Optional[FearGreedClient] = None,
                 correlation_engine: Optional[FilteredAssetCorrelationEngine] = None,
                 settings=None):
        """Initialize composite regime engine with existing components."""
        
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Existing components integration
        self.fear_greed_client = fear_greed_client or FearGreedClient(self.settings)
        self.correlation_engine = correlation_engine or FilteredAssetCorrelationEngine(self.settings)
        
        # New regime detectors
        self.volatility_detector = VolatilityRegimeDetector(self.settings)
        self.correlation_detector = CorrelationRegimeDetector(self.correlation_engine, self.settings)
        self.volume_detector = VolumeRegimeDetector(self.settings)
        
        # Regime fusion configuration following phase plan specifications
        regime_settings = getattr(self.settings, 'regime_detection', None)
        if regime_settings:
            self.regime_weights = getattr(regime_settings, 'regime_weights', {
                'sentiment': 0.3,     # Fear/greed sentiment weight
                'volatility': 0.25,   # Market volatility state weight
                'correlation': 0.25,  # Asset correlation state weight
                'volume': 0.2         # Volume participation state weight
            })
            self.confidence_threshold = getattr(regime_settings, 'regime_confidence_threshold', 0.7)
            self.stability_threshold = getattr(regime_settings, 'regime_stability_threshold', 0.8)
        else:
            # Safe defaults matching phase plan
            self.regime_weights = {
                'sentiment': 0.3,     # Fear/greed sentiment
                'volatility': 0.25,   # Market volatility state
                'correlation': 0.25,  # Asset correlation state
                'volume': 0.2         # Volume participation state
            }
            self.confidence_threshold = 0.7
            self.stability_threshold = 0.8
        
        # Historical regime tracking for stability analysis
        self._regime_history: List[Tuple[CompositeRegime, datetime]] = []
        self._max_history_length = 50  # Keep last 50 regime detections
        
        # Performance tracking
        self._analysis_cache: Dict[str, Tuple[RegimeAnalysis, datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)  # 10-minute cache for composite analysis
        
        self.logger.info(f"🎯 CompositeRegimeDetectionEngine initialized")
        self.logger.info(f"   Regime weights: {self.regime_weights}")
    
    async def detect_composite_regime(
        self,
        filtered_assets: List[str],
        timeframe: str = '1h',
        force_refresh: bool = False
    ) -> RegimeAnalysis:
        """
        Detect comprehensive market regime from all sources.
        Main interface for multi-source regime detection.
        
        Args:
            filtered_assets: List of asset symbols for analysis
            timeframe: Data timeframe for analysis
            force_refresh: Force refresh of all cached data
            
        Returns:
            RegimeAnalysis with comprehensive regime classification
        """
        cache_key = f"{'-'.join(sorted(filtered_assets))}_{timeframe}"
        
        # Check cache first
        if not force_refresh and cache_key in self._analysis_cache:
            cached_analysis, cache_time = self._analysis_cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl:
                self.logger.debug(f"Using cached composite regime analysis")
                return cached_analysis
        
        self.logger.info(f"🎯 Detecting composite market regime for {len(filtered_assets)} assets")
        
        try:
            # Gather individual regime signals concurrently for performance
            regime_tasks = await self._gather_individual_regime_signals(
                filtered_assets, timeframe, force_refresh
            )
            
            individual_regimes, data_quality_scores = regime_tasks
            
            # Calculate composite regime from individual signals
            composite_regime = self._calculate_composite_regime(individual_regimes)
            
            # Calculate regime confidence based on signal alignment
            regime_confidence = self._calculate_regime_confidence(individual_regimes)
            
            # Assess regime stability based on history
            regime_stability = self._assess_regime_stability(composite_regime)
            
            # Calculate regime scores breakdown
            regime_scores = self._calculate_regime_scores(individual_regimes)
            
            # Create comprehensive analysis
            analysis = RegimeAnalysis(
                sentiment_regime=individual_regimes['sentiment_regime'],
                volatility_regime=individual_regimes['volatility_regime'],
                correlation_regime=individual_regimes['correlation_regime'],
                volume_regime=individual_regimes['volume_regime'],
                composite_regime=composite_regime,
                regime_confidence=regime_confidence,
                regime_stability=regime_stability,
                calculation_timestamp=datetime.now(),
                asset_count=len(filtered_assets),
                data_quality_score=np.mean(list(data_quality_scores.values())),
                regime_scores=regime_scores
            )
            
            # Update regime history for stability tracking
            self._update_regime_history(composite_regime)
            
            # Cache results
            self._analysis_cache[cache_key] = (analysis, datetime.now())
            
            self.logger.info(f"   ✅ Composite regime: {composite_regime.value}")
            self.logger.info(f"   📊 Confidence: {regime_confidence:.1%}, Stability: {regime_stability:.1%}")
            self.logger.info(f"   🔍 Individual regimes: S:{individual_regimes['sentiment_regime'][:4]} "
                           f"V:{individual_regimes['volatility_regime'][:4]} "
                           f"C:{individual_regimes['correlation_regime'][:4]} "
                           f"Vol:{individual_regimes['volume_regime'][:4]}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Composite regime detection failed: {e}")
            # Return safe defaults
            return RegimeAnalysis(
                sentiment_regime="neutral",
                volatility_regime="medium_volatility",
                correlation_regime="normal_correlation",
                volume_regime="normal_volume",
                composite_regime=CompositeRegime.NEUTRAL,
                regime_confidence=0.5,
                regime_stability=0.5,
                calculation_timestamp=datetime.now(),
                asset_count=len(filtered_assets),
                data_quality_score=0.0,
                regime_scores={"risk_on": 0.25, "risk_off": 0.25, "neutral": 0.5, "transitional": 0.0}
            )
    
    async def _gather_individual_regime_signals(
        self,
        filtered_assets: List[str],
        timeframe: str,
        force_refresh: bool
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Gather all individual regime signals concurrently."""
        
        # Create concurrent tasks for all regime detections
        tasks = {
            'sentiment': self._get_sentiment_regime(),
            'volatility': self.volatility_detector.detect_volatility_regime(
                filtered_assets, timeframe, force_refresh
            ),
            'correlation': self.correlation_detector.detect_correlation_regime(
                filtered_assets, timeframe, force_refresh
            ),
            'volume': self.volume_detector.detect_volume_regime(
                filtered_assets, timeframe, force_refresh
            )
        }
        
        # Execute all tasks concurrently
        results = {}
        for regime_type, task in tasks.items():
            try:
                results[regime_type] = await task
            except Exception as e:
                self.logger.warning(f"Failed to get {regime_type} regime: {e}")
                # Provide safe defaults for failed detections
                if regime_type == 'sentiment':
                    results[regime_type] = 'neutral'
                elif regime_type == 'volatility':
                    from src.analysis.regime_detectors.volatility_regime_detector import VolatilityMetrics
                    results[regime_type] = VolatilityMetrics()
                elif regime_type == 'correlation':
                    from src.analysis.regime_detectors.correlation_regime_detector import CorrelationRegimeMetrics
                    results[regime_type] = CorrelationRegimeMetrics()
                elif regime_type == 'volume':
                    from src.analysis.regime_detectors.volume_regime_detector import VolumeRegimeMetrics
                    results[regime_type] = VolumeRegimeMetrics()
        
        # Extract regime classifications and data quality scores
        individual_regimes = {
            'sentiment_regime': results['sentiment'] if isinstance(results['sentiment'], str) else 'neutral',
            'volatility_regime': results['volatility'].volatility_regime if hasattr(results['volatility'], 'volatility_regime') else 'medium_volatility',
            'correlation_regime': results['correlation'].correlation_regime if hasattr(results['correlation'], 'correlation_regime') else 'normal_correlation',
            'volume_regime': results['volume'].volume_regime if hasattr(results['volume'], 'volume_regime') else 'normal_volume'
        }
        
        data_quality_scores = {
            'sentiment': 1.0,  # Fear/greed API is always high quality
            'volatility': getattr(results['volatility'], 'data_quality_score', 0.0),
            'correlation': getattr(results['correlation'], 'data_quality_score', 0.0),
            'volume': getattr(results['volume'], 'data_quality_score', 0.0)
        }
        
        return individual_regimes, data_quality_scores
    
    async def _get_sentiment_regime(self) -> str:
        """Get sentiment regime from fear/greed client."""
        try:
            fear_greed_data = await self.fear_greed_client.get_current_index()
            return fear_greed_data.regime.value
        except Exception as e:
            self.logger.warning(f"Failed to get sentiment regime: {e}")
            return "neutral"
    
    def _calculate_composite_regime(self, individual_regimes: Dict[str, str]) -> CompositeRegime:
        """
        Calculate overall market regime from individual signals.
        Follows phase plan regime scoring system.
        """
        # Initialize regime scores
        regime_scores = {
            CompositeRegime.RISK_ON: 0.0,
            CompositeRegime.RISK_OFF: 0.0,
            CompositeRegime.NEUTRAL: 0.0,
            CompositeRegime.TRANSITIONAL: 0.0
        }
        
        # Sentiment regime scoring (following phase plan logic)
        sentiment = individual_regimes.get('sentiment_regime', 'neutral')
        if sentiment in ['extreme_greed', 'greed']:
            regime_scores[CompositeRegime.RISK_ON] += self.regime_weights['sentiment']
        elif sentiment in ['extreme_fear', 'fear']:
            regime_scores[CompositeRegime.RISK_OFF] += self.regime_weights['sentiment']
        else:
            regime_scores[CompositeRegime.NEUTRAL] += self.regime_weights['sentiment']
        
        # Volatility regime scoring
        volatility = individual_regimes.get('volatility_regime', 'medium_volatility')
        if volatility == 'low_volatility':
            regime_scores[CompositeRegime.RISK_ON] += self.regime_weights['volatility']
        elif volatility == 'high_volatility':
            regime_scores[CompositeRegime.RISK_OFF] += self.regime_weights['volatility']
        else:
            regime_scores[CompositeRegime.NEUTRAL] += self.regime_weights['volatility']
        
        # Correlation regime scoring
        correlation = individual_regimes.get('correlation_regime', 'normal_correlation')
        if correlation == 'high_correlation':
            regime_scores[CompositeRegime.RISK_OFF] += self.regime_weights['correlation']
        elif correlation == 'correlation_breakdown':
            regime_scores[CompositeRegime.RISK_ON] += self.regime_weights['correlation']
        else:
            regime_scores[CompositeRegime.NEUTRAL] += self.regime_weights['correlation']
        
        # Volume regime scoring
        volume = individual_regimes.get('volume_regime', 'normal_volume')
        if volume == 'high_volume':
            regime_scores[CompositeRegime.TRANSITIONAL] += self.regime_weights['volume']
        elif volume == 'low_volume':
            regime_scores[CompositeRegime.NEUTRAL] += self.regime_weights['volume']
        else:
            regime_scores[CompositeRegime.NEUTRAL] += self.regime_weights['volume']
        
        # Return highest scoring regime
        return max(regime_scores, key=regime_scores.get)
    
    def _calculate_regime_confidence(self, individual_regimes: Dict[str, str]) -> float:
        """
        Calculate confidence in regime classification based on signal alignment.
        Improved logic following Phase 3 plan specifications.
        """
        # Weighted regime signal scoring based on plan specifications
        regime_signal_weights = {
            'sentiment': 0.3,
            'volatility': 0.25, 
            'correlation': 0.25,
            'volume': 0.2
        }
        
        # Score each regime direction
        risk_on_score = 0.0
        risk_off_score = 0.0
        neutral_score = 0.0
        
        # Sentiment scoring (greed = risk_on, fear = risk_off)
        sentiment = individual_regimes.get('sentiment_regime', 'neutral')
        if sentiment in ['extreme_greed', 'greed']:
            risk_on_score += regime_signal_weights['sentiment']
        elif sentiment in ['extreme_fear', 'fear']:
            risk_off_score += regime_signal_weights['sentiment']
        else:
            neutral_score += regime_signal_weights['sentiment']
        
        # Volatility scoring (low_vol = risk_on, high_vol = risk_off)
        volatility = individual_regimes.get('volatility_regime', 'medium_volatility')
        if volatility == 'low_volatility':
            risk_on_score += regime_signal_weights['volatility']
        elif volatility == 'high_volatility':
            risk_off_score += regime_signal_weights['volatility']
        else:
            neutral_score += regime_signal_weights['volatility']
        
        # Correlation scoring (breakdown = risk_on, high = risk_off)
        correlation = individual_regimes.get('correlation_regime', 'normal_correlation')
        if correlation == 'correlation_breakdown':
            risk_on_score += regime_signal_weights['correlation']
        elif correlation == 'high_correlation':
            risk_off_score += regime_signal_weights['correlation']
        else:
            neutral_score += regime_signal_weights['correlation']
        
        # Volume scoring (more nuanced - high volume strengthens dominant signal)
        volume = individual_regimes.get('volume_regime', 'normal_volume')
        if volume == 'high_volume':
            # High volume amplifies the currently strongest signal
            max_score = max(risk_on_score, risk_off_score, neutral_score)
            if risk_on_score == max_score:
                risk_on_score += regime_signal_weights['volume']
            elif risk_off_score == max_score:
                risk_off_score += regime_signal_weights['volume']
            else:
                neutral_score += regime_signal_weights['volume']
        elif volume == 'low_volume':
            # Low volume slightly favors neutral
            neutral_score += regime_signal_weights['volume']
        else:
            neutral_score += regime_signal_weights['volume']
        
        # Calculate confidence as the strength of the winning regime
        total_possible_score = sum(regime_signal_weights.values())
        max_regime_score = max(risk_on_score, risk_off_score, neutral_score)
        
        # Confidence is the ratio of the winning score to total possible
        raw_confidence = max_regime_score / total_possible_score
        
        # Boost confidence when signals are strongly aligned (threshold boost)
        alignment_bonus = 0.0
        if max_regime_score > 0.6:  # If winning regime has >60% of total weight
            alignment_bonus = 0.2  # 20% confidence boost
        elif max_regime_score > 0.5:  # If winning regime has >50% of total weight
            alignment_bonus = 0.1  # 10% confidence boost
        
        final_confidence = min(1.0, raw_confidence + alignment_bonus)
        
        return max(0.0, final_confidence)
    
    def _assess_regime_stability(self, current_regime: CompositeRegime) -> float:
        """Assess regime stability based on historical regime changes."""
        if len(self._regime_history) < 2:
            return 0.5  # Default stability for new systems
        
        # Look at recent regime history
        recent_history = self._regime_history[-10:]  # Last 10 detections
        
        # Count regime changes
        regime_changes = 0
        for i in range(1, len(recent_history)):
            if recent_history[i][0] != recent_history[i-1][0]:
                regime_changes += 1
        
        # Stability is inverse of change frequency
        if len(recent_history) <= 1:
            return 1.0
        
        change_rate = regime_changes / (len(recent_history) - 1)
        stability = 1.0 - change_rate
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_regime_scores(self, individual_regimes: Dict[str, str]) -> Dict[str, float]:
        """Calculate detailed regime scores for analysis."""
        regime_scores = {
            "risk_on": 0.0,
            "risk_off": 0.0,
            "neutral": 0.0,
            "transitional": 0.0
        }
        
        # This mirrors the composite regime calculation but returns all scores
        sentiment = individual_regimes.get('sentiment_regime', 'neutral')
        if sentiment in ['extreme_greed', 'greed']:
            regime_scores["risk_on"] += self.regime_weights['sentiment']
        elif sentiment in ['extreme_fear', 'fear']:
            regime_scores["risk_off"] += self.regime_weights['sentiment']
        else:
            regime_scores["neutral"] += self.regime_weights['sentiment']
        
        volatility = individual_regimes.get('volatility_regime', 'medium_volatility')
        if volatility == 'low_volatility':
            regime_scores["risk_on"] += self.regime_weights['volatility']
        elif volatility == 'high_volatility':
            regime_scores["risk_off"] += self.regime_weights['volatility']
        else:
            regime_scores["neutral"] += self.regime_weights['volatility']
        
        correlation = individual_regimes.get('correlation_regime', 'normal_correlation')
        if correlation == 'high_correlation':
            regime_scores["risk_off"] += self.regime_weights['correlation']
        elif correlation == 'correlation_breakdown':
            regime_scores["risk_on"] += self.regime_weights['correlation']
        else:
            regime_scores["neutral"] += self.regime_weights['correlation']
        
        volume = individual_regimes.get('volume_regime', 'normal_volume')
        if volume == 'high_volume':
            regime_scores["transitional"] += self.regime_weights['volume']
        else:
            regime_scores["neutral"] += self.regime_weights['volume']
        
        return regime_scores
    
    def _update_regime_history(self, regime: CompositeRegime) -> None:
        """Update regime history for stability analysis."""
        self._regime_history.append((regime, datetime.now()))
        
        # Limit history length
        if len(self._regime_history) > self._max_history_length:
            self._regime_history = self._regime_history[-self._max_history_length:]
    
    def generate_genetic_algorithm_pressure(self, analysis: RegimeAnalysis) -> Dict[str, float]:
        """
        Generate genetic algorithm environmental pressure from regime analysis.
        Integrates with existing fear_greed_client genetic pressure system.
        """
        base_pressure = {
            "contrarian_bias": 0.5,      # Contrarian strategy preference
            "momentum_bias": 0.5,        # Momentum strategy preference
            "volatility_tolerance": 0.5,  # Risk tolerance adjustment
            "position_sizing": 0.5,      # Position size preference
            "holding_period": 0.5,       # Time horizon preference
            "regime_adaptation": analysis.regime_confidence  # Regime awareness strength
        }
        
        # Adjust pressure based on composite regime
        if analysis.composite_regime == CompositeRegime.RISK_ON:
            base_pressure.update({
                "momentum_bias": 0.7,        # Favor momentum in risk-on
                "volatility_tolerance": 0.6,  # Higher risk tolerance
                "position_sizing": 0.6,      # Larger positions
                "holding_period": 0.6        # Longer time horizons
            })
            
        elif analysis.composite_regime == CompositeRegime.RISK_OFF:
            base_pressure.update({
                "contrarian_bias": 0.7,      # Favor contrarian in risk-off
                "volatility_tolerance": 0.3,  # Lower risk tolerance
                "position_sizing": 0.4,      # Smaller positions
                "holding_period": 0.4        # Shorter time horizons
            })
            
        elif analysis.composite_regime == CompositeRegime.TRANSITIONAL:
            base_pressure.update({
                "contrarian_bias": 0.6,      # Slight contrarian bias
                "momentum_bias": 0.4,        # Reduce momentum
                "volatility_tolerance": 0.4,  # Reduce risk tolerance
                "position_sizing": 0.4,      # Reduce position sizes
                "holding_period": 0.3        # Shorter horizons during transitions
            })
        
        # Apply confidence weighting - higher confidence = stronger adjustments
        if analysis.regime_confidence > self.confidence_threshold:
            # Amplify adjustments for high-confidence regimes
            for key in ["contrarian_bias", "momentum_bias", "volatility_tolerance", "position_sizing", "holding_period"]:
                adjustment = (base_pressure[key] - 0.5) * analysis.regime_confidence
                base_pressure[key] = 0.5 + adjustment
        
        return base_pressure
    
    def get_regime_analysis_summary(self, analysis: RegimeAnalysis) -> Dict[str, Any]:
        """Get comprehensive regime analysis summary."""
        return {
            "composite_regime_analysis": {
                "composite_regime": analysis.composite_regime.value,
                "regime_confidence": analysis.regime_confidence,
                "regime_stability": analysis.regime_stability,
                "is_favorable_conditions": analysis.is_favorable_conditions,
                "is_defensive_conditions": analysis.is_defensive_conditions,
                "is_high_confidence": analysis.is_high_confidence
            },
            "individual_regimes": {
                "sentiment_regime": analysis.sentiment_regime,
                "volatility_regime": analysis.volatility_regime,
                "correlation_regime": analysis.correlation_regime,
                "volume_regime": analysis.volume_regime
            },
            "regime_scores": analysis.regime_scores,
            "configuration": {
                "regime_weights": self.regime_weights,
                "confidence_threshold": self.confidence_threshold,
                "stability_threshold": self.stability_threshold
            },
            "data_quality": {
                "overall_data_quality": analysis.data_quality_score,
                "asset_count": analysis.asset_count
            },
            "metadata": {
                "calculation_timestamp": analysis.calculation_timestamp.isoformat(),
                "engine_version": "1.0.0",
                "integration": "MultiSource_RegimeDetection"
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for composite regime detection engine."""
        try:
            # Check all component health
            component_health = {
                'fear_greed_client': await self._check_fear_greed_health(),
                'volatility_detector': await self.volatility_detector.health_check(),
                'correlation_detector': await self.correlation_detector.health_check(), 
                'volume_detector': await self.volume_detector.health_check()
            }
            
            # Test composite detection
            test_assets = ['BTC', 'ETH']
            composite_healthy = True
            try:
                test_analysis = await self.detect_composite_regime(test_assets, force_refresh=True)
                composite_test_passed = test_analysis.calculation_timestamp is not None
            except Exception as e:
                composite_healthy = False
                composite_test_passed = False
            
            # Determine overall health
            all_components_healthy = all(
                comp.get('status') == 'healthy' 
                for comp in component_health.values()
            )
            
            overall_health = all_components_healthy and composite_healthy
            
            return {
                "status": "healthy" if overall_health else "degraded",
                "component": "CompositeRegimeDetectionEngine",
                "component_health": component_health,
                "composite_detection": "healthy" if composite_healthy else "degraded",
                "regime_history_length": len(self._regime_history),
                "cache_size": len(self._analysis_cache),
                "configuration": {
                    "regime_weights": self.regime_weights,
                    "confidence_threshold": self.confidence_threshold,
                    "stability_threshold": self.stability_threshold
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "CompositeRegimeDetectionEngine",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_fear_greed_health(self) -> Dict[str, Any]:
        """Check fear/greed client health."""
        try:
            # Try to get current index
            await self.fear_greed_client.get_current_index()
            return {"status": "healthy", "component": "fear_greed_client"}
        except Exception as e:
            return {"status": "degraded", "component": "fear_greed_client", "error": str(e)}