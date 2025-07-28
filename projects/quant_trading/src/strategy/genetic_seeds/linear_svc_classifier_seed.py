
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import warnings

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
Linear SVC Classifier Genetic Seed - Seed #11
This seed implements machine learning-based signal generation using Support Vector
Classification with genetic feature engineering and hyperparameter optimization.
Key Features:
- Genetic feature selection and engineering
- SVM hyperparameter evolution (C, kernel parameters)
- Ensemble model creation with voting
- Cross-validation scoring optimization
"""
@genetic_seed
class LinearSVCClassifierSeed(BaseSeed):
    """Linear SVC classifier seed with genetic ML optimization."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "Linear_SVC_Classifier"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("Machine learning classifier using Support Vector Classification. "
                "Genetically optimizes feature engineering, hyperparameters, and "
                "ensemble configuration for robust signal generation.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'lookback_window',
            'feature_count',
            'regularization',
            'ensemble_size',
            'cross_validation'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max)."""
        return {
            'lookback_window': (20.0, 200.0),        # Training data window
            'feature_count': (3.0, 15.0),            # Number of features to use
            'regularization': (0.01, 100.0),         # SVM C parameter
            'ensemble_size': (1.0, 5.0),             # Number of models in ensemble
            'cross_validation': (3.0, 10.0)          # CV folds
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize Linear SVC classifier seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.ML_CLASSIFIER
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'lookback_window': 100.0,
                'feature_count': 8.0,
                'regularization': 1.0,
                'ensemble_size': 3.0,
                'cross_validation': 5.0
            }
        super().__init__(genes, settings)
        # Initialize model cache
        self._model_cache = {}
        self._feature_cache = {}
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators for ML feature engineering.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Basic price features
        returns = data['close'].pct_change()
        log_returns = np.log(data['close']).diff()
        # Moving averages
        sma_5 = data['close'].rolling(window=5).mean()
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        # Momentum indicators
        rsi_14 = self._calculate_rsi(data['close'], 14)
        rsi_21 = self._calculate_rsi(data['close'], 21)
        # MACD
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        # Volatility indicators
        volatility = returns.rolling(window=20).std()
        atr = self._calculate_atr(data, 14)
        # Volume indicators (if available)
        if 'volume' in data.columns:
            volume_sma = data['volume'].rolling(window=20).mean()
            volume_ratio = data['volume'] / volume_sma
            price_volume = data['close'] * data['volume']
            vwap = price_volume.rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
        else:
            volume_ratio = pd.Series(1.0, index=data.index)
            vwap = data['close']
        # Price position indicators
        high_20 = data['high'].rolling(window=20).max()
        low_20 = data['low'].rolling(window=20).min()
        price_position = (data['close'] - low_20) / (high_20 - low_20)
        # Trend indicators
        sma_slope = sma_20.diff()
        price_above_sma = (data['close'] > sma_20).astype(int)
        return {
            'returns': safe_fillna_zero(returns),
            'log_returns': safe_fillna_zero(log_returns),
            'sma_5': sma_5.fillna(data['close']),
            'sma_20': sma_20.fillna(data['close']),
            'sma_50': sma_50.fillna(data['close']),
            'ema_12': ema_12.fillna(data['close']),
            'ema_26': ema_26.fillna(data['close']),
            'rsi_14': rsi_14.fillna(50),
            'rsi_21': rsi_21.fillna(50),
            'macd': safe_fillna_zero(macd),
            'macd_signal': safe_fillna_zero(macd_signal),
            'macd_histogram': safe_fillna_zero(macd_histogram),
            'volatility': volatility.fillna(0.02),
            'atr': atr.fillna(0.02),
            'volume_ratio': volume_ratio.fillna(1.0),
            'vwap': vwap.fillna(data['close']),
            'price_position': price_position.fillna(0.5),
            'sma_slope': safe_fillna_zero(sma_slope),
            'price_above_sma': safe_fillna_zero(price_above_sma)
        }
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    def _engineer_features(self, indicators: Dict[str, pd.Series]) -> pd.DataFrame:
        """Engineer ML features from technical indicators.
        Args:
            indicators: Technical indicators
        Returns:
            DataFrame of engineered features
        """
        features = pd.DataFrame(index=indicators['returns'].index)
        # Basic features
        features['returns'] = indicators['returns']
        features['log_returns'] = indicators['log_returns']
        features['volatility'] = indicators['volatility']
        features['rsi_14'] = indicators['rsi_14']
        features['macd'] = indicators['macd']
        features['price_position'] = indicators['price_position']
        features['volume_ratio'] = indicators['volume_ratio']
        # Ratio features
        features['sma_ratio_5_20'] = indicators['sma_5'] / indicators['sma_20']
        features['sma_ratio_20_50'] = indicators['sma_20'] / indicators['sma_50']
        features['ema_ratio'] = indicators['ema_12'] / indicators['ema_26']
        features['price_vwap_ratio'] = indicators['vwap'] / indicators['sma_20']
        # Momentum features
        features['rsi_momentum'] = indicators['rsi_14'].diff()
        features['macd_momentum'] = indicators['macd_histogram']
        features['price_momentum_3'] = indicators['returns'].rolling(3).sum()
        features['price_momentum_5'] = indicators['returns'].rolling(5).sum()
        # Lagged features
        for lag in [1, 2, 3]:
            features[f'returns_lag_{lag}'] = indicators['returns'].shift(lag)
            features[f'rsi_lag_{lag}'] = indicators['rsi_14'].shift(lag)
        # Volatility features
        features['vol_ratio'] = indicators['volatility'] / indicators['volatility'].rolling(50).mean()
        features['atr_ratio'] = indicators['atr'] / indicators['sma_20']
        # Trend features
        features['trend_strength'] = indicators['sma_slope'] / indicators['sma_20']
        features['price_above_sma'] = indicators['price_above_sma']
        # Cross-feature interactions
        features['rsi_vol_interaction'] = features['rsi_14'] * features['volatility']
        features['momentum_vol_interaction'] = features['price_momentum_3'] * features['volatility']
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        return features
    def _create_target_labels(self, data: pd.DataFrame, prediction_horizon: int = 5) -> pd.Series:
        """Create target labels for classification.
        Args:
            data: Market data
            prediction_horizon: Periods ahead to predict
        Returns:
            Target labels: 1 (up), 0 (down)
        """
        # Future return calculation
        future_returns = data['close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
        # Binary classification: 1 if positive return, 0 if negative
        labels = (future_returns > 0).astype(int)
        return labels
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate ML-based trading signals.
        Args:
            data: OHLCV market data
        Returns:
            Series of trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Get genetic parameters
        lookback_window = int(self.genes.parameters['lookback_window'])
        feature_count = int(self.genes.parameters['feature_count'])
        regularization = self.genes.parameters['regularization']
        ensemble_size = int(self.genes.parameters['ensemble_size'])
        cv_folds = int(self.genes.parameters['cross_validation'])
        # Need sufficient data for training
        if len(data) < lookback_window + 10:
            return pd.Series(0, index=data.index)
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        # Engineer features
        features_df = self._engineer_features(indicators)
        # Select top features based on genetic parameter
        feature_cols = features_df.columns[:feature_count]
        features_selected = features_df[feature_cols]
        # Create target labels
        targets = self._create_target_labels(data)
        # Prepare training data
        valid_data_mask = ~(features_selected.isna().any(axis=1) | targets.isna())
        features_clean = features_selected[valid_data_mask]
        targets_clean = targets[valid_data_mask]
        if len(features_clean) < 50:  # Need minimum data for training
            return pd.Series(0, index=data.index)
        # Use recent data for training
        train_end = len(features_clean)
        train_start = max(0, train_end - lookback_window)
        X_train = features_clean.iloc[train_start:train_end]
        y_train = targets_clean.iloc[train_start:train_end]
        # Generate predictions
        try:
            predictions = self._train_and_predict(X_train, y_train, features_selected,
                                                regularization, ensemble_size, cv_folds)
        except Exception as e:
            # Fallback to simple momentum signals if ML fails
            warnings.warn(f"ML prediction failed: {e}, using fallback")
            momentum = data['close'].pct_change(5)
            predictions = pd.Series(np.where(momentum > 0, 1, -1), index=data.index)
        # Convert predictions to trading signals
        signals = pd.Series(0, index=data.index)
        if isinstance(predictions, pd.Series) and len(predictions) > 0:
            # Map predictions to signals: 1 -> buy, 0 -> sell
            signals = predictions.map({1: 1, 0: -1}).fillna(0)
        return signals
    def _train_and_predict(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_full: pd.DataFrame, regularization: float,
                          ensemble_size: int, cv_folds: int) -> pd.Series:
        """Train SVC model and generate predictions."""
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import VotingClassifier
            sklearn_available = True
        except ImportError:
            sklearn_available = False
        if not sklearn_available:
            # Fallback to simple threshold-based prediction
            momentum_feature = X_full.get('returns', X_full.iloc[:, 0])
            predictions = (momentum_feature > momentum_feature.median()).astype(int)
            return predictions
        # Prepare training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # Create ensemble models
        models = []
        for i in range(ensemble_size):
            model = SVC(
                C=regularization,
                kernel='linear',  # Linear SVC for speed
                probability=True,
                random_state=42 + i
            )
            models.append((f'svc_{i}', model))
        # Ensemble classifier
        if len(models) > 1:
            ensemble = VotingClassifier(models, voting='soft')
        else:
            ensemble = models[0][1]
        # Cross-validation scoring
        try:
            cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')
            cv_mean = cv_scores.mean()
            # Only proceed if model shows reasonable performance
            if cv_mean < 0.45:  # Below random chance
                # Fallback to momentum
                momentum_feature = X_full.get('returns', X_full.iloc[:, 0])
                return (momentum_feature > 0).astype(int)
        except Exception:
            # CV failed, use simple prediction
            momentum_feature = X_full.get('returns', X_full.iloc[:, 0])
            return (momentum_feature > 0).astype(int)
        # Train final model
        ensemble.fit(X_train_scaled, y_train)
        # Scale full dataset and predict
        X_full_scaled = scaler.transform(safe_fillna_zero(X_full))
        predictions = ensemble.predict(X_full_scaled)
        return pd.Series(predictions, index=X_full.index)
    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance scores (simplified for SVC).
        Args:
            data: Market data
        Returns:
            Dictionary of feature importance scores
        """
        indicators = self.calculate_technical_indicators(data)
        features_df = self._engineer_features(indicators)
        # For SVC, use correlation with future returns as proxy for importance
        targets = self._create_target_labels(data)
        importance_scores = {}
        for col in features_df.columns:
            try:
                correlation = features_df[col].corr(targets)
                importance_scores[col] = abs(correlation) if not np.isnan(correlation) else 0.0
            except:
                importance_scores[col] = 0.0
        return importance_scores
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate position size based on ML model confidence.
        Args:
            data: Current market data
            signal: Signal strength (-1 to 1)
        Returns:
            Position size as percentage of capital
        """
        if abs(signal) < self.genes.filter_threshold:
            return 0.0
        # Base position size from genes
        base_size = self.genes.position_size
        # For ML models, signal strength represents confidence
        # Higher confidence = larger position size
        confidence_multiplier = abs(signal)
        position_size = base_size * confidence_multiplier
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        lookback = self.genes.parameters.get('lookback_window', 100)
        features = self.genes.parameters.get('feature_count', 8)
        reg = self.genes.parameters.get('regularization', 1.0)
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"LinearSVC(win={lookback:.0f},feat={features:.0f},C={reg:.2f}){fitness_str}"
