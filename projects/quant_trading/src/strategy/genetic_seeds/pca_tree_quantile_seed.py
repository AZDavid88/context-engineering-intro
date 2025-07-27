


from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import warnings

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional



from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

    
    
    
    
    
        
        
        
        
    
        
            
        
        
        
        
        
        
            
        
        
        
        
        
                
        
        
        
        
        
        
        
    
    
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
            
        
        
        
        
        
    
        
            
        
        
        
        
        
        
        
        
        
            
        
            
        
    
        
        
        
        
        
        
        
        
        
    
        
            
        
        
        
    
        
            
        
        
        
        
    
        
"""
PCA Tree Quantile Genetic Seed - Seed #12
This seed implements advanced machine learning using PCA dimensionality reduction,
tree-based models, and quantile-based signal generation. Represents the most
sophisticated ML approach in the genetic seed library.
Key Features:
- PCA-based feature dimensionality reduction
- Random Forest / Decision Tree ensemble
- Quantile-based signal thresholds
- Advanced feature engineering with genetic optimization
"""
@genetic_seed
class PCATreeQuantileSeed(BaseSeed):
    """PCA Tree Quantile seed with advanced ML optimization."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "PCA_Tree_Quantile"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("Advanced ML strategy using PCA dimensionality reduction, "
                "tree-based ensemble models, and quantile-based signal generation. "
                "Genetically optimizes feature engineering and model parameters.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'pca_components',
            'tree_depth',
            'n_estimators',
            'signal_quantile',
            'feature_window'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max)."""
        return {
            'pca_components': (3.0, 20.0),           # Number of PCA components
            'tree_depth': (3.0, 15.0),               # Maximum tree depth
            'n_estimators': (10.0, 200.0),           # Number of trees
            'signal_quantile': (0.1, 0.9),           # Signal threshold quantile
            'feature_window': (50.0, 500.0)          # Feature calculation window
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize PCA Tree Quantile seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.ML_CLASSIFIER
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'pca_components': 8.0,
                'tree_depth': 6.0,
                'n_estimators': 50.0,
                'signal_quantile': 0.7,
                'feature_window': 200.0
            }
        super().__init__(genes, settings)
        # Initialize model and PCA cache
        self._model_cache = {}
        self._pca_cache = {}
        self._feature_cache = {}
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate comprehensive technical indicators for advanced ML.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        feature_window = int(self.genes.parameters['feature_window'])
        # Price-based features
        returns = data['close'].pct_change()
        log_returns = np.log(data['close']).diff()
        # Multiple timeframe moving averages
        ma_periods = [5, 10, 20, 50, 100]
        sma_features = {}
        ema_features = {}
        for period in ma_periods:
            if period <= len(data):
                sma_features[f'sma_{period}'] = data['close'].rolling(window=period).mean()
                ema_features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        # Advanced momentum indicators
        rsi_periods = [7, 14, 21, 28]
        rsi_features = {}
        for period in rsi_periods:
            rsi_features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
        # MACD variations
        macd_fast = [8, 12, 16]
        macd_slow = [21, 26, 34]
        macd_features = {}
        for i, (fast, slow) in enumerate(zip(macd_fast, macd_slow)):
            ema_fast = data['close'].ewm(span=fast).mean()
            ema_slow = data['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=9).mean()
            macd_features[f'macd_{i}'] = macd
            macd_features[f'macd_signal_{i}'] = macd_signal
            macd_features[f'macd_histogram_{i}'] = macd - macd_signal
        # Volatility indicators
        volatility_windows = [10, 20, 50]
        vol_features = {}
        for window in volatility_windows:
            vol_features[f'volatility_{window}'] = returns.rolling(window=window).std()
            vol_features[f'realized_vol_{window}'] = np.sqrt(252) * vol_features[f'volatility_{window}']
        # Price position and range indicators
        lookback_periods = [10, 20, 50, 100]
        position_features = {}
        for period in lookback_periods:
            if period <= len(data):
                high_period = data['high'].rolling(window=period).max()
                low_period = data['low'].rolling(window=period).min()
                position_features[f'price_position_{period}'] = (data['close'] - low_period) / (high_period - low_period)
                position_features[f'range_position_{period}'] = (data['close'] - data['close'].rolling(period).min()) / (data['close'].rolling(period).max() - data['close'].rolling(period).min())
        # Advanced volume indicators (if available)
        volume_features = {}
        if 'volume' in data.columns:
            volume_windows = [5, 20, 50]
            for window in volume_windows:
                vol_ma = data['volume'].rolling(window=window).mean()
                volume_features[f'volume_ratio_{window}'] = data['volume'] / vol_ma
                # VWAP calculations
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                volume_price = typical_price * data['volume']
                volume_features[f'vwap_{window}'] = volume_price.rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
        else:
            # Default volume features
            for window in [5, 20, 50]:
                volume_features[f'volume_ratio_{window}'] = pd.Series(1.0, index=data.index)
                volume_features[f'vwap_{window}'] = data['close']
        # Statistical features
        stat_windows = [20, 50, 100]
        stat_features = {}
        for window in stat_windows:
            if window <= len(data):
                stat_features[f'skewness_{window}'] = returns.rolling(window=window).skew()
                stat_features[f'kurtosis_{window}'] = returns.rolling(window=window).kurt()
                stat_features[f'std_ratio_{window}'] = vol_features[f'volatility_{min(window, 20)}'] / vol_features[f'volatility_{min(window, 20)}'].rolling(window).mean()
        # Trend and momentum features
        trend_windows = [5, 10, 20]
        trend_features = {}
        for window in trend_windows:
            trend_features[f'momentum_{window}'] = data['close'].pct_change(window)
            trend_features[f'acceleration_{window}'] = returns.rolling(window=window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0)
        # Combine all features
        all_indicators = {
            'returns': safe_fillna_zero(returns),
            'log_returns': safe_fillna_zero(log_returns),
        }
        # Add all feature dictionaries
        for feature_dict in [sma_features, ema_features, rsi_features, macd_features,
                           vol_features, position_features, volume_features, 
                           stat_features, trend_features]:
            for key, value in feature_dict.items():
                all_indicators[key] = value.ffill().fillna(0)
        return all_indicators
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    def _engineer_advanced_features(self, indicators: Dict[str, pd.Series]) -> pd.DataFrame:
        """Engineer advanced features for PCA and tree models.
        Args:
            indicators: Technical indicators
        Returns:
            DataFrame of engineered features
        """
        features = pd.DataFrame(index=indicators['returns'].index)
        # Basic features
        basic_features = ['returns', 'log_returns']
        for feature in basic_features:
            if feature in indicators:
                features[feature] = indicators[feature]
        # Moving average ratios
        sma_keys = [k for k in indicators.keys() if k.startswith('sma_')]
        ema_keys = [k for k in indicators.keys() if k.startswith('ema_')]
        # Price ratios to moving averages
        for sma_key in sma_keys:
            if sma_key in indicators:
                close_price = indicators.get('sma_5', features.index.to_series()).iloc[0] if len(features) > 0 else 100
                features[f'price_{sma_key}_ratio'] = close_price / indicators[sma_key]
        # Moving average cross ratios
        if 'sma_5' in indicators and 'sma_20' in indicators:
            features['sma_cross_5_20'] = indicators['sma_5'] / indicators['sma_20']
        if 'sma_20' in indicators and 'sma_50' in indicators:
            features['sma_cross_20_50'] = indicators['sma_20'] / indicators['sma_50']
        # RSI features
        rsi_keys = [k for k in indicators.keys() if k.startswith('rsi_')]
        for rsi_key in rsi_keys:
            if rsi_key in indicators:
                features[rsi_key] = indicators[rsi_key]
                features[f'{rsi_key}_momentum'] = indicators[rsi_key].diff()
                features[f'{rsi_key}_zscore'] = (indicators[rsi_key] - 50) / 20  # Normalize around neutral
        # MACD features
        macd_keys = [k for k in indicators.keys() if k.startswith('macd_') and not k.endswith('_signal') and not k.endswith('_histogram')]
        for macd_key in macd_keys:
            if macd_key in indicators:
                features[macd_key] = indicators[macd_key]
                signal_key = f'{macd_key.replace("macd_", "macd_signal_")}'
                if signal_key in indicators:
                    features[f'{macd_key}_cross'] = indicators[macd_key] - indicators[signal_key]
        # Volatility features
        vol_keys = [k for k in indicators.keys() if 'volatility' in k]
        for vol_key in vol_keys:
            if vol_key in indicators:
                features[vol_key] = indicators[vol_key]
                features[f'{vol_key}_ratio'] = indicators[vol_key] / indicators[vol_key].rolling(50).mean()
        # Position features
        pos_keys = [k for k in indicators.keys() if 'position' in k]
        for pos_key in pos_keys:
            if pos_key in indicators:
                features[pos_key] = indicators[pos_key]
        # Statistical features
        stat_keys = [k for k in indicators.keys() if any(stat in k for stat in ['skewness', 'kurtosis', 'std_ratio'])]
        for stat_key in stat_keys:
            if stat_key in indicators:
                features[stat_key] = indicators[stat_key]
        # Momentum features
        momentum_keys = [k for k in indicators.keys() if 'momentum' in k or 'acceleration' in k]
        for momentum_key in momentum_keys:
            if momentum_key in indicators:
                features[momentum_key] = indicators[momentum_key]
        # Interaction features (genetic feature engineering)
        if 'returns' in features.columns and len(vol_keys) > 0:
            vol_key = vol_keys[0]
            if vol_key in features.columns:
                features['returns_vol_interaction'] = features['returns'] * features[vol_key]
        if len(rsi_keys) > 0 and len(momentum_keys) > 0:
            rsi_key = rsi_keys[0]
            momentum_key = momentum_keys[0]
            if rsi_key in features.columns and momentum_key in features.columns:
                features['rsi_momentum_interaction'] = features[rsi_key] * features[momentum_key]
        # Lagged features
        lag_features = ['returns']
        if len(rsi_keys) > 0:
            lag_features.append(rsi_keys[0])
        for feature in lag_features:
            if feature in features.columns:
                for lag in [1, 2, 3, 5]:
                    features[f'{feature}_lag_{lag}'] = features[feature].shift(lag)
        # Fill NaN values
        features = features.ffill().fillna(0)
        # Remove any infinite values
        features = features.replace([np.inf, -np.inf], 0)
        return features
    def _apply_pca_transformation(self, features: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """Apply PCA dimensionality reduction.
        Args:
            features: Input features
            n_components: Number of PCA components
        Returns:
            PCA-transformed features
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            sklearn_available = True
        except ImportError:
            sklearn_available = False
        if not sklearn_available or len(features) < 50:
            # Fallback: just select top features by variance
            feature_variance = features.var()
            top_features = feature_variance.nlargest(min(n_components, len(features.columns))).index
            return features[top_features]
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(safe_fillna_zero(features))
        # Apply PCA
        pca = PCA(n_components=min(n_components, features.shape[1]))
        features_pca = pca.fit_transform(features_scaled)
        # Create DataFrame with PCA components
        pca_columns = [f'pca_component_{i}' for i in range(features_pca.shape[1])]
        pca_df = pd.DataFrame(features_pca, index=features.index, columns=pca_columns)
        return pca_df
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate advanced ML-based trading signals.
        Args:
            data: OHLCV market data
        Returns:
            Series of trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Get genetic parameters
        pca_components = int(self.genes.parameters['pca_components'])
        tree_depth = int(self.genes.parameters['tree_depth'])
        n_estimators = int(self.genes.parameters['n_estimators'])
        signal_quantile = self.genes.parameters['signal_quantile']
        feature_window = int(self.genes.parameters['feature_window'])
        # Need sufficient data
        if len(data) < 100:
            return pd.Series(0, index=data.index)
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        # Engineer advanced features
        features_df = self._engineer_advanced_features(indicators)
        # Apply PCA transformation
        features_pca = self._apply_pca_transformation(features_df, pca_components)
        # Create targets (future returns)
        future_returns = data['close'].pct_change(5).shift(-5)  # 5-period ahead returns
        # Prepare training data
        valid_mask = ~(features_pca.isna().any(axis=1) | future_returns.isna())
        features_clean = features_pca[valid_mask]
        returns_clean = future_returns[valid_mask]
        if len(features_clean) < 50:
            return pd.Series(0, index=data.index)
        # Use recent data for training
        train_size = min(feature_window, len(features_clean))
        X_train = features_clean.tail(train_size)
        y_train = returns_clean.tail(train_size)
        # Generate predictions using tree-based model
        try:
            predictions = self._train_tree_model(X_train, y_train, features_pca, 
                                               tree_depth, n_estimators)
        except Exception as e:
            warnings.warn(f"Tree model failed: {e}, using quantile fallback")
            # Fallback to quantile-based signals
            momentum = data['close'].pct_change(10)
            upper_quantile = momentum.rolling(100).quantile(signal_quantile)
            lower_quantile = momentum.rolling(100).quantile(1 - signal_quantile)
            signals = pd.Series(0, index=data.index)
            signals[momentum > upper_quantile] = 1
            signals[momentum < lower_quantile] = -1
            return signals
        # Convert predictions to signals using quantile thresholds
        if len(predictions) > 50:
            upper_threshold = predictions.rolling(100).quantile(signal_quantile)
            lower_threshold = predictions.rolling(100).quantile(1 - signal_quantile)
            signals = pd.Series(0, index=data.index)
            signals[predictions > upper_threshold] = 1
            signals[predictions < lower_threshold] = -1
        else:
            signals = pd.Series(0, index=data.index)
        return signals
    def _train_tree_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_full: pd.DataFrame, tree_depth: int, 
                         n_estimators: int) -> pd.Series:
        """Train tree-based model and generate predictions."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.tree import DecisionTreeRegressor
            sklearn_available = True
        except ImportError:
            sklearn_available = False
        if not sklearn_available:
            # Fallback to simple momentum prediction
            if 'pca_component_0' in X_full.columns:
                return X_full['pca_component_0']
            else:
                return pd.Series(0, index=X_full.index)
        # Create ensemble of tree models
        models = []
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=min(n_estimators, 100),  # Cap for speed
            max_depth=tree_depth,
            random_state=42,
            n_jobs=1  # Single thread for stability
        )
        models.append(rf_model)
        # Decision Tree
        dt_model = DecisionTreeRegressor(
            max_depth=tree_depth,
            random_state=42
        )
        models.append(dt_model)
        # Train models and generate predictions
        predictions_ensemble = []
        for model in models:
            try:
                model.fit(safe_fillna_zero(X_train), safe_fillna_zero(y_train))
                pred = model.predict(safe_fillna_zero(X_full))
                predictions_ensemble.append(pred)
            except Exception:
                # If model fails, use zero prediction
                predictions_ensemble.append(np.zeros(len(X_full)))
        # Average ensemble predictions
        if predictions_ensemble:
            final_predictions = np.mean(predictions_ensemble, axis=0)
        else:
            final_predictions = np.zeros(len(X_full))
        return pd.Series(final_predictions, index=X_full.index)
    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance from tree models.
        Args:
            data: Market data
        Returns:
            Dictionary of feature importance scores
        """
        indicators = self.calculate_technical_indicators(data)
        features_df = self._engineer_advanced_features(indicators)
        # Use correlation with future returns as importance proxy
        future_returns = data['close'].pct_change(5).shift(-5)
        importance_scores = {}
        for col in features_df.columns:
            try:
                correlation = features_df[col].corr(future_returns)
                importance_scores[col] = abs(correlation) if not np.isnan(correlation) else 0.0
            except:
                importance_scores[col] = 0.0
        return importance_scores
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate position size based on model confidence and quantile strength.
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
        # For quantile-based signals, signal strength represents how extreme the signal is
        # More extreme signals (further from quantile thresholds) get larger positions
        quantile_strength = abs(signal)
        position_size = base_size * quantile_strength
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        pca_comp = self.genes.parameters.get('pca_components', 8)
        tree_depth = self.genes.parameters.get('tree_depth', 6)
        n_est = self.genes.parameters.get('n_estimators', 50)
        quantile = self.genes.parameters.get('signal_quantile', 0.7)
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"PCATree(pca={pca_comp:.0f},depth={tree_depth:.0f},est={n_est:.0f},q={quantile:.2f}){fitness_str}"