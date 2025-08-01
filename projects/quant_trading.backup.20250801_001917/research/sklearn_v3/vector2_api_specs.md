# Vector 2: sklearn API Specifications and ML Model Documentation

## LinearSVC (Support Vector Classifier) API Specification

### Class Definition
```python
sklearn.svm.LinearSVC(
    penalty='l2',
    loss='squared_hinge',
    *,
    dual='auto',
    tol=0.0001,
    C=1.0,
    multi_class='ovr',
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    verbose=0,
    random_state=None,
    max_iter=1000
)
```

### Key Features for Genetic Seed Implementation
- **Linear Support Vector Classification** using liblinear backend
- **Scalable to large datasets** - better than SVC for high sample counts
- **Flexible penalty and loss functions** - ideal for genetic parameter evolution
- **Multi-class support** via one-vs-rest scheme

### Genetic Algorithm Optimization Parameters

#### 1. Penalty and Regularization
```python
genetic_params = {
    'penalty': ['l1', 'l2'],                    # Sparsity control
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],       # Regularization strength
    'loss': ['hinge', 'squared_hinge'],        # Loss function selection
}
```

#### 2. Dual Optimization Control
```python
dual_optimization = {
    'dual': ['auto', True, False],             # Algorithm selection
    'tol': [1e-5, 1e-4, 1e-3],               # Convergence tolerance
    'max_iter': [500, 1000, 2000, 5000],     # Iteration limits
}
```

#### 3. Class Balancing (Critical for Trading Signals)
```python
class_handling = {
    'class_weight': [None, 'balanced'],        # Handle imbalanced datasets
    'multi_class': ['ovr', 'crammer_singer'], # Multi-class strategy
}
```

### Trading-Specific Implementation Patterns

#### Pattern 1: Binary Trading Signal Classification
```python
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class LinearSVCTradingSeed:
    def __init__(self, genetic_params):
        self.C = genetic_params['C']
        self.penalty = genetic_params['penalty']
        self.loss = genetic_params['loss']
        self.class_weight = genetic_params.get('class_weight', None)
        
        # Essential for trading features (price ratios, indicators)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', LinearSVC(
                C=self.C,
                penalty=self.penalty,
                loss=self.loss,
                class_weight=self.class_weight,
                random_state=42,
                max_iter=2000
            ))
        ])
    
    def fit(self, X_features, y_signals):
        """
        X_features: Technical indicators, price features, volume data
        y_signals: [-1, 0, 1] for [sell, hold, buy] signals
        """
        return self.model.fit(X_features, y_signals)
    
    def predict_signal(self, X_features):
        """Generate trading signals"""
        return self.model.predict(X_features)
    
    def predict_confidence(self, X_features):
        """Get decision function confidence for position sizing"""
        return self.model.decision_function(X_features)
```

#### Pattern 2: Feature Importance for Signal Quality
```python
def get_feature_importance(self):
    """Extract coefficients for feature analysis"""
    svc_model = self.model.named_steps['svc']
    feature_weights = svc_model.coef_[0]  # Binary classification
    
    # Genetic algorithm can evolve feature selection based on weights
    return {
        'feature_coefficients': feature_weights,
        'intercept': svc_model.intercept_[0],
        'support_vectors_count': len(svc_model.support_vectors_),
    }
```

#### Key Performance Characteristics
- **Memory Efficiency**: Uses sparse internal representation
- **Speed**: liblinear implementation optimized for linear kernels
- **Reproducibility**: random_state parameter for consistent genetic evolution
- **Convergence Monitoring**: n_iter_ attribute tracks actual iterations

---

## PCA (Principal Component Analysis) API Specification

### Class Definition
```python
sklearn.decomposition.PCA(
    n_components=None,
    *,
    copy=True,
    whiten=False,
    svd_solver='auto',
    tol=0.0,
    iterated_power='auto',
    n_oversamples=10,
    power_iteration_normalizer='auto',
    random_state=None
)
```

### Key Features for Genetic Seed Implementation
- **Dimensionality Reduction** via Singular Value Decomposition
- **Variance Explanation** through principal components
- **Multiple SVD Solvers** for different data characteristics
- **Whitening Support** for uncorrelated outputs

### Genetic Algorithm Optimization Parameters

#### 1. Component Selection and Dimensionality
```python
genetic_params = {
    'n_components': [0.8, 0.9, 0.95, 0.99],   # Variance retention ratios
    'n_components_int': [5, 10, 15, 20, 25],  # Fixed component counts
    'whiten': [True, False],                   # Decorrelation
}
```

#### 2. SVD Solver Optimization
```python
svd_optimization = {
    'svd_solver': ['auto', 'full', 'arpack', 'randomized'],
    'tol': [0.0, 1e-6, 1e-4],                 # ARPACK tolerance
    'iterated_power': ['auto', 1, 2, 3, 5],   # Randomized solver power
    'n_oversamples': [5, 10, 15, 20],         # Randomized solver oversampling
}
```

### Trading-Specific Implementation Patterns

#### Pattern 1: Feature Reduction for Technical Indicators
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class PCAFeatureReductionSeed:
    def __init__(self, genetic_params):
        self.n_components = genetic_params.get('n_components', 0.95)
        self.whiten = genetic_params.get('whiten', False)
        self.svd_solver = genetic_params.get('svd_solver', 'auto')
        
        self.scaler = StandardScaler()
        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=42
        )
        
    def fit_transform_features(self, technical_indicators):
        """
        technical_indicators: Matrix of [RSI, MACD, SMA, EMA, ATR, etc.]
        Returns: Reduced dimensional feature space
        """
        # Standardize features (critical for PCA)
        scaled_features = self.scaler.fit_transform(technical_indicators)
        
        # Apply PCA transformation
        reduced_features = self.pca.fit_transform(scaled_features)
        
        return reduced_features
    
    def get_feature_analysis(self):
        """Analyze PCA results for genetic fitness evaluation"""
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_),
            'n_components_used': self.pca.n_components_,
            'singular_values': self.pca.singular_values_,
            'principal_components': self.pca.components_,
        }
    
    def transform_new_data(self, new_indicators):
        """Transform new market data using fitted PCA"""
        scaled_new = self.scaler.transform(new_indicators)
        return self.pca.transform(scaled_new)
```

#### Pattern 2: Inverse Transform for Signal Reconstruction
```python
def reconstruct_signals(self, pca_features):
    """Reconstruct original feature space from PCA components"""
    # Transform back to original scaled space
    reconstructed_scaled = self.pca.inverse_transform(pca_features)
    
    # Inverse scale to original feature space
    reconstructed_original = self.scaler.inverse_transform(reconstructed_scaled)
    
    return reconstructed_original

def calculate_reconstruction_error(self, original_features):
    """Measure information loss for genetic fitness"""
    reduced = self.fit_transform_features(original_features)
    reconstructed = self.reconstruct_signals(reduced)
    
    # Mean squared reconstruction error
    mse = np.mean((original_features - reconstructed) ** 2)
    
    return {
        'reconstruction_mse': mse,
        'information_retention': 1 - (mse / np.var(original_features)),
        'compression_ratio': self.pca.n_components_ / original_features.shape[1]
    }
```

---

## QuantileRegressor API Specification

### Class Definition
```python
sklearn.linear_model.QuantileRegressor(
    *,
    quantile=0.5,
    alpha=1.0,
    fit_intercept=True,
    solver='highs',
    solver_options=None
)
```

### Key Features for Genetic Seed Implementation
- **Quantile-based predictions** for risk-aware trading
- **Robust to outliers** via pinball loss optimization
- **L1 regularization** like Lasso for feature selection
- **Multiple solver options** for optimization

### Genetic Algorithm Optimization Parameters

#### 1. Quantile Selection for Risk Profiles
```python
genetic_params = {
    'quantile': [0.1, 0.25, 0.5, 0.75, 0.9],  # Different risk percentiles
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],    # L1 regularization strength
    'fit_intercept': [True, False],             # Bias term inclusion
}
```

#### 2. Solver Optimization
```python
solver_optimization = {
    'solver': ['highs-ds', 'highs-ipm', 'highs', 'revised simplex'],
    'solver_options': [
        None,
        {'presolve': True},
        {'time_limit': 300},
        {'method': 'dual'}
    ]
}
```

### Trading-Specific Implementation Patterns

#### Pattern 1: Multi-Quantile Risk Analysis
```python
from sklearn.linear_model import QuantileRegressor
import numpy as np

class QuantileRiskAnalysisSeed:
    def __init__(self, genetic_params):
        self.quantiles = genetic_params.get('quantiles', [0.1, 0.5, 0.9])
        self.alpha = genetic_params.get('alpha', 1.0)
        self.solver = genetic_params.get('solver', 'highs')
        
        # Create ensemble of quantile regressors
        self.quantile_models = {}
        for q in self.quantiles:
            self.quantile_models[q] = QuantileRegressor(
                quantile=q,
                alpha=self.alpha,
                solver=self.solver,
                fit_intercept=True
            )
    
    def fit_risk_models(self, X_features, y_returns):
        """
        X_features: Technical indicators, market features
        y_returns: Target returns/price changes
        """
        fitted_models = {}
        for quantile, model in self.quantile_models.items():
            fitted_models[quantile] = model.fit(X_features, y_returns)
        
        return fitted_models
    
    def predict_risk_envelope(self, X_features):
        """Generate risk-based predictions for multiple quantiles"""
        predictions = {}
        for quantile, model in self.quantile_models.items():
            predictions[quantile] = model.predict(X_features)
        
        return predictions
    
    def calculate_risk_metrics(self, predictions):
        """Calculate trading risk metrics from quantile predictions"""
        q10 = predictions[0.1]  # 10th percentile (downside risk)
        q50 = predictions[0.5]  # Median (central estimate)
        q90 = predictions[0.9]  # 90th percentile (upside potential)
        
        return {
            'expected_return': q50,
            'downside_risk': q50 - q10,
            'upside_potential': q90 - q50,
            'risk_asymmetry': (q90 - q50) / (q50 - q10),
            'confidence_interval': q90 - q10,
            'risk_adjusted_signal': np.where(
                (q90 - q50) > 2 * (q50 - q10),  # Asymmetric upside
                1,   # Buy signal
                np.where(
                    (q50 - q10) > 2 * (q90 - q50),  # Asymmetric downside
                    -1,  # Sell signal
                    0    # Hold
                )
            )
        }
```

#### Pattern 2: Dynamic Position Sizing Based on Quantiles
```python
def calculate_quantile_position_size(self, risk_metrics, base_position=1.0):
    """Size positions based on quantile risk analysis"""
    risk_adjusted_size = []
    
    for i in range(len(risk_metrics['expected_return'])):
        expected = risk_metrics['expected_return'][i]
        downside = risk_metrics['downside_risk'][i]
        upside = risk_metrics['upside_potential'][i]
        
        # Kelly Criterion adapted for quantile predictions
        if downside > 0:  # Avoid division by zero
            kelly_fraction = (expected * upside) / (downside ** 2)
            # Constrain position size
            position_size = np.clip(kelly_fraction * base_position, 0, 2.0)
        else:
            position_size = 0.0
            
        risk_adjusted_size.append(position_size)
    
    return np.array(risk_adjusted_size)
```

---

## Combined PCA + Tree + Quantile Implementation Pattern

### Composite Genetic Seed Architecture
```python
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

class PCATreeQuantileSeed:
    def __init__(self, genetic_params):
        # PCA parameters
        self.pca_components = genetic_params.get('pca_components', 0.95)
        self.pca_whiten = genetic_params.get('pca_whiten', False)
        
        # Tree parameters  
        self.tree_max_depth = genetic_params.get('tree_max_depth', 10)
        self.tree_min_samples_split = genetic_params.get('tree_min_samples_split', 5)
        
        # Quantile parameters
        self.quantiles = genetic_params.get('quantiles', [0.25, 0.5, 0.75])
        self.quantile_alpha = genetic_params.get('quantile_alpha', 1.0)
        
        # Build composite pipeline
        self.build_pipeline()
    
    def build_pipeline(self):
        """Construct PCA -> Tree -> Quantile pipeline"""
        # Stage 1: PCA feature reduction
        self.pca_stage = PCA(
            n_components=self.pca_components,
            whiten=self.pca_whiten,
            random_state=42
        )
        
        # Stage 2: Decision tree feature learning
        self.tree_stage = DecisionTreeRegressor(
            max_depth=self.tree_max_depth,
            min_samples_split=self.tree_min_samples_split,
            random_state=42
        )
        
        # Stage 3: Quantile prediction ensemble
        self.quantile_stages = {}
        for q in self.quantiles:
            self.quantile_stages[q] = QuantileRegressor(
                quantile=q,
                alpha=self.quantile_alpha,
                solver='highs'
            )
    
    def fit_composite_model(self, X_technical_features, y_target_returns):
        """Train the complete PCA -> Tree -> Quantile pipeline"""
        # Stage 1: Reduce dimensionality with PCA
        X_pca = self.pca_stage.fit_transform(X_technical_features)
        
        # Stage 2: Generate tree-based features
        tree_features = self.generate_tree_features(X_pca, y_target_returns)
        
        # Stage 3: Train quantile regressors on tree features
        self.fit_quantile_ensemble(tree_features, y_target_returns)
    
    def generate_tree_features(self, X_pca, y_target):
        """Create tree-based feature representations"""
        # Fit decision tree
        self.tree_stage.fit(X_pca, y_target)
        
        # Extract tree-based features
        tree_features = {
            'leaf_indices': self.tree_stage.apply(X_pca),  # Leaf node assignments
            'decision_path': self.tree_stage.decision_path(X_pca).toarray(),  # Path features
            'tree_predictions': self.tree_stage.predict(X_pca),  # Tree predictions
        }
        
        # Combine features for quantile regression
        combined_features = np.column_stack([
            X_pca,  # Original PCA features
            tree_features['tree_predictions'].reshape(-1, 1),  # Tree predictions
            tree_features['leaf_indices'].reshape(-1, 1),  # Leaf assignments
        ])
        
        return combined_features
    
    def fit_quantile_ensemble(self, tree_features, y_target):
        """Train multiple quantile regressors"""
        for quantile, model in self.quantile_stages.items():
            model.fit(tree_features, y_target)
    
    def predict_composite(self, X_new_features):
        """Generate predictions using full pipeline"""
        # Transform through PCA
        X_pca_new = self.pca_stage.transform(X_new_features)
        
        # Generate tree features
        tree_predictions = self.tree_stage.predict(X_pca_new)
        leaf_indices = self.tree_stage.apply(X_pca_new)
        
        combined_features_new = np.column_stack([
            X_pca_new,
            tree_predictions.reshape(-1, 1),
            leaf_indices.reshape(-1, 1),
        ])
        
        # Generate quantile predictions
        quantile_predictions = {}
        for quantile, model in self.quantile_stages.items():
            quantile_predictions[quantile] = model.predict(combined_features_new)
        
        return quantile_predictions
    
    def get_genetic_fitness_metrics(self):
        """Extract metrics for genetic algorithm evaluation"""
        return {
            'pca_explained_variance': np.sum(self.pca_stage.explained_variance_ratio_),
            'pca_n_components': self.pca_stage.n_components_,
            'tree_depth': self.tree_stage.get_depth(),
            'tree_n_leaves': self.tree_stage.get_n_leaves(),
            'tree_feature_importance': self.tree_stage.feature_importances_,
            'quantile_coefficients': {
                q: model.coef_ for q, model in self.quantile_stages.items()
            },
            'quantile_intercepts': {
                q: model.intercept_ for q, model in self.quantile_stages.items()
            }
        }
```

## Integration with Genetic Algorithm Framework

### Fitness Function Integration
```python
def calculate_ml_seed_fitness(seed_model, X_test, y_test):
    """Evaluate ML seed performance for genetic algorithm"""
    predictions = seed_model.predict_composite(X_test)
    
    # Multi-objective fitness components
    fitness_components = {
        'prediction_accuracy': calculate_prediction_accuracy(predictions, y_test),
        'risk_adjusted_return': calculate_risk_adjusted_metrics(predictions, y_test),
        'model_complexity': calculate_complexity_penalty(seed_model),
        'robustness_score': calculate_robustness_metrics(seed_model, X_test, y_test)
    }
    
    # Combined fitness score
    fitness_score = (
        0.4 * fitness_components['prediction_accuracy'] +
        0.3 * fitness_components['risk_adjusted_return'] +
        0.2 * fitness_components['robustness_score'] +
        0.1 * (1 - fitness_components['model_complexity'])  # Penalty for complexity
    )
    
    return fitness_score, fitness_components
```

### Parameter Space for Genetic Evolution
```python
ML_GENETIC_PARAMETER_SPACE = {
    'linear_svc_params': {
        'C': (0.001, 1000.0),           # Log-uniform distribution
        'penalty': ['l1', 'l2'],
        'loss': ['hinge', 'squared_hinge'],
        'class_weight': [None, 'balanced'],
    },
    'pca_params': {
        'n_components': (0.80, 0.99),   # Variance retention ratio
        'whiten': [True, False],
        'svd_solver': ['auto', 'full', 'randomized'],
    },
    'quantile_params': {
        'quantiles': [
            [0.1, 0.5, 0.9],
            [0.25, 0.5, 0.75],
            [0.05, 0.25, 0.5, 0.75, 0.95]
        ],
        'alpha': (0.001, 10.0),         # Log-uniform distribution
    },
    'tree_params': {
        'max_depth': (3, 20),           # Integer range
        'min_samples_split': (2, 20),   # Integer range
        'min_samples_leaf': (1, 10),    # Integer range
    }
}
```

This comprehensive API specification provides all the necessary details for implementing sophisticated ML-based genetic seeds that leverage sklearn's robust machine learning algorithms for trading signal generation and risk management.