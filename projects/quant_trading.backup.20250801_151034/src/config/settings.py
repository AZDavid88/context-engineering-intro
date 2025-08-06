"""
Core Configuration Management for Quant Trading Organism

This module provides Pydantic-based configuration management for the entire
trading system, including Hyperliquid API settings, genetic algorithm parameters,
risk management rules, and environment-specific configurations.

Based on research from:
- Hyperliquid Python SDK V3 Comprehensive Research
- DEAP Genetic Programming Framework
- Vectorbt Backtesting Engine Integration
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, SecretStr
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Trading environment configuration."""
    DEVELOPMENT = "development"
    TESTNET = "testnet"
    MAINNET = "mainnet"


class LogLevel(str, Enum):
    """Logging level configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HyperliquidConfig(BaseModel):
    """Hyperliquid exchange configuration."""
    
    # API Endpoints - from research
    mainnet_url: str = "https://api.hyperliquid.xyz"
    testnet_url: str = "https://api.hyperliquid-testnet.xyz"
    websocket_url: str = "wss://api.hyperliquid.xyz/ws"
    websocket_testnet_url: str = "wss://api.hyperliquid-testnet.xyz/ws"
    
    # Authentication
    api_key: Optional[SecretStr] = None
    private_key: Optional[SecretStr] = None
    wallet_address: Optional[str] = None
    
    # Rate Limiting - from research (1200 requests per minute = 20 per second)
    max_requests_per_second: int = Field(default=20, ge=1, le=200000)
    max_websocket_subscriptions: int = Field(default=50, ge=1, le=100)
    
    # Connection Settings
    connection_timeout: float = Field(default=30.0, ge=1.0, le=120.0)
    reconnect_attempts: int = Field(default=5, ge=1, le=20)
    reconnect_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    
    # VPN Requirements - critical for Hyperliquid
    require_vpn: bool = True
    vpn_provider: str = "nordvpn"
    
    @field_validator('wallet_address')
    @classmethod
    def validate_wallet_address(cls, v):
        """Validate Ethereum wallet address format."""
        if v and not (v.startswith('0x') and len(v) == 42):
            raise ValueError('Invalid wallet address format')
        return v


class TradingConfig(BaseModel):
    """Core trading system configuration."""
    
    # Capital Management
    initial_capital: float = Field(default=10000.0, ge=100.0, le=1000000.0)
    max_position_size: float = Field(default=0.25, ge=0.01, le=0.5)  # 25% max per position
    max_strategy_allocation: float = Field(default=0.4, ge=0.1, le=0.5)  # 40% max per strategy
    
    # Risk Management - from PRP requirements
    max_daily_drawdown: float = Field(default=0.02, ge=0.001, le=0.1)  # 2% daily max
    max_total_drawdown: float = Field(default=0.1, ge=0.01, le=0.3)   # 10% total max
    sharpe_threshold: float = Field(default=2.0, ge=0.5, le=10.0)     # Required Sharpe > 2
    
    # Transaction Costs - realistic modeling
    maker_fee: float = Field(default=0.0002, ge=0.0, le=0.01)  # 0.02% maker fee
    taker_fee: float = Field(default=0.0005, ge=0.0, le=0.01)  # 0.05% taker fee
    slippage: float = Field(default=0.001, ge=0.0, le=0.01)    # 0.1% slippage
    
    # Asset Selection
    target_assets: List[str] = Field(default=["BTC", "ETH", "SOL", "AVAX", "MATIC"])
    max_assets: int = Field(default=10, ge=1, le=50)
    min_volume_usd: float = Field(default=1000000.0, ge=10000.0)  # $1M min daily volume
    
    # Strategy Correlation Limits
    max_strategy_correlation: float = Field(default=0.7, ge=0.1, le=0.95)
    
    # Data Processing
    bar_duration_minutes: int = Field(default=1, ge=1, le=1440)  # 1 minute to 1 day bars


class GeneticAlgorithmConfig(BaseModel):
    """Genetic algorithm parameters for strategy evolution."""
    
    # Population Parameters - from DEAP research
    population_size: int = Field(default=200, ge=20, le=1000)
    max_generations: int = Field(default=50, ge=5, le=200)
    elite_size: int = Field(default=20, ge=1, le=50)
    
    # Evolution Parameters
    crossover_probability: float = Field(default=0.8, ge=0.1, le=1.0)
    mutation_probability: float = Field(default=0.2, ge=0.01, le=0.5)
    tournament_size: int = Field(default=7, ge=2, le=20)
    
    # Strategy Complexity Control - prevent overfitting
    max_tree_height: int = Field(default=17, ge=3, le=25)
    max_tree_size: int = Field(default=50, ge=5, le=200)
    
    # Multi-objective Optimization Weights
    fitness_weights: Dict[str, float] = Field(default={
        "sharpe_ratio": 1.0,
        "max_drawdown": -1.0,  # Minimize drawdown
        "win_rate": 0.5,
        "consistency": 0.3
    })
    
    # Parallel Processing
    use_multiprocessing: bool = True
    max_workers: Optional[int] = None  # Auto-detect CPU cores
    
    # Validation Periods
    train_split: float = Field(default=0.6, ge=0.4, le=0.8)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.3)
    test_split: float = Field(default=0.2, ge=0.1, le=0.3)
    
    @field_validator('fitness_weights')
    @classmethod
    def validate_fitness_weights(cls, v):
        """Ensure fitness weights sum validation."""
        if not v:
            raise ValueError("Fitness weights cannot be empty")
        return v
    
    @field_validator('train_split', 'validation_split', 'test_split')
    @classmethod
    def validate_splits(cls, v):
        """Ensure data splits sum to 1.0."""
        return v


class BacktestingConfig(BaseModel):
    """Backtesting engine configuration."""
    
    # Data Parameters
    lookback_days: int = Field(default=365, ge=30, le=1825)  # 1 year default
    timeframe: str = Field(default="1h", pattern=r"^(1m|5m|15m|1h|4h|1d)$")
    
    # Vectorbt Parameters - from research
    initial_cash: float = Field(default=10000.0, ge=1000.0, le=1000000.0)
    commission: float = Field(default=0.001, ge=0.0, le=0.01)  # 0.1% total cost
    
    # Validation Requirements
    min_trades: int = Field(default=30, ge=10, le=1000)
    min_trade_frequency: float = Field(default=0.1, ge=0.01, le=10.0)  # Trades per day
    
    # Performance Thresholds
    min_sharpe_ratio: float = Field(default=2.0, ge=0.5, le=10.0)
    max_drawdown_threshold: float = Field(default=0.15, ge=0.05, le=0.5)
    min_win_rate: float = Field(default=0.4, ge=0.2, le=0.8)
    
    # Walk-forward Analysis
    walk_forward_periods: int = Field(default=4, ge=2, le=12)
    walk_forward_step: int = Field(default=30, ge=1, le=90)  # Days


class MarketRegimeConfig(BaseModel):
    """Market regime detection configuration."""
    
    # Fear & Greed Index - from research
    fear_greed_api_url: str = "https://api.alternative.me/fng/"
    fear_threshold: int = Field(default=25, ge=0, le=50)    # Extreme fear
    greed_threshold: int = Field(default=75, ge=50, le=100) # Extreme greed
    
    # Volatility Regimes
    volatility_lookback: int = Field(default=30, ge=7, le=90)
    high_volatility_threshold: float = Field(default=0.03, ge=0.01, le=0.1)  # 3% daily
    
    # Trend Detection
    trend_lookback: int = Field(default=20, ge=5, le=100)
    trend_threshold: float = Field(default=0.02, ge=0.005, le=0.05)  # 2% move


class MonitoringConfig(BaseModel):
    """System monitoring and alerting configuration."""
    
    # Dashboard Settings
    refresh_interval: float = Field(default=5.0, ge=1.0, le=60.0)  # Seconds
    max_display_strategies: int = Field(default=20, ge=5, le=100)
    
    # Alerting Thresholds
    drawdown_alert_threshold: float = Field(default=0.05, ge=0.01, le=0.2)  # 5%
    pnl_alert_threshold: float = Field(default=0.1, ge=0.01, le=0.5)       # 10%
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_file_max_size: int = Field(default=100, ge=1, le=1000)  # MB
    log_file_backup_count: int = Field(default=5, ge=1, le=20)


class DatabaseConfig(BaseModel):
    """Database and storage configuration."""
    
    # DuckDB for analytics (Phase 1)
    duckdb_path: str = "data/quant_trading.duckdb"
    
    # TimescaleDB for scaling (Phase 2+)
    timescale_host: Optional[str] = None
    timescale_port: int = Field(default=5432, ge=1024, le=65535)
    timescale_database: str = "quant_trading"
    timescale_user: Optional[str] = None
    timescale_password: Optional[SecretStr] = None
    
    # Data retention
    historical_data_retention_days: int = Field(default=730, ge=90, le=2555)  # 2 years
    strategy_data_retention_days: int = Field(default=365, ge=30, le=1825)    # 1 year
    
    # Parquet storage
    parquet_base_path: str = "data/parquet"
    parquet_compression: str = Field(default="snappy", pattern=r"^(snappy|gzip|brotli)$")


class SupervisorConfig(BaseModel):
    """Process management configuration."""
    
    # Process settings
    unix_http_server_file: str = "/tmp/supervisor.sock"
    inet_http_server_port: int = Field(default=9001, ge=8000, le=9999)
    
    # Process priorities (lower = higher priority)
    data_ingestion_priority: int = Field(default=100, ge=1, le=1000)
    strategy_evolution_priority: int = Field(default=200, ge=1, le=1000)
    execution_engine_priority: int = Field(default=50, ge=1, le=1000)
    monitoring_priority: int = Field(default=300, ge=1, le=1000)
    
    # Auto-restart settings
    autorestart: bool = True
    startretries: int = Field(default=3, ge=0, le=10)
    
    # Logging
    stdout_logfile_maxbytes: str = "50MB"
    stdout_logfile_backups: int = Field(default=10, ge=1, le=50)


class Settings(BaseSettings):
    """Main configuration class combining all sub-configurations."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=False)
    
    # Sub-configurations
    hyperliquid: HyperliquidConfig = Field(default_factory=HyperliquidConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    genetic_algorithm: GeneticAlgorithmConfig = Field(default_factory=GeneticAlgorithmConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    market_regime: MarketRegimeConfig = Field(default_factory=MarketRegimeConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        
        # Environment variable prefixes
        env_prefix = "QUANT_"
    
    def __init__(self, **kwargs):
        """Initialize settings with directory creation."""
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.project_root / self.data_dir,
            self.project_root / self.logs_dir,
            self.project_root / self.data_dir / "parquet",
            self.project_root / self.data_dir / "duckdb",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.MAINNET
    
    @property
    def is_testnet(self) -> bool:
        """Check if running in testnet environment."""
        return self.environment == Environment.TESTNET
    
    @property
    def hyperliquid_api_url(self) -> str:
        """Get appropriate Hyperliquid API URL based on environment."""
        if self.is_production:
            return self.hyperliquid.mainnet_url
        return self.hyperliquid.testnet_url
    
    @property
    def hyperliquid_websocket_url(self) -> str:
        """Get appropriate Hyperliquid WebSocket URL based on environment."""
        if self.is_production:
            return self.hyperliquid.websocket_url
        return self.hyperliquid.websocket_testnet_url
    
    def get_data_splits(self) -> Dict[str, float]:
        """Get data splitting configuration for backtesting."""
        ga_config = self.genetic_algorithm
        return {
            "train": ga_config.train_split,
            "validation": ga_config.validation_split,
            "test": ga_config.test_split
        }
    
    def get_fitness_weights(self) -> Dict[str, float]:
        """Get genetic algorithm fitness weights."""
        return self.genetic_algorithm.fitness_weights
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate complete configuration and return status."""
        validation_results = {}
        
        try:
            # Validate data splits sum to 1.0
            splits = self.get_data_splits()
            splits_sum = sum(splits.values())
            validation_results["data_splits"] = abs(splits_sum - 1.0) < 0.001
            
            # Validate fitness weights are reasonable
            weights = self.get_fitness_weights()
            validation_results["fitness_weights"] = len(weights) > 0
            
            # Validate trading parameters are reasonable
            validation_results["trading_config"] = (
                0 < self.trading.max_position_size <= 0.5 and
                0 < self.trading.max_strategy_allocation <= 0.5 and
                self.trading.sharpe_threshold >= 2.0
            )
            
            # Validate GA parameters
            validation_results["genetic_algorithm"] = (
                self.genetic_algorithm.population_size >= 20 and
                self.genetic_algorithm.max_generations >= 5 and
                0 < self.genetic_algorithm.crossover_probability <= 1.0
            )
            
            # Validate backtesting configuration
            validation_results["backtesting"] = (
                self.backtesting.min_sharpe_ratio >= 2.0 and
                self.backtesting.max_drawdown_threshold <= 0.15
            )
            
        except Exception as e:
            validation_results["error"] = str(e)
        
        return validation_results


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment and config files."""
    global settings
    settings = Settings()
    return settings


if __name__ == "__main__":
    """Configuration validation and testing."""
    
    # Load and validate settings
    config = get_settings()
    
    print("=== Quant Trading Organism Configuration ===")
    print(f"Environment: {config.environment}")
    print(f"Debug Mode: {config.debug}")
    print(f"Project Root: {config.project_root}")
    print(f"Hyperliquid API: {config.hyperliquid_api_url}")
    print(f"WebSocket URL: {config.hyperliquid_websocket_url}")
    
    print("\n=== Configuration Validation ===")
    validation_results = config.validate_configuration()
    for check, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check}: {status}")
    
    print("\n=== Trading Parameters ===")
    print(f"Initial Capital: ${config.trading.initial_capital:,.2f}")
    print(f"Max Position Size: {config.trading.max_position_size:.1%}")
    print(f"Sharpe Threshold: {config.trading.sharpe_threshold}")
    print(f"Max Drawdown: {config.trading.max_total_drawdown:.1%}")
    
    print("\n=== Genetic Algorithm ===")
    print(f"Population Size: {config.genetic_algorithm.population_size}")
    print(f"Max Generations: {config.genetic_algorithm.max_generations}")
    print(f"Fitness Weights: {config.genetic_algorithm.fitness_weights}")
    
    print("\n=== Data Splits ===")
    splits = config.get_data_splits()
    for split_name, split_value in splits.items():
        print(f"{split_name.title()}: {split_value:.1%}")
    
    all_valid = all(validation_results.values())
    print(f"\n=== Overall Status: {'✅ READY' if all_valid else '❌ CONFIGURATION ERRORS'} ===")