"""
Configuration Management Package

Provides centralized configuration management for the quantitative trading system.
Handles environment-specific settings, API keys, and system parameters.
"""

from .settings import get_settings, Settings

__all__ = ['get_settings', 'Settings']