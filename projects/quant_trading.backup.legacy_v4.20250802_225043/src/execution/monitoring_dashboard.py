"""
Monitoring Dashboard - Web interface and visualization for genetic trading monitoring.
Provides dashboard data, UI components, and visualization endpoints for monitoring system.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
from collections import defaultdict, deque
import pandas as pd
import numpy as np

from .monitoring_core import (
    MonitoringSnapshot, SystemHealthMetrics, GeneticEvolutionMetrics, 
    TradingPerformanceMetrics, MonitoringStatus, AlertLevel, AlertCategory
)

logger = logging.getLogger(__name__)


class DashboardInterface:
    """Web dashboard interface for monitoring system."""
    
    def __init__(self, snapshot_history: deque):
        """Initialize dashboard interface."""
        self.snapshot_history = snapshot_history
        self.chart_cache = {}
        self.cache_duration = 60  # seconds
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data."""
        if not self.snapshot_history:
            return {'status': 'no_data', 'message': 'No monitoring data available'}
        
        current_snapshot = self.snapshot_history[-1]
        
        return {
            'status': current_snapshot.status.value,
            'timestamp': current_snapshot.timestamp.isoformat(),
            'system_overview': self._get_system_overview(current_snapshot),
            'genetic_overview': self._get_genetic_overview(current_snapshot),
            'trading_overview': self._get_trading_overview(current_snapshot),
            'alerts_overview': self._get_alerts_overview(current_snapshot),
            'charts': self._get_chart_data(),
            'health_score': self._calculate_health_score(current_snapshot)
        }
    
    def _get_system_overview(self, snapshot: MonitoringSnapshot) -> Dict[str, Any]:
        """Get system health overview."""
        metrics = snapshot.system_metrics
        
        return {
            'cpu_usage': metrics.cpu_usage_percent,
            'memory_usage': metrics.memory_usage_percent,
            'memory_gb': metrics.memory_usage_gb,
            'disk_usage': metrics.disk_usage_percent,
            'network_latency': metrics.network_latency_ms,
            'active_threads': metrics.active_threads,
            'uptime_hours': metrics.uptime_hours,
            'requests_per_second': metrics.requests_per_second,
            'throughput_mbps': metrics.throughput_mbps,
            'error_count': metrics.error_count,
            'status_color': self._get_status_color(metrics.cpu_usage_percent, 70, 90)
        }
    
    def _get_genetic_overview(self, snapshot: MonitoringSnapshot) -> Dict[str, Any]:
        """Get genetic algorithm overview."""
        metrics = snapshot.genetic_metrics
        
        return {
            'generation': metrics.generation,
            'population_size': metrics.population_size,
            'best_fitness': metrics.best_fitness,
            'average_fitness': metrics.average_fitness,
            'diversity_score': metrics.diversity_score,
            'convergence_rate': metrics.convergence_rate,
            'evaluation_time': metrics.evaluation_time,
            'memory_usage_gb': metrics.memory_usage_gb,
            'active_strategies': metrics.active_strategies,
            'cache_hit_rate': metrics.cache_hit_rate,
            'status_color': self._get_status_color(metrics.diversity_score, 0.3, 0.1, inverted=True)
        }
    
    def _get_trading_overview(self, snapshot: MonitoringSnapshot) -> Dict[str, Any]:
        """Get trading performance overview."""
        metrics = snapshot.trading_metrics
        
        return {
            'total_return': metrics.total_return,
            'daily_return': metrics.daily_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'current_drawdown': metrics.current_drawdown,
            'total_trades': metrics.total_trades,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'execution_success_rate': metrics.execution_success_rate,
            'avg_slippage': metrics.avg_slippage,
            'avg_latency_ms': metrics.avg_latency_ms,
            'active_positions': metrics.active_positions,
            'total_exposure': metrics.total_exposure,
            'status_color': self._get_status_color(metrics.execution_success_rate, 0.95, 0.90, inverted=True)
        }
    
    def _get_alerts_overview(self, snapshot: MonitoringSnapshot) -> Dict[str, Any]:
        """Get alerts overview."""
        alerts = snapshot.active_alerts
        
        alert_counts = {level.value: 0 for level in AlertLevel}
        category_counts = {category.value: 0 for category in AlertCategory}
        
        for alert in alerts:
            alert_counts[alert.level.value] += 1
            category_counts[alert.category.value] += 1
        
        recent_alerts = sorted(alerts, key=lambda a: a.last_triggered, reverse=True)[:5]
        
        return {
            'total_alerts': len(alerts),
            'by_level': alert_counts,
            'by_category': category_counts,
            'recent_alerts': [
                {
                    'id': alert.alert_id,
                    'level': alert.level.value,
                    'category': alert.category.value,
                    'title': alert.title,
                    'message': alert.message,
                    'source': alert.source_component,
                    'triggered': alert.last_triggered.isoformat(),
                    'count': alert.trigger_count
                } for alert in recent_alerts
            ],
            'status_color': self._get_alert_status_color(alerts)
        }
    
    def _get_chart_data(self) -> Dict[str, Any]:
        """Get chart data for dashboard visualization."""
        cache_key = f"charts_{len(self.snapshot_history)}"
        
        # Check cache
        if (cache_key in self.chart_cache and 
            time.time() - self.chart_cache[cache_key]['timestamp'] < self.cache_duration):
            return self.chart_cache[cache_key]['data']
        
        # Generate chart data
        chart_data = {
            'system_metrics': self._generate_system_chart_data(),
            'genetic_metrics': self._generate_genetic_chart_data(),
            'trading_metrics': self._generate_trading_chart_data(),
            'alert_timeline': self._generate_alert_timeline_data()
        }
        
        # Cache results
        self.chart_cache[cache_key] = {
            'data': chart_data,
            'timestamp': time.time()
        }
        
        return chart_data
    
    def _generate_system_chart_data(self) -> Dict[str, Any]:
        """Generate system metrics chart data."""
        if len(self.snapshot_history) < 2:
            return {'timestamps': [], 'cpu': [], 'memory': [], 'network_latency': []}
        
        # Get last 50 data points for performance
        recent_snapshots = list(self.snapshot_history)[-50:]
        
        timestamps = [s.timestamp.isoformat() for s in recent_snapshots]
        cpu_data = [s.system_metrics.cpu_usage_percent for s in recent_snapshots]
        memory_data = [s.system_metrics.memory_usage_percent for s in recent_snapshots]
        latency_data = [s.system_metrics.network_latency_ms for s in recent_snapshots]
        
        return {
            'timestamps': timestamps,
            'cpu_usage': cpu_data,
            'memory_usage': memory_data,
            'network_latency': latency_data,
            'trend_cpu': self._calculate_trend(cpu_data),
            'trend_memory': self._calculate_trend(memory_data)
        }
    
    def _generate_genetic_chart_data(self) -> Dict[str, Any]:
        """Generate genetic algorithm chart data."""
        if len(self.snapshot_history) < 2:
            return {'timestamps': [], 'fitness': [], 'diversity': []}
        
        recent_snapshots = list(self.snapshot_history)[-50:]
        
        timestamps = [s.timestamp.isoformat() for s in recent_snapshots]
        fitness_data = [s.genetic_metrics.best_fitness for s in recent_snapshots]
        diversity_data = [s.genetic_metrics.diversity_score for s in recent_snapshots]
        evaluation_time = [s.genetic_metrics.evaluation_time for s in recent_snapshots]
        
        return {
            'timestamps': timestamps,
            'best_fitness': fitness_data,
            'diversity_score': diversity_data,
            'evaluation_time': evaluation_time,
            'trend_fitness': self._calculate_trend(fitness_data),
            'trend_diversity': self._calculate_trend(diversity_data)
        }
    
    def _generate_trading_chart_data(self) -> Dict[str, Any]:
        """Generate trading performance chart data."""
        if len(self.snapshot_history) < 2:
            return {'timestamps': [], 'returns': [], 'drawdown': []}
        
        recent_snapshots = list(self.snapshot_history)[-50:]
        
        timestamps = [s.timestamp.isoformat() for s in recent_snapshots]
        return_data = [s.trading_metrics.daily_return for s in recent_snapshots]
        drawdown_data = [s.trading_metrics.current_drawdown for s in recent_snapshots]
        sharpe_data = [s.trading_metrics.sharpe_ratio for s in recent_snapshots]
        
        return {
            'timestamps': timestamps,
            'daily_returns': return_data,
            'drawdown': drawdown_data,
            'sharpe_ratio': sharpe_data,
            'trend_returns': self._calculate_trend(return_data),
            'trend_sharpe': self._calculate_trend(sharpe_data)
        }
    
    def _generate_alert_timeline_data(self) -> Dict[str, Any]:
        """Generate alert timeline chart data."""
        # Collect alerts from recent snapshots
        alert_timeline = []
        
        for snapshot in list(self.snapshot_history)[-20:]:  # Last 20 snapshots
            for alert in snapshot.active_alerts:
                alert_timeline.append({
                    'timestamp': snapshot.timestamp.isoformat(),
                    'level': alert.level.value,
                    'category': alert.category.value,
                    'title': alert.title,
                    'count': alert.trigger_count
                })
        
        # Group by time periods
        hourly_counts = defaultdict(lambda: defaultdict(int))
        for alert in alert_timeline:
            hour = alert['timestamp'][:13]  # YYYY-MM-DDTHH
            hourly_counts[hour][alert['level']] += 1
        
        return {
            'timeline': alert_timeline,
            'hourly_summary': dict(hourly_counts),
            'total_alerts': len(alert_timeline)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend for chart data."""
        if len(values) < 3:
            return {'direction': 'stable', 'slope': 0.0, 'confidence': 0.0}
        
        # Use numpy for trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Handle NaN values
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return {'direction': 'stable', 'slope': 0.0, 'confidence': 0.0}
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Linear regression
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        
        # Calculate R-squared
        y_pred = slope * x_clean + intercept
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'slope': float(slope),
            'confidence': float(max(0, r_squared))
        }
    
    def _calculate_health_score(self, snapshot: MonitoringSnapshot) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0
        
        # System health penalties
        if snapshot.system_metrics.cpu_usage_percent > 90:
            score -= 30
        elif snapshot.system_metrics.cpu_usage_percent > 70:
            score -= 15
        
        if snapshot.system_metrics.memory_usage_percent > 90:
            score -= 30
        elif snapshot.system_metrics.memory_usage_percent > 70:
            score -= 15
        
        # Trading performance penalties
        if snapshot.trading_metrics.execution_success_rate < 0.9:
            score -= 20
        elif snapshot.trading_metrics.execution_success_rate < 0.95:
            score -= 10
        
        # Alert penalties
        for alert in snapshot.active_alerts:
            if alert.level == AlertLevel.EMERGENCY:
                score -= 15
            elif alert.level == AlertLevel.CRITICAL:
                score -= 10
            elif alert.level == AlertLevel.WARNING:
                score -= 5
        
        return max(0.0, min(100.0, score))
    
    def _get_status_color(self, value: float, warning_threshold: float, 
                         critical_threshold: float, inverted: bool = False) -> str:
        """Get status color based on thresholds."""
        if inverted:
            if value <= critical_threshold:
                return 'red'
            elif value <= warning_threshold:
                return 'yellow'
            else:
                return 'green'
        else:
            if value >= critical_threshold:
                return 'red'
            elif value >= warning_threshold:
                return 'yellow'
            else:
                return 'green'
    
    def _get_alert_status_color(self, alerts: List) -> str:
        """Get overall alert status color."""
        if not alerts:
            return 'green'
        
        for alert in alerts:
            if alert.level == AlertLevel.EMERGENCY:
                return 'red'
            elif alert.level == AlertLevel.CRITICAL:
                return 'red'
        
        for alert in alerts:
            if alert.level == AlertLevel.WARNING:
                return 'yellow'
        
        return 'green'


class DataVisualization:
    """Data visualization and chart generation for monitoring dashboard."""
    
    def __init__(self):
        """Initialize data visualization."""
        self.supported_chart_types = [
            'line', 'area', 'bar', 'scatter', 'heatmap', 'gauge'
        ]
    
    def generate_chart_config(self, chart_type: str, data: Dict[str, Any], 
                            title: str, **kwargs) -> Dict[str, Any]:
        """Generate chart configuration for frontend."""
        if chart_type not in self.supported_chart_types:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        base_config = {
            'type': chart_type,
            'title': title,
            'data': data,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'animation': {'duration': 300}
            }
        }
        
        # Chart-specific configurations
        if chart_type == 'line':
            base_config['options'].update({
                'scales': {
                    'x': {'type': 'time'},
                    'y': {'beginAtZero': True}
                },
                'elements': {
                    'point': {'radius': 2},
                    'line': {'tension': 0.2}
                }
            })
        elif chart_type == 'gauge':
            base_config['options'].update({
                'circumference': 180,
                'rotation': 270,
                'cutout': '80%'
            })
        
        # Apply custom options
        base_config['options'].update(kwargs.get('options', {}))
        
        return base_config
    
    def create_system_health_charts(self, system_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create system health visualization charts."""
        charts = []
        
        if system_data.get('timestamps'):
            charts.append(self.generate_chart_config(
                'line',
                {
                    'labels': system_data['timestamps'],
                    'datasets': [
                        {
                            'label': 'CPU Usage (%)',
                            'data': system_data['cpu_usage'],
                            'borderColor': 'rgb(255, 99, 132)'
                        },
                        {
                            'label': 'Memory Usage (%)',
                            'data': system_data['memory_usage'],
                            'borderColor': 'rgb(54, 162, 235)'
                        }
                    ]
                },
                'System Resource Usage'
            ))
        
        return charts
    
    def create_genetic_evolution_charts(self, genetic_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create genetic algorithm evolution charts."""
        charts = []
        
        # Fitness evolution chart
        if genetic_data.get('timestamps'):
            charts.append(self.generate_chart_config(
                'line',
                {
                    'labels': genetic_data['timestamps'],
                    'datasets': [{
                        'label': 'Best Fitness',
                        'data': genetic_data['best_fitness'],
                        'borderColor': 'rgb(255, 205, 86)',
                        'fill': False
                    }]
                },
                'Genetic Algorithm Fitness Evolution'
            ))
        
        # Diversity and evaluation time dual-axis chart
        if genetic_data.get('diversity_score') and genetic_data.get('evaluation_time'):
            charts.append(self.generate_chart_config(
                'line',
                {
                    'labels': genetic_data['timestamps'],
                    'datasets': [
                        {
                            'label': 'Diversity Score',
                            'data': genetic_data['diversity_score'],
                            'borderColor': 'rgb(153, 102, 255)',
                            'yAxisID': 'y'
                        },
                        {
                            'label': 'Evaluation Time (s)',
                            'data': genetic_data['evaluation_time'],
                            'borderColor': 'rgb(255, 159, 64)',
                            'yAxisID': 'y1'
                        }
                    ]
                },
                'Diversity and Performance',
                options={
                    'scales': {
                        'y': {'type': 'linear', 'display': True, 'position': 'left'},
                        'y1': {'type': 'linear', 'display': True, 'position': 'right'}
                    }
                }
            ))
        
        return charts
    
    def create_trading_performance_charts(self, trading_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create comprehensive trading performance charts with dual-axis analysis."""
        charts = []
        
        if trading_data.get('timestamps'):
            # Basic daily returns chart
            charts.append(self.generate_chart_config(
                'line',
                {
                    'labels': trading_data['timestamps'],
                    'datasets': [{
                        'label': 'Daily Returns (%)',
                        'data': trading_data['daily_returns'],
                        'borderColor': 'rgb(75, 192, 192)',
                        'fill': False
                    }]
                },
                'Daily Trading Returns'
            ))
            
            # CRITICAL: Returns vs Drawdown dual-axis chart
            if trading_data.get('drawdown'):
                charts.append(self.generate_chart_config(
                    'line',
                    {
                        'labels': trading_data['timestamps'],
                        'datasets': [
                            {
                                'label': 'Daily Returns (%)',
                                'data': trading_data['daily_returns'],
                                'borderColor': 'rgb(75, 192, 192)',
                                'yAxisID': 'y',
                                'fill': False
                            },
                            {
                                'label': 'Current Drawdown (%)',
                                'data': trading_data['drawdown'],
                                'borderColor': 'rgb(255, 99, 132)',
                                'yAxisID': 'y1',
                                'fill': True,
                                'backgroundColor': 'rgba(255, 99, 132, 0.1)'
                            }
                        ]
                    },
                    'Returns vs Drawdown Analysis',
                    options={
                        'scales': {
                            'y': {
                                'type': 'linear',
                                'display': True,
                                'position': 'left',
                                'title': {'display': True, 'text': 'Returns (%)'}
                            },
                            'y1': {
                                'type': 'linear',
                                'display': True,
                                'position': 'right',
                                'title': {'display': True, 'text': 'Drawdown (%)'},
                                'grid': {'drawOnChartArea': False}
                            }
                        }
                    }
                ))
            
            # Sharpe ratio evolution chart
            if trading_data.get('sharpe_ratio'):
                charts.append(self.generate_chart_config(
                    'line',
                    {
                        'labels': trading_data['timestamps'],
                        'datasets': [{
                            'label': 'Sharpe Ratio',
                            'data': trading_data['sharpe_ratio'],
                            'borderColor': 'rgb(255, 205, 86)',
                            'fill': False
                        }]
                    },
                    'Sharpe Ratio Evolution'
                ))
        
        return charts
    
    def create_alert_timeline_chart(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create alert timeline visualization with severity breakdown."""
        if not alert_data.get('timeline'):
            return {}
        
        # Aggregate by hour and severity level
        hourly_data = defaultdict(lambda: defaultdict(int))
        for alert in alert_data['timeline']:
            hour = alert['timestamp'][:13]
            level = alert.get('level', 'info')
            hourly_data[hour][level] += alert.get('count', 1)
        
        # Prepare datasets for each severity level
        hours = sorted(hourly_data.keys())
        severity_colors = {
            'emergency': 'rgba(220, 53, 69, 0.8)',
            'critical': 'rgba(255, 99, 132, 0.8)', 
            'warning': 'rgba(255, 193, 7, 0.8)',
            'info': 'rgba(54, 162, 235, 0.8)'
        }
        
        datasets = []
        for level, color in severity_colors.items():
            data = [hourly_data[hour].get(level, 0) for hour in hours]
            if any(data):  # Only include levels that have data
                datasets.append({
                    'label': level.title(),
                    'data': data,
                    'backgroundColor': color
                })
        
        return self.generate_chart_config(
            'bar',
            {
                'labels': hours,
                'datasets': datasets
            },
            'Alert Timeline by Severity',
            options={
                'scales': {
                    'x': {'stacked': True},
                    'y': {'stacked': True}
                }
            }
        )


# Utility classes for dashboard support
