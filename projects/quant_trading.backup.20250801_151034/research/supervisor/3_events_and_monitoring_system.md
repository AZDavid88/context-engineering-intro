# Supervisor - Events and Monitoring System

**Source**: http://supervisord.org/events.html  
**Extracted**: 2025-07-25  
**Purpose**: Advanced event system for trading system monitoring, alerting, and automated responses

## Event System Overview

Supervisor's event system enables real-time monitoring and automated responses to process state changes. This is crucial for trading systems where immediate response to failures is essential.

### Key Concepts

- **Event Listeners**: Specialized programs that subscribe to supervisor events
- **Event Notifications**: Messages sent when specific conditions occur
- **Event Types**: Predefined categories of events (process state, ticks, communications)
- **Event Pools**: Groups of listener processes handling events

## Event Listener Configuration

### Basic Event Listener Setup

```ini
[eventlistener:trading_monitor]
command = python /app/monitoring/trading_monitor.py
events = PROCESS_STATE,TICK_60
numprocs = 1
autostart = true
autorestart = unexpected
stdout_logfile = /app/logs/trading_monitor.log
stderr_logfile = /app/logs/trading_monitor_error.log
```

### Multiple Event Listeners for Trading System

```ini
# Process failure monitor
[eventlistener:failure_monitor]
command = python /app/monitoring/failure_monitor.py
events = PROCESS_STATE_FATAL,PROCESS_STATE_EXITED
numprocs = 2
buffer_size = 100
priority = 10

# Performance monitor (runs every minute)
[eventlistener:performance_monitor]
command = python /app/monitoring/performance_monitor.py
events = TICK_60
numprocs = 1
buffer_size = 50

# Communication event monitor
[eventlistener:comm_monitor]
command = python /app/monitoring/comm_monitor.py
events = PROCESS_COMMUNICATION
numprocs = 1

# Real-time heartbeat monitor (every 5 seconds)
[eventlistener:heartbeat_monitor]
command = python /app/monitoring/heartbeat_monitor.py
events = TICK_5
numprocs = 1
```

## Event Listener States and Protocol

### Event Listener States

1. **ACKNOWLEDGED**: Listener has processed previous event
2. **READY**: Available to receive new events
3. **BUSY**: Currently processing an event

### Event Notification Protocol

```python
# Event header format (space-separated tokens)
# ver:3.0 server:supervisor serial:21 pool:listener poolserial:10 eventname:TICK_5 len:54

# Event payload follows (len bytes)
# processname:genetic_engine groupname:trading_system from_state:STOPPED
```

## Critical Event Types for Trading Systems

### Process State Events

#### PROCESS_STATE_FATAL
**Use Case**: Critical process failed to start after maximum retries

```python
# Event body example
processname:risk_manager groupname:trading_system from_state:BACKOFF

# Trading response: 
# - Stop all trading immediately
# - Send critical alert
# - Activate backup risk management
```

#### PROCESS_STATE_EXITED
**Use Case**: Process exited (expected or unexpected)

```python
# Event body example  
processname:genetic_engine groupname:trading_system from_state:RUNNING expected:0 pid:2766

# Trading response:
# - Check if exit was expected (expected:1) or crash (expected:0)
# - Log exit reason and restart if needed
# - Preserve strategy state if genetic engine crashed
```

#### PROCESS_STATE_RUNNING
**Use Case**: Process successfully started

```python
# Event body example
processname:hyperliquid_feed groupname:trading_system from_state:STARTING pid:2766

# Trading response:
# - Log successful startup
# - Enable dependent processes
# - Resume trading if all critical processes running
```

### Tick Events for Periodic Monitoring

#### TICK_5 (Every 5 seconds)
**Use Case**: High-frequency health checks

```python
# Event body
when:1705063880

# Trading uses:
# - Check WebSocket connection health
# - Monitor order execution latency
# - Verify position synchronization
# - Check VPN connection status
```

#### TICK_60 (Every minute)
**Use Case**: Regular system monitoring

```python
# Trading uses:
# - Calculate performance metrics
# - Check memory usage
# - Validate strategy signals
# - Update market regime detection
# - Log system health summary
```

#### TICK_3600 (Every hour)
**Use Case**: Long-term maintenance

```python
# Trading uses:
# - Generate hourly reports
# - Rotate log files
# - Backup strategy states
# - Update research database
# - Send daily summary emails
```

### Process Communication Events

#### PROCESS_COMMUNICATION_STDOUT/STDERR
**Use Case**: Processes send structured data to supervisor

```python
# Process sends message between special tags:
print("<!--XSUPERVISOR:BEGIN-->")
print("ALERT:HIGH_DRAWDOWN:genetic_engine:5.2%")
print("<!--XSUPERVISOR:END-->")

# Event body
processname:genetic_engine groupname:trading_system pid:2766
ALERT:HIGH_DRAWDOWN:genetic_engine:5.2%
```

## Event Listener Implementation Examples

### 1. Failure Monitor (Critical for Trading)

```python
#!/usr/bin/env python3
"""
Failure Monitor - Responds to process failures
Handles PROCESS_STATE_FATAL and PROCESS_STATE_EXITED events
"""

import sys
import json
import logging
from datetime import datetime
import requests

def write_stdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()

def write_stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

def send_alert(process_name, event_type, details):
    """Send critical alert to monitoring system"""
    alert = {
        'timestamp': datetime.utcnow().isoformat(),
        'severity': 'CRITICAL',
        'process': process_name,
        'event': event_type,
        'details': details,
        'system': 'trading_system'
    }
    
    # Send to monitoring endpoint
    try:
        requests.post('http://monitoring.internal/alerts', json=alert, timeout=5)
    except Exception as e:
        write_stderr(f"Failed to send alert: {e}\n")

def handle_process_fatal(headers, payload):
    """Handle FATAL process state - critical trading system failure"""
    data = dict([x.split(':') for x in payload.strip().split()])
    process_name = data['processname']
    
    write_stderr(f"CRITICAL: Process {process_name} entered FATAL state\n")
    
    # Critical processes that require immediate action
    critical_processes = ['risk_manager', 'hyperliquid_feed', 'order_executor']
    
    if process_name in critical_processes:
        send_alert(process_name, 'PROCESS_FATAL', {
            'message': f'Critical process {process_name} failed to start',
            'from_state': data.get('from_state'),
            'action_required': 'immediate_intervention'
        })
        
        # Stop all trading if risk manager fails
        if process_name == 'risk_manager':
            write_stderr("EMERGENCY: Risk manager failed - stopping all trading\n")
            # Implementation would call supervisorctl to stop trading processes
    
    return 'OK'

def handle_process_exited(headers, payload):
    """Handle process exit events"""
    data = dict([x.split(':') for x in payload.strip().split()])
    process_name = data['processname']
    expected = data.get('expected', '0')
    pid = data.get('pid')
    
    if expected == '0':  # Unexpected exit
        write_stderr(f"WARNING: Process {process_name} (PID {pid}) exited unexpectedly\n")
        
        # Log unexpected exits for analysis
        send_alert(process_name, 'UNEXPECTED_EXIT', {
            'pid': pid,
            'from_state': data.get('from_state'),
            'requires_investigation': True
        })
    
    return 'OK'

def main():
    while True:
        write_stdout('READY\n')
        
        # Read event header
        line = sys.stdin.readline()
        headers = dict([x.split(':') for x in line.split()])
        
        # Read event payload
        data = sys.stdin.read(int(headers['len']))
        
        # Process event based on type
        event_name = headers['eventname']
        
        try:
            if event_name == 'PROCESS_STATE_FATAL':
                result = handle_process_fatal(headers, data)
            elif event_name == 'PROCESS_STATE_EXITED':
                result = handle_process_exited(headers, data)
            else:
                result = 'OK'  # Ignore other events
                
            write_stdout(f'RESULT {len(result)}\n{result}')
            
        except Exception as e:
            write_stderr(f"Error processing event: {e}\n")
            write_stdout('RESULT 4\nFAIL')

if __name__ == '__main__':
    main()
```

### 2. Performance Monitor (Resource Tracking)

```python
#!/usr/bin/env python3
"""
Performance Monitor - Tracks system performance metrics
Runs on TICK_60 events (every minute)
"""

import sys
import psutil
import json
from datetime import datetime

def write_stdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()

def write_stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

def collect_system_metrics():
    """Collect system performance metrics"""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'network_io': dict(psutil.net_io_counters()._asdict()),
        'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
    }

def collect_process_metrics():
    """Collect metrics for specific trading processes"""
    trading_processes = [
        'genetic_engine', 'risk_manager', 'hyperliquid_feed', 
        'order_executor', 'backtest_worker'
    ]
    
    process_metrics = {}
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            for trading_proc in trading_processes:
                if trading_proc in cmdline:
                    process_metrics[trading_proc] = {
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                        'status': proc.status()
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return process_metrics

def check_performance_thresholds(system_metrics, process_metrics):
    """Check if any performance thresholds are exceeded"""
    alerts = []
    
    # System-level alerts
    if system_metrics['cpu_percent'] > 80:
        alerts.append({
            'type': 'HIGH_CPU',
            'value': system_metrics['cpu_percent'],
            'threshold': 80
        })
    
    if system_metrics['memory_percent'] > 85:
        alerts.append({
            'type': 'HIGH_MEMORY',
            'value': system_metrics['memory_percent'],
            'threshold': 85
        })
    
    # Process-level alerts
    for proc_name, metrics in process_metrics.items():
        if metrics['memory_mb'] > 1000:  # 1GB memory limit
            alerts.append({
                'type': 'PROCESS_HIGH_MEMORY',
                'process': proc_name,
                'value': metrics['memory_mb'],
                'threshold': 1000
            })
    
    return alerts

def main():
    while True:
        write_stdout('READY\n')
        
        # Read event header
        line = sys.stdin.readline()
        headers = dict([x.split(':') for x in line.split()])
        
        # Read event payload
        data = sys.stdin.read(int(headers['len']))
        
        try:
            if headers['eventname'] == 'TICK_60':
                # Collect performance metrics
                system_metrics = collect_system_metrics()
                process_metrics = collect_process_metrics()
                
                # Check thresholds
                alerts = check_performance_thresholds(system_metrics, process_metrics)
                
                # Log metrics
                metrics_log = {
                    'system': system_metrics,
                    'processes': process_metrics,
                    'alerts': alerts
                }
                
                write_stderr(f"Performance metrics: {json.dumps(metrics_log, indent=2)}\n")
                
                # Send alerts if any
                if alerts:
                    for alert in alerts:
                        write_stderr(f"PERFORMANCE ALERT: {alert}\n")
            
            write_stdout('RESULT 2\nOK')
            
        except Exception as e:
            write_stderr(f"Error in performance monitor: {e}\n")
            write_stdout('RESULT 4\nFAIL')

if __name__ == '__main__':
    main()
```

### 3. Communication Monitor (Strategy Signals)

```python
#!/usr/bin/env python3
"""
Communication Monitor - Processes strategy communications
Handles PROCESS_COMMUNICATION events from trading processes
"""

import sys
import json
from datetime import datetime

def write_stdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()

def write_stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

def process_strategy_signal(process_name, message):
    """Process strategy communication signals"""
    try:
        # Parse structured messages from strategies
        if message.startswith('SIGNAL:'):
            parts = message.split(':')
            signal_type = parts[1]
            strategy_id = parts[2]
            data = ':'.join(parts[3:])
            
            write_stderr(f"Strategy signal from {process_name}: {signal_type} on {strategy_id}\n")
            
            # Handle different signal types
            if signal_type == 'ENTRY':
                handle_entry_signal(strategy_id, data)
            elif signal_type == 'EXIT':
                handle_exit_signal(strategy_id, data)
            elif signal_type == 'RISK_ALERT':
                handle_risk_alert(strategy_id, data)
            elif signal_type == 'PERFORMANCE_UPDATE':
                handle_performance_update(strategy_id, data)
                
        elif message.startswith('ALERT:'):
            parts = message.split(':')
            alert_level = parts[1]
            alert_type = parts[2]
            details = ':'.join(parts[3:])
            
            write_stderr(f"Process alert from {process_name}: {alert_level} - {alert_type}: {details}\n")
            
    except Exception as e:
        write_stderr(f"Error processing strategy signal: {e}\n")

def handle_entry_signal(strategy_id, data):
    """Handle entry signal from genetic algorithm"""
    # Log entry signal for audit trail
    entry_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'strategy_id': strategy_id,
        'signal_type': 'ENTRY',
        'data': data
    }
    write_stderr(f"ENTRY_SIGNAL: {json.dumps(entry_data)}\n")

def handle_exit_signal(strategy_id, data):
    """Handle exit signal from genetic algorithm"""
    exit_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'strategy_id': strategy_id,
        'signal_type': 'EXIT',
        'data': data
    }
    write_stderr(f"EXIT_SIGNAL: {json.dumps(exit_data)}\n")

def handle_risk_alert(strategy_id, data):
    """Handle risk alert from strategy"""
    risk_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'strategy_id': strategy_id,
        'alert_type': 'RISK',
        'data': data,
        'requires_action': True
    }
    write_stderr(f"RISK_ALERT: {json.dumps(risk_data)}\n")

def handle_performance_update(strategy_id, data):
    """Handle performance update from strategy"""
    perf_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'strategy_id': strategy_id,
        'update_type': 'PERFORMANCE',
        'data': data
    }
    write_stderr(f"PERFORMANCE_UPDATE: {json.dumps(perf_data)}\n")

def main():
    while True:
        write_stdout('READY\n')
        
        # Read event header
        line = sys.stdin.readline()
        headers = dict([x.split(':') for x in line.split()])
        
        # Read event payload
        payload = sys.stdin.read(int(headers['len']))
        
        try:
            if headers['eventname'].startswith('PROCESS_COMMUNICATION'):
                # Parse payload (first line is process info, rest is message)
                lines = payload.strip().split('\n', 1)
                if len(lines) == 2:
                    process_info = dict([x.split(':') for x in lines[0].split()])
                    message = lines[1]
                    
                    process_name = process_info['processname']
                    process_strategy_signal(process_name, message)
            
            write_stdout('RESULT 2\nOK')
            
        except Exception as e:
            write_stderr(f"Error in communication monitor: {e}\n")
            write_stdout('RESULT 4\nFAIL')

if __name__ == '__main__':
    main()
```

## Event System Integration with Trading Architecture

### Trading System Event Configuration

```ini
# Complete event monitoring for trading system
[eventlistener:critical_failure_monitor]
command = python /app/monitoring/failure_monitor.py
events = PROCESS_STATE_FATAL,PROCESS_STATE_EXITED
numprocs = 2
autostart = true
autorestart = unexpected
priority = 1

[eventlistener:performance_monitor]
command = python /app/monitoring/performance_monitor.py
events = TICK_60
numprocs = 1
autostart = true
autorestart = unexpected

[eventlistener:strategy_communication]
command = python /app/monitoring/comm_monitor.py
events = PROCESS_COMMUNICATION
numprocs = 1
autostart = true
autorestart = unexpected

[eventlistener:heartbeat_monitor]
command = python /app/monitoring/heartbeat_monitor.py
events = TICK_5
numprocs = 1
autostart = true
autorestart = unexpected

[eventlistener:startup_monitor]
command = python /app/monitoring/startup_monitor.py
events = PROCESS_STATE_RUNNING,PROCESS_STATE_STARTING
numprocs = 1
autostart = true
autorestart = unexpected
```

### Event-Driven Trading Actions

1. **Process Failure Response**
   - Stop trading on critical process failure
   - Activate backup systems
   - Send immediate alerts

2. **Performance Monitoring**
   - Track resource usage
   - Detect memory leaks
   - Monitor CPU utilization

3. **Strategy Communication**
   - Log all trading signals
   - Monitor risk alerts
   - Track performance updates

4. **System Health Checks**
   - Verify WebSocket connections
   - Check VPN status
   - Monitor data feed health

This event system provides comprehensive monitoring and automated response capabilities essential for a production quantitative trading system.