# Supervisor - Configuration Documentation

**Source**: http://supervisord.org/configuration.html  
**Extracted**: 2025-07-25  
**Purpose**: Process management and auto-restart configuration for trading system components

## Overview

Supervisor is a client/server system that allows users to monitor and control processes on UNIX-like operating systems. It's designed for managing processes related to a project rather than being an init replacement.

## Configuration File Structure

The configuration file `supervisord.conf` uses Windows-INI-style format with sections and key/value pairs. Supervisor searches for the config file in this order:

1. `../etc/supervisord.conf` (Relative to executable)
2. `../supervisord.conf` (Relative to executable)  
3. `$CWD/supervisord.conf`
4. `$CWD/etc/supervisord.conf`
5. `/etc/supervisord.conf`
6. `/etc/supervisor/supervisord.conf`

## Core Configuration Sections

### 1. `[supervisord]` Section - Main Daemon Settings

```ini
[supervisord]
logfile = /tmp/supervisord.log          # Activity log path
logfile_maxbytes = 50MB                 # Max log size before rotation
logfile_backups = 10                    # Number of backup files
loglevel = info                         # critical, error, warn, info, debug, trace, blather
pidfile = /tmp/supervisord.pid          # PID file location
nodaemon = false                        # Run in foreground (true) or daemon (false)
minfds = 1024                           # Minimum file descriptors
minprocs = 200                          # Minimum process descriptors
umask = 022                             # Process umask
user = chrism                           # Switch to this user (if started as root)
directory = /tmp                        # Change to this directory when daemonizing
childlogdir = /tmp                      # Directory for AUTO child logs
environment = KEY1="value1",KEY2="value2"  # Environment variables
```

### 2. `[program:x]` Section - Process Definitions

This is the core section for defining processes to manage:

```ini
[program:trading_engine]
command = /usr/bin/python /app/trading_engine.py    # Command to run
directory = /app                                     # Working directory
user = trader                                        # Run as this user
autostart = true                                     # Start automatically
autorestart = unexpected                             # Restart policy: true, false, unexpected
startsecs = 10                                       # Seconds to run before considering started
startretries = 3                                     # Number of restart attempts
exitcodes = 0                                        # Expected exit codes
stopsignal = TERM                                    # Signal to send for graceful stop
stopwaitsecs = 10                                    # Seconds to wait before SIGKILL
priority = 999                                       # Start/stop order priority
numprocs = 1                                         # Number of process instances
process_name = %(program_name)s                      # Process naming pattern

# Logging configuration
stdout_logfile = /var/log/trading_engine.log         # Stdout log file
stdout_logfile_maxbytes = 50MB                       # Max log size
stdout_logfile_backups = 10                          # Backup files
stderr_logfile = /var/log/trading_engine_error.log   # Stderr log file
redirect_stderr = false                              # Redirect stderr to stdout

# Environment and security
environment = API_KEY="%(ENV_API_KEY)s"              # Environment variables
umask = 022                                          # File creation mask
```

### 3. `[group:x]` Section - Process Grouping

Group related processes together for collective management:

```ini
[group:trading_system]
programs = data_feed,strategy_engine,risk_manager    # Comma-separated program names
priority = 999                                       # Group priority
```

### 4. `[unix_http_server]` Section - Control Interface

```ini
[unix_http_server]
file = /tmp/supervisor.sock                          # Socket file path
chmod = 0700                                         # Socket permissions
chown = trader:trader                                # Socket ownership
username = admin                                     # Authentication username
password = secure_password                           # Authentication password
```

### 5. `[supervisorctl]` Section - Client Configuration

```ini
[supervisorctl]
serverurl = unix:///tmp/supervisor.sock              # Connection URL
username = admin                                     # Username for authentication
password = secure_password                           # Password for authentication
prompt = trading_supervisor                          # Command prompt
```

## Key Features for Trading System

### Auto-Restart Capabilities

- **`autorestart = unexpected`**: Restart only on unexpected exits
- **`autorestart = true`**: Always restart when process exits
- **`autorestart = false`**: Never restart automatically

### Process States and Lifecycle

1. **STOPPED** → **STARTING** → **RUNNING** → **STOPPING** → **STOPPED**
2. **FATAL**: Process failed to start after maximum retries
3. **BACKOFF**: Waiting between restart attempts

### Startup and Shutdown Control

- **`startsecs`**: Time process must run to be considered successfully started
- **`startretries`**: Maximum restart attempts before marking FATAL
- **`priority`**: Controls start/stop order (lower numbers start first)
- **`stopwaitsecs`**: Grace period before SIGKILL

### Multi-Process Management

```ini
[program:strategy_workers]
command = /usr/bin/python worker.py --id=%(process_num)s
process_name = strategy_worker_%(process_num)02d
numprocs = 4                                         # Run 4 instances
numprocs_start = 0                                   # Start numbering from 0
directory = /app
autostart = true
autorestart = unexpected
```

Creates processes: `strategy_worker_00`, `strategy_worker_01`, `strategy_worker_02`, `strategy_worker_03`

## Environment Variables

Support for environment variable expansion:

```ini
[program:api_service]
command = /usr/bin/python api.py --port=%(ENV_API_PORT)s
environment = 
    LOG_LEVEL="%(ENV_LOG_LEVEL)s",
    DATABASE_URL="%(ENV_DATABASE_URL)s",
    API_KEY="%(ENV_HYPERLIQUID_API_KEY)s"
```

## Logging Configuration

### Automatic Log Management

- **AUTO**: Supervisor manages log files automatically
- **NONE**: No logging
- **Custom Path**: Specify exact log file location

### Log Rotation

```ini
stdout_logfile_maxbytes = 50MB                       # Rotate when file reaches 50MB
stdout_logfile_backups = 10                          # Keep 10 backup files
```

### Centralized Logging

```ini
redirect_stderr = true                               # Combine stdout and stderr
stdout_logfile = /var/log/%(program_name)s.log       # Use program name in log path
```

## Implementation for Quant Trading System

### Example Configuration Structure

```ini
[supervisord]
logfile = /app/logs/supervisord.log
pidfile = /app/supervisord.pid
user = trader
directory = /app
loglevel = info

[unix_http_server]
file = /app/supervisor.sock
chmod = 0700

[supervisorctl]
serverurl = unix:///app/supervisor.sock

# Data ingestion service
[program:hyperliquid_feed]
command = /usr/bin/python hyperliquid_websocket.py
directory = /app/src
user = trader
autostart = true
autorestart = unexpected
startsecs = 5
startretries = 3
stdout_logfile = /app/logs/hyperliquid_feed.log
environment = VPN_REQUIRED="true"

# Genetic algorithm engine
[program:genetic_engine]
command = /usr/bin/python genetic_evolution.py
directory = /app/src
user = trader
autostart = true
autorestart = unexpected
startsecs = 10
priority = 100
stdout_logfile = /app/logs/genetic_engine.log

# Strategy backtesting workers
[program:backtest_workers]
command = /usr/bin/python backtest_worker.py --worker-id=%(process_num)s
process_name = backtest_worker_%(process_num)02d
numprocs = 4
directory = /app/src
autostart = false
autorestart = unexpected
stdout_logfile = /app/logs/backtest_worker_%(process_num)02d.log

# Risk management service
[program:risk_manager]
command = /usr/bin/python risk_manager.py
directory = /app/src
user = trader
autostart = true
autorestart = true
priority = 50
stdout_logfile = /app/logs/risk_manager.log

# Group all trading components
[group:trading_system]
programs = hyperliquid_feed,genetic_engine,risk_manager
priority = 100
```

### Command Line Operations

```bash
# Start all services
supervisorctl start all

# Start specific group
supervisorctl start trading_system:*

# Restart a service
supervisorctl restart genetic_engine

# Check status
supervisorctl status

# View logs
supervisorctl tail genetic_engine

# Stop all services gracefully
supervisorctl stop all
```

## Security Considerations

1. **File Permissions**: Use `chmod = 0700` for socket files
2. **User Isolation**: Run processes as non-root users
3. **Authentication**: Set username/password for HTTP interface
4. **Network Access**: Use UNIX sockets instead of network sockets when possible

## Integration with Trading System Architecture

### VPN Zone Separation

```ini
# Non-VPN zone processes
[program:genetic_engine]
# ... configuration

[program:backtesting_engine]  
# ... configuration

# VPN zone processes (requiring Hyperliquid access)
[program:order_execution]
command = /usr/bin/python order_executor.py
environment = VPN_REQUIRED="true"
# ... other configuration
```

### Process Dependencies

Use `priority` to control startup order:

1. **Priority 1-50**: Infrastructure (databases, logging)
2. **Priority 51-100**: Core services (data feeds, risk management)
3. **Priority 101-200**: Trading engines
4. **Priority 201+**: Non-critical services

This ensures proper system initialization and graceful shutdown sequences.

## Error Recovery Patterns

### Exponential Backoff

Supervisor automatically implements backoff between restart attempts, increasing delay with each failure.

### Circuit Breaker Pattern

```ini
startretries = 5                                     # Maximum attempts
startsecs = 30                                       # Must run 30 seconds to be "successful"
```

### Health Check Integration

Use external health check scripts with event listeners to monitor process health beyond simple process existence.