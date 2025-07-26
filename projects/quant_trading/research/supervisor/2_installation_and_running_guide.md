# Supervisor - Installation and Running Guide

**Source**: http://supervisord.org/installing.html, http://supervisord.org/running.html  
**Extracted**: 2025-07-25  
**Purpose**: Complete installation and operational guide for trading system process management

## Installation Methods

### Option 1: Pip Installation (Recommended)

```bash
# System-wide installation (requires root)
pip install supervisor

# Virtual environment installation (recommended for trading systems)
python -m venv venv
source venv/bin/activate
pip install supervisor
```

### Option 2: Manual Installation (Offline Systems)

For systems without internet access (common in production trading environments):

1. Download dependencies on internet-connected machine:
   - `setuptools` from https://pypi.org/pypi/setuptools/
   - `supervisor` from https://pypi.org/pypi/supervisor/

2. Transfer files to target system and install:
```bash
python setup.py install  # For each dependency
```

### Option 3: Distribution Package

```bash
# Ubuntu/Debian
apt-cache show supervisor
sudo apt install supervisor

# CentOS/RHEL
yum info supervisor
sudo yum install supervisor
```

**Note**: Distribution packages may lag behind official releases and include system-specific modifications.

## Initial Configuration

### Generate Sample Configuration

```bash
# Generate sample config to terminal
echo_supervisord_conf

# Generate config file (requires root for /etc/)
echo_supervisord_conf > /etc/supervisord.conf

# Generate config in current directory (recommended for trading systems)
echo_supervisord_conf > supervisord.conf
```

### Trading System Directory Structure

```bash
# Recommended structure for trading systems
/app/
├── supervisord.conf          # Main config file
├── logs/                     # All process logs
├── supervisor.sock           # Unix socket for control
├── supervisord.pid           # Daemon PID file
├── src/                      # Application source code
└── config/                   # Additional config files
```

## Running Supervisor

### Starting supervisord

```bash
# Start daemon (background)
supervisord

# Start with specific config
supervisord -c /app/supervisord.conf

# Start in foreground (debugging)
supervisord -n

# Start with specific user (when started as root)
supervisord -u trader

# Start with custom log level
supervisord -e debug
```

### Key Command-Line Options

| Option | Description | Trading System Usage |
|--------|-------------|---------------------|
| `-c FILE` | Configuration file path | Always specify for security |
| `-n` | Run in foreground | Development and debugging |
| `-u USER` | Switch to user | Run as 'trader' not root |
| `-d PATH` | Change directory before daemonize | Set to /app |
| `-l FILE` | Log file location | /app/logs/supervisord.log |
| `-e LEVEL` | Log level | 'info' for production, 'debug' for dev |
| `-j FILE` | PID file location | /app/supervisord.pid |

### Security Considerations

⚠️ **WARNING**: When started as root, supervisord searches current directory for config file. Always use `-c` flag in production:

```bash
# INSECURE (searches current directory)
sudo supervisord

# SECURE (explicit config path)
sudo supervisord -c /app/supervisord.conf
```

## Process Control with supervisorctl

### Interactive Mode

```bash
# Start interactive shell
supervisorctl

# With specific config
supervisorctl -c /app/supervisord.conf

# Interactive prompt example
trading_supervisor> help
trading_supervisor> status all
trading_supervisor> start genetic_engine
trading_supervisor> tail -f hyperliquid_feed
```

### Command-Line Mode (One-shot Commands)

```bash
# Check status of all processes
supervisorctl status all

# Start specific process
supervisorctl start genetic_engine

# Restart process group
supervisorctl restart trading_system:*

# Stop all processes
supervisorctl stop all

# View process logs
supervisorctl tail genetic_engine
supervisorctl tail -f hyperliquid_feed stdout

# Configuration reload
supervisorctl reread
supervisorctl update
```

### Key supervisorctl Actions

| Action | Description | Example |
|--------|-------------|---------|
| `status` | Show process status | `status all` |
| `start` | Start process/group | `start genetic_engine` |
| `stop` | Stop process/group | `stop trading_system:*` |
| `restart` | Restart process/group | `restart all` |
| `reload` | Restart supervisord | `reload` |
| `reread` | Reload config (no restart) | `reread` |
| `update` | Apply config changes | `update` |
| `tail` | View process logs | `tail -f risk_manager` |
| `clear` | Clear process logs | `clear all` |

## Signal Handling

### Daemon Control Signals

Send signals to supervisord process (PID found in pidfile):

```bash
# Graceful shutdown (wait for processes to stop)
kill -TERM $(cat /app/supervisord.pid)
kill -INT $(cat /app/supervisord.pid)
kill -QUIT $(cat /app/supervisord.pid)

# Reload configuration (restart all processes)
kill -HUP $(cat /app/supervisord.pid)

# Reopen log files (log rotation)
kill -USR2 $(cat /app/supervisord.pid)
```

### Integration with Log Rotation

```bash
# logrotate.d/supervisor
/app/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 trader trader
    postrotate
        kill -USR2 $(cat /app/supervisord.pid) 2>/dev/null || true
    endscript
}
```

## Startup Configuration

### systemd Integration (Modern Linux)

Create `/etc/systemd/system/supervisord.service`:

```ini
[Unit]
Description=Supervisor daemon
Documentation=http://supervisord.org
After=network.target

[Service]
ExecStart=/usr/local/bin/supervisord -n -c /app/supervisord.conf
ExecStop=/usr/local/bin/supervisorctl -c /app/supervisord.conf shutdown
ExecReload=/usr/local/bin/supervisorctl -c /app/supervisord.conf reread
ExecReload=/usr/local/bin/supervisorctl -c /app/supervisord.conf update
KillMode=mixed
Restart=on-failure
RestartSec=5s
Type=exec
User=trader
Group=trader

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable supervisord
sudo systemctl start supervisord
sudo systemctl status supervisord
```

### Trading System Boot Sequence

```bash
# 1. System boot
# 2. VPN connection (if required)
# 3. Database services
# 4. supervisord starts
# 5. Trading processes start in priority order
```

## Process Management Patterns

### Dependency Management

Use `priority` in config to control startup order:

```ini
# Infrastructure (start first)
[program:database]
priority = 10

# Core services  
[program:risk_manager]
priority = 50

# Data feeds
[program:hyperliquid_feed]
priority = 100

# Trading engines (start last)
[program:genetic_engine]
priority = 200
```

### Health Monitoring

```bash
# Check if all critical processes are running
supervisorctl status | grep -E "(RUNNING|STARTING)" | wc -l

# Get PIDs for monitoring
supervisorctl pid all

# Check specific process health
supervisorctl status genetic_engine | grep RUNNING
```

### Development vs Production

**Development Mode:**
```bash
# Run in foreground for debugging
supervisord -n -c dev_supervisord.conf

# Single process testing
supervisorctl start genetic_engine
supervisorctl tail -f genetic_engine
```

**Production Mode:**
```bash
# Daemon mode with full logging
supervisord -c /app/supervisord.conf

# Automated health checks
supervisorctl status | grep -v RUNNING && echo "ALERT: Process down!"
```

## Integration with Trading System Architecture

### VPN Zone Separation

```ini
# Non-VPN processes (can run anywhere)
[group:analysis]
programs = genetic_engine,backtesting_engine,performance_analyzer

# VPN-required processes (Hyperliquid access)
[group:execution]  
programs = hyperliquid_feed,order_executor,position_monitor
```

### Resource Management

```ini
[program:backtest_workers]
numprocs = 4                    # Parallel workers
process_name = worker_%(process_num)02d
command = python worker.py --id=%(process_num)s
```

### Error Recovery

```ini
[program:critical_service]
autorestart = unexpected        # Restart on unexpected exits
startretries = 3               # Maximum restart attempts
startsecs = 10                 # Must run 10s to be "successful"
stopwaitsecs = 30              # Grace period before SIGKILL
```

This comprehensive guide provides everything needed to deploy and manage Supervisor for a production quantitative trading system.