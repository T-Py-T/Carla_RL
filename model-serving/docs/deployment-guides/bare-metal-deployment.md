# Bare Metal Deployment Guide

This guide covers deploying the CarlaRL Policy-as-a-Service directly on bare metal servers without containerization.

## Table of Contents

- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Service Management](#service-management)
- [Monitoring Setup](#monitoring-setup)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Security](#security)

## Prerequisites

- Linux server (Ubuntu 20.04+, CentOS 8+, or RHEL 8+)
- Python 3.11+
- Root or sudo access
- Network connectivity
- At least 4GB RAM
- 10GB disk space

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 4GB
- **Storage**: 10GB SSD
- **Network**: 100 Mbps

### Recommended Requirements

- **CPU**: 8+ cores, 3.0+ GHz (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB+
- **Storage**: 50GB+ NVMe SSD
- **Network**: 1 Gbps
- **GPU**: NVIDIA RTX 3080+ (optional, for GPU acceleration)

### Hardware Optimization

- **CPU**: Intel with AVX2/AVX-512 support
- **Memory**: DDR4-3200 or faster
- **Storage**: NVMe SSD for model artifacts
- **Network**: Low-latency network interface

## Installation

### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    curl \
    wget \
    git \
    htop \
    iotop \
    nethogs \
    nginx \
    supervisor

# Create application user
sudo useradd -m -s /bin/bash carla-rl
sudo usermod -aG sudo carla-rl
```

### 2. Python Environment Setup

```bash
# Switch to application user
sudo su - carla-rl

# Create application directory
mkdir -p /home/carla-rl/carla-rl-serving
cd /home/carla-rl/carla-rl-serving

# Clone repository
git clone <repository-url> .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. GPU Support (Optional)

```bash
# Install NVIDIA drivers (Ubuntu)
sudo apt install -y nvidia-driver-525

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Create Model Artifacts

```bash
# Create artifacts directory
mkdir -p artifacts/v0.1.0

# Generate example artifacts
python scripts/create_example_artifacts.py

# Verify artifacts
ls -la artifacts/v0.1.0/
```

## Configuration

### 1. Application Configuration

Create `/home/carla-rl/carla-rl-serving/config/production.yaml`:

```yaml
# Production configuration
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  log_level: "info"
  access_log: true

model:
  artifact_dir: "/home/carla-rl/carla-rl-serving/artifacts"
  version: "v0.1.0"
  use_gpu: true
  gpu_memory_fraction: 0.8

monitoring:
  enable_metrics: true
  metrics_port: 8080
  enable_tracing: true
  log_format: "json"

optimization:
  enable_memory_pinning: true
  cache_size: 1000
  batch_size: 1
  enable_jit: true
  enable_avx: true
  enable_mkl: true

security:
  cors_origins: ["https://yourdomain.com"]
  allowed_hosts: ["yourdomain.com", "api.yourdomain.com"]
  max_request_size: "10MB"
  rate_limit: 1000  # requests per minute
```

### 2. Environment Variables

Create `/home/carla-rl/carla-rl-serving/.env`:

```bash
# Environment configuration
ARTIFACT_DIR=/home/carla-rl/carla-rl-serving/artifacts
MODEL_VERSION=v0.1.0
USE_GPU=1
LOG_LEVEL=info
WORKERS=4
CORS_ORIGINS=https://yourdomain.com
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
CONFIG_FILE=/home/carla-rl/carla-rl-serving/config/production.yaml
```

### 3. Nginx Configuration

Create `/etc/nginx/sites-available/carla-rl-serving`:

```nginx
upstream carla_rl_backend {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Request size limit
    client_max_body_size 10M;
    
    # Timeouts
    proxy_connect_timeout 5s;
    proxy_send_timeout 30s;
    proxy_read_timeout 30s;
    
    location / {
        proxy_pass http://carla_rl_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Connection pooling
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    # Health check endpoint
    location /healthz {
        proxy_pass http://carla_rl_backend/healthz;
        access_log off;
    }
    
    # Metrics endpoint
    location /metrics {
        proxy_pass http://carla_rl_backend/metrics;
        access_log off;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/carla-rl-serving /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. SSL Configuration (Optional)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Service Management

### 1. Systemd Service

Create `/etc/systemd/system/carla-rl-serving.service`:

```ini
[Unit]
Description=CarlaRL Policy-as-a-Service
After=network.target
Wants=network.target

[Service]
Type=exec
User=carla-rl
Group=carla-rl
WorkingDirectory=/home/carla-rl/carla-rl-serving
Environment=PATH=/home/carla-rl/carla-rl-serving/venv/bin
EnvironmentFile=/home/carla-rl/carla-rl-serving/.env
ExecStart=/home/carla-rl/carla-rl-serving/venv/bin/python -m uvicorn src.server:app --host 0.0.0.0 --port 8080 --workers 4
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=carla-rl-serving

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/carla-rl/carla-rl-serving
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable carla-rl-serving
sudo systemctl start carla-rl-serving
sudo systemctl status carla-rl-serving
```

### 2. Process Management

```bash
# Start service
sudo systemctl start carla-rl-serving

# Stop service
sudo systemctl stop carla-rl-serving

# Restart service
sudo systemctl restart carla-rl-serving

# Reload configuration
sudo systemctl reload carla-rl-serving

# Check status
sudo systemctl status carla-rl-serving

# View logs
sudo journalctl -u carla-rl-serving -f
```

### 3. Log Management

Create `/etc/logrotate.d/carla-rl-serving`:

```
/home/carla-rl/carla-rl-serving/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 carla-rl carla-rl
    postrotate
        systemctl reload carla-rl-serving
    endscript
}
```

## Monitoring Setup

### 1. Prometheus Installation

```bash
# Create Prometheus user
sudo useradd --no-create-home --shell /bin/false prometheus

# Create directories
sudo mkdir /etc/prometheus
sudo mkdir /var/lib/prometheus
sudo chown prometheus:prometheus /etc/prometheus
sudo chown prometheus:prometheus /var/lib/prometheus

# Download and install Prometheus
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvf prometheus-2.45.0.linux-amd64.tar.gz
sudo cp prometheus-2.45.0.linux-amd64/prometheus /usr/local/bin/
sudo cp prometheus-2.45.0.linux-amd64/promtool /usr/local/bin/
sudo chown prometheus:prometheus /usr/local/bin/prometheus
sudo chown prometheus:prometheus /usr/local/bin/promtool

# Create Prometheus configuration
sudo tee /etc/prometheus/prometheus.yml > /dev/null <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'carla-rl-serving'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF

sudo chown prometheus:prometheus /etc/prometheus/prometheus.yml

# Create systemd service
sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOF
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \\
    --config.file /etc/prometheus/prometheus.yml \\
    --storage.tsdb.path /var/lib/prometheus/ \\
    --web.console.templates=/etc/prometheus/consoles \\
    --web.console.libraries=/etc/prometheus/console_libraries \\
    --web.listen-address=0.0.0.0:9090 \\
    --web.enable-lifecycle

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
```

### 2. Grafana Installation

```bash
# Install Grafana
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
sudo apt update
sudo apt install -y grafana

# Start Grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

# Access Grafana at http://your-server:3000 (admin/admin)
```

### 3. Node Exporter

```bash
# Install Node Exporter
wget https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz
tar xvf node_exporter-1.6.1.linux-amd64.tar.gz
sudo cp node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/
sudo chown prometheus:prometheus /usr/local/bin/node_exporter

# Create systemd service
sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<EOF
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter
```

## Performance Tuning

### 1. System Optimization

```bash
# CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Network optimization
sudo tee -a /etc/sysctl.conf > /dev/null <<EOF
# Network optimizations
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr
EOF

sudo sysctl -p

# File descriptor limits
sudo tee -a /etc/security/limits.conf > /dev/null <<EOF
carla-rl soft nofile 65536
carla-rl hard nofile 65536
carla-rl soft nproc 4096
carla-rl hard nproc 4096
EOF
```

### 2. Python Optimization

```bash
# Set Python optimizations
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Add to .env file
echo "PYTHONOPTIMIZE=1" >> /home/carla-rl/carla-rl-serving/.env
echo "PYTHONUNBUFFERED=1" >> /home/carla-rl/carla-rl-serving/.env
echo "PYTHONDONTWRITEBYTECODE=1" >> /home/carla-rl/carla-rl-serving/.env
```

### 3. GPU Optimization

```bash
# Set GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

# Add to .env file
echo "TF_FORCE_GPU_ALLOW_GROWTH=true" >> /home/carla-rl/carla-rl-serving/.env
echo "CUDA_VISIBLE_DEVICES=0" >> /home/carla-rl/carla-rl-serving/.env
```

### 4. Memory Optimization

```bash
# Enable huge pages
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages

# Set swappiness
echo 10 | sudo tee /proc/sys/vm/swappiness

# Add to /etc/sysctl.conf
sudo tee -a /etc/sysctl.conf > /dev/null <<EOF
vm.nr_hugepages = 1024
vm.swappiness = 10
EOF
```

## Troubleshooting

### 1. Service Issues

```bash
# Check service status
sudo systemctl status carla-rl-serving

# View logs
sudo journalctl -u carla-rl-serving -f

# Check configuration
sudo systemctl cat carla-rl-serving

# Test configuration
sudo systemd-analyze verify /etc/systemd/system/carla-rl-serving.service
```

### 2. Performance Issues

```bash
# Check resource usage
htop
iotop
nethogs

# Check Python process
ps aux | grep python
top -p $(pgrep -f "uvicorn")

# Check memory usage
free -h
cat /proc/meminfo

# Check disk I/O
iostat -x 1
```

### 3. Network Issues

```bash
# Check port binding
sudo netstat -tulpn | grep 8080
sudo ss -tulpn | grep 8080

# Test connectivity
curl -v http://localhost:8080/healthz
curl -v http://api.yourdomain.com/healthz

# Check firewall
sudo ufw status
sudo iptables -L
```

### 4. GPU Issues

```bash
# Check GPU status
nvidia-smi
nvidia-smi -q

# Check CUDA installation
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Security

### 1. Firewall Configuration

```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow internal monitoring
sudo ufw allow from 127.0.0.1 to any port 9090
sudo ufw allow from 127.0.0.1 to any port 3000
```

### 2. User Security

```bash
# Disable root login
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# Set up SSH keys
sudo su - carla-rl
mkdir -p ~/.ssh
chmod 700 ~/.ssh
# Add your public key to ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### 3. Application Security

```bash
# Set proper permissions
sudo chown -R carla-rl:carla-rl /home/carla-rl/carla-rl-serving
sudo chmod -R 755 /home/carla-rl/carla-rl-serving
sudo chmod 600 /home/carla-rl/carla-rl-serving/.env

# Disable unnecessary services
sudo systemctl disable snapd
sudo systemctl disable bluetooth
sudo systemctl disable cups
```

### 4. Monitoring Security

```bash
# Set up fail2ban
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Configure fail2ban for nginx
sudo tee /etc/fail2ban/jail.local > /dev/null <<EOF
[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
EOF

sudo systemctl restart fail2ban
```

## Backup and Recovery

### 1. Backup Script

Create `/home/carla-rl/backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR="/home/carla-rl/backups"
APP_DIR="/home/carla-rl/carla-rl-serving"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup application
tar -czf $BACKUP_DIR/carla-rl-serving_$DATE.tar.gz -C $APP_DIR .

# Backup configuration
sudo tar -czf $BACKUP_DIR/system-config_$DATE.tar.gz /etc/nginx/sites-available/carla-rl-serving /etc/systemd/system/carla-rl-serving.service

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/carla-rl-serving_$DATE.tar.gz"
```

Make it executable:

```bash
chmod +x /home/carla-rl/backup.sh
```

### 2. Automated Backup

```bash
# Add to crontab
crontab -e
# Add: 0 2 * * * /home/carla-rl/backup.sh
```

### 3. Recovery

```bash
# Stop service
sudo systemctl stop carla-rl-serving

# Restore application
tar -xzf /home/carla-rl/backups/carla-rl-serving_YYYYMMDD_HHMMSS.tar.gz -C /home/carla-rl/carla-rl-serving/

# Restore configuration
sudo tar -xzf /home/carla-rl/backups/system-config_YYYYMMDD_HHMMSS.tar.gz -C /

# Restart service
sudo systemctl start carla-rl-serving
```

## Next Steps

- [Configuration Reference](../configuration-reference.md)
- [Performance Tuning Guide](../performance-tuning/performance-tuning.md)
- [Monitoring Setup Guide](../monitoring/monitoring-setup.md)
- [Troubleshooting Guide](../troubleshooting/troubleshooting.md)
