#!/bin/bash

# CarlaRL Monitoring Setup Script
# This script sets up the complete monitoring stack for CarlaRL serving

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up CarlaRL Monitoring Stack${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed. Please install docker-compose and try again.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating monitoring directories...${NC}"
mkdir -p prometheus_data grafana_data alertmanager_data

# Set proper permissions
echo -e "${YELLOW}Setting up permissions...${NC}"
chmod 755 prometheus_data grafana_data alertmanager_data

# Start monitoring stack
echo -e "${YELLOW}Starting monitoring stack...${NC}"
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check if services are running
echo -e "${YELLOW}Checking service status...${NC}"
if docker-compose -f docker-compose.monitoring.yml ps | grep -q "Up"; then
    echo -e "${GREEN}Monitoring stack started successfully!${NC}"
    echo ""
    echo -e "${GREEN}Services available at:${NC}"
    echo -e "  Prometheus: http://localhost:9090"
    echo -e "  Grafana: http://localhost:3000 (admin/admin)"
    echo -e "  Alertmanager: http://localhost:9093"
    echo -e "  CarlaRL Service: http://localhost:8080"
    echo ""
    echo -e "${YELLOW}To stop the monitoring stack, run:${NC}"
    echo -e "  docker-compose -f docker-compose.monitoring.yml down"
    echo ""
    echo -e "${YELLOW}To view logs, run:${NC}"
    echo -e "  docker-compose -f docker-compose.monitoring.yml logs -f"
else
    echo -e "${RED}Error: Failed to start monitoring stack${NC}"
    echo -e "${YELLOW}Check logs with: docker-compose -f docker-compose.monitoring.yml logs${NC}"
    exit 1
fi

# Import Grafana dashboard
echo -e "${YELLOW}Importing Grafana dashboard...${NC}"
sleep 5

# Check if Grafana is ready
for i in {1..30}; do
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}Grafana is ready!${NC}"
        break
    fi
    echo -e "${YELLOW}Waiting for Grafana... (${i}/30)${NC}"
    sleep 2
done

# Create dashboard via API
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo -e "${YELLOW}Creating Grafana dashboard...${NC}"
    
    # Get admin API key
    ADMIN_KEY=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"name":"apikey", "role":"Admin"}' \
        http://admin:admin@localhost:3000/api/auth/keys | \
        jq -r '.key' 2>/dev/null || echo "")
    
    if [ -n "$ADMIN_KEY" ]; then
        # Import dashboard
        curl -s -X POST \
            -H "Authorization: Bearer $ADMIN_KEY" \
            -H "Content-Type: application/json" \
            -d @grafana/carla-rl-dashboard.json \
            http://localhost:3000/api/dashboards/db > /dev/null
        
        echo -e "${GREEN}Dashboard imported successfully!${NC}"
    else
        echo -e "${YELLOW}Could not create API key. Please import dashboard manually.${NC}"
        echo -e "${YELLOW}Dashboard file: grafana/carla-rl-dashboard.json${NC}"
    fi
else
    echo -e "${YELLOW}Grafana is not ready. Please import dashboard manually later.${NC}"
    echo -e "${YELLOW}Dashboard file: grafana/carla-rl-dashboard.json${NC}"
fi

echo -e "${GREEN}Monitoring setup complete!${NC}"
