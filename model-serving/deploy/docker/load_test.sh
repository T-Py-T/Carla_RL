#!/bin/sh
# Simple load testing script for Docker Compose testing

SERVICE_URL=${1:-"http://carla-rl-serving:8080"}
REQUESTS=${2:-50}
CONCURRENCY=${3:-5}

echo "Load Testing CarlaRL Policy Service"
echo "URL: $SERVICE_URL"
echo "Requests: $REQUESTS"
echo "Concurrency: $CONCURRENCY"
echo "================================"

# Wait for service to be ready
echo "Waiting for service..."
until curl -s "$SERVICE_URL/healthz" > /dev/null; do
  echo "Waiting for service to be ready..."
  sleep 2
done

echo "Service is ready!"

# Test payload
PAYLOAD='{
  "observations": [{
    "speed": 25.0,
    "steering": 0.0,
    "sensors": [0.5, 0.5, 0.5, 0.5, 0.5]
  }],
  "deterministic": true
}'

echo "Starting load test..."

# Simple concurrent load test
for i in $(seq 1 $CONCURRENCY); do
  (
    echo "Worker $i starting..."
    requests_per_worker=$((REQUESTS / CONCURRENCY))
    for j in $(seq 1 $requests_per_worker); do
      start_time=$(date +%s%3N)
      
      response=$(curl -s -w "%{http_code}" -X POST "$SERVICE_URL/predict" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD")
      
      end_time=$(date +%s%3N)
      latency=$((end_time - start_time))
      
      http_code=${response: -3}
      if [ "$http_code" = "200" ]; then
        echo "Worker $i Request $j: ${latency}ms [SUCCESS]"
      else
        echo "Worker $i Request $j: HTTP $http_code [FAILED]"
      fi
      
      # Small delay between requests
      sleep 0.1
    done
    echo "Worker $i completed"
  ) &
done

# Wait for all workers to complete
wait

echo "Load test completed!"

# Get final metrics
echo "Final service metrics:"
curl -s "$SERVICE_URL/metrics" | grep carla_rl || echo "No metrics available"
