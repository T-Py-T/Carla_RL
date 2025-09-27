#!/bin/bash
set -e

# Docker Compose Testing Script for OrbStack
# Simplified testing approach using Docker Compose

echo "CarlaRL Policy-as-a-Service Docker Compose Testing"
echo "===================================================="

# Configuration
COMPOSE_FILE="deploy/docker/docker-compose.test.yml"
PRODUCTION_COMPOSE_FILE="deploy/docker/docker-compose.yml"
TEST_TIMEOUT=300  # 5 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to Docker
    if ! docker info &> /dev/null; then
        log_error "Cannot connect to Docker daemon"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        log_info "Make sure you're running this from the carla-rl-serving directory"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create example artifacts if they don't exist
ensure_artifacts() {
    log_info "Ensuring model artifacts exist..."
    
    if [[ ! -f "artifacts/v0.1.0/model_card.yaml" ]]; then
        log_info "Creating example artifacts..."
        if command -v python3 &> /dev/null; then
            python3 scripts/create_example_artifacts.py --output artifacts --version v0.1.0
        else
            log_warning "Python3 not available, using existing artifacts"
        fi
    else
        log_info "Artifacts already exist"
    fi
}

# Build and start services
start_services() {
    log_info "Building and starting services..."
    
    # Use docker compose or docker-compose based on availability
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    # Build and start the main service
    $COMPOSE_CMD -f $COMPOSE_FILE build carla-rl-serving
    $COMPOSE_CMD -f $COMPOSE_FILE up -d carla-rl-serving
    
    # Wait for service to be healthy
    log_info "Waiting for service to be healthy..."
    timeout $TEST_TIMEOUT bash -c '
        while ! docker inspect carla-rl-serving-test --format="{{.State.Health.Status}}" | grep -q "healthy"; do
            echo "Waiting for service health check..."
            sleep 5
        done
    '
    
    if docker inspect carla-rl-serving-test --format="{{.State.Health.Status}}" | grep -q "healthy"; then
        log_success "Service is healthy and ready"
    else
        log_error "Service failed to become healthy"
        $COMPOSE_CMD -f $COMPOSE_FILE logs carla-rl-serving
        exit 1
    fi
}

# Run validation tests
run_validation_tests() {
    log_info "Running validation tests..."
    
    # Method 1: Run tests from host machine
    if command -v python3 &> /dev/null && [[ -f "scripts/cluster_validation.py" ]]; then
        log_info "Running validation tests from host..."
        python3 scripts/cluster_validation.py --url http://localhost:8080 --output docker_test_results.json
        local test_exit_code=$?
        
        if [[ $test_exit_code -eq 0 ]]; then
            log_success "Host-based validation tests passed!"
        else
            log_error "Host-based validation tests failed"
            return $test_exit_code
        fi
    else
        log_warning "Cannot run Python tests from host"
    fi
    
    # Method 2: Run tests using test-runner container
    log_info "Running containerized validation tests..."
    
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    $COMPOSE_CMD -f $COMPOSE_FILE --profile testing run --rm test-runner
    local container_test_exit_code=$?
    
    if [[ $container_test_exit_code -eq 0 ]]; then
        log_success "Containerized validation tests passed!"
    else
        log_error "Containerized validation tests failed"
        return $container_test_exit_code
    fi
}

# Run load tests
run_load_tests() {
    log_info "Running load tests..."
    
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    # Make load test script executable
    chmod +x deploy/docker/load_test.sh
    
    # Run load tests
    $COMPOSE_CMD -f $COMPOSE_FILE --profile load-testing run --rm load-tester
    
    log_success "Load tests completed"
}

# Check service metrics and logs
check_service_status() {
    log_info "Checking service status and metrics..."
    
    # Show container status
    echo "Container Status:"
    docker ps --filter "name=carla-rl-serving-test"
    
    # Show resource usage
    echo -e "\nResource Usage:"
    docker stats carla-rl-serving-test --no-stream
    
    # Test basic endpoints
    echo -e "\nTesting endpoints:"
    
    # Health check
    if curl -s http://localhost:8080/healthz | jq . 2>/dev/null; then
        echo "[SUCCESS] Health endpoint working"
    else
        echo "[FAILED] Health endpoint failed"
    fi
    
    # Metadata
    if curl -s http://localhost:8080/metadata | jq . 2>/dev/null; then
        echo "[SUCCESS] Metadata endpoint working"
    else
        echo "[FAILED] Metadata endpoint failed"
    fi
    
    # Metrics
    if curl -s http://localhost:8080/metrics | head -5; then
        echo "[SUCCESS] Metrics endpoint working"
    else
        echo "[FAILED] Metrics endpoint failed"
    fi
    
    # Show recent logs
    echo -e "\nRecent logs:"
    docker logs carla-rl-serving-test --tail=20
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    # Stop and remove containers
    $COMPOSE_CMD -f $COMPOSE_FILE down --volumes --remove-orphans
    
    # Optionally remove images
    if [[ "$1" == "--remove-images" ]]; then
        docker image rm carla-rl-serving:latest 2>/dev/null || true
        log_info "Images removed"
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    local run_load_tests=false
    local cleanup_after_test=true
    local remove_images=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --load-test)
                run_load_tests=true
                shift
                ;;
            --no-cleanup)
                cleanup_after_test=false
                shift
                ;;
            --remove-images)
                remove_images=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --load-test       Run load tests in addition to validation tests"
                echo "  --no-cleanup      Don't cleanup containers after testing"
                echo "  --remove-images   Remove Docker images during cleanup"
                echo "  --help            Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Set trap for cleanup
    if [[ "$cleanup_after_test" == true ]]; then
        if [[ "$remove_images" == true ]]; then
            trap 'cleanup --remove-images' EXIT
        else
            trap 'cleanup' EXIT
        fi
    fi
    
    # Execute test pipeline
    check_prerequisites
    ensure_artifacts
    start_services
    
    # Run tests
    run_validation_tests
    local validation_exit_code=$?
    
    if [[ "$run_load_tests" == true ]]; then
        run_load_tests
    fi
    
    # Show service status
    check_service_status
    
    # Final summary
    if [[ $validation_exit_code -eq 0 ]]; then
        log_success "All Docker Compose tests passed!"
        echo ""
        echo "Service is running at: http://localhost:8080"
        echo "Test results saved to: docker_test_results.json"
        echo ""
        if [[ "$cleanup_after_test" == false ]]; then
            echo "Service will continue running (use 'docker-compose -f $COMPOSE_FILE down' to stop)"
        fi
    else
        log_error "Some tests failed"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
