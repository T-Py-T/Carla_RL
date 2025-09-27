#!/bin/bash
set -e

# CarlaRL Policy-as-a-Service Cluster Testing Script
# This script performs comprehensive testing on OrbStack or any Kubernetes cluster

echo "🚀 CarlaRL Policy-as-a-Service Cluster Testing"
echo "=============================================="

# Configuration
NAMESPACE=${NAMESPACE:-default}
SERVICE_NAME=${SERVICE_NAME:-carla-rl-serving-service}
IMAGE_NAME=${IMAGE_NAME:-carla-rl-serving:latest}
TEST_URL=${TEST_URL:-""}

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
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if Docker is available for building
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available - assuming image is already built"
    fi
    
    log_success "Prerequisites check passed"
}

# Build and load image
build_and_load_image() {
    log_info "Building and loading Docker image..."
    
    if command -v docker &> /dev/null; then
        # Build the image
        cd "$(dirname "$0")/.."
        docker build -t ${IMAGE_NAME} .
        
        # For OrbStack, images are automatically available to k8s
        # For other systems, you might need to load or push to registry
        log_success "Image built: ${IMAGE_NAME}"
    else
        log_warning "Skipping image build (Docker not available)"
    fi
}

# Deploy to cluster
deploy_to_cluster() {
    log_info "Deploying to Kubernetes cluster..."
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/ -n ${NAMESPACE}
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/carla-rl-serving -n ${NAMESPACE}
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready --timeout=300s pod -l app=carla-rl-serving -n ${NAMESPACE}
    
    log_success "Deployment completed"
}

# Get service URL
get_service_url() {
    if [[ -n "$TEST_URL" ]]; then
        echo "$TEST_URL"
        return
    fi
    
    # Try to get LoadBalancer external IP
    EXTERNAL_IP=$(kubectl get svc ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [[ -n "$EXTERNAL_IP" ]]; then
        echo "http://${EXTERNAL_IP}"
        return
    fi
    
    # Try NodePort
    NODE_PORT=$(kubectl get svc ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")
    if [[ -n "$NODE_PORT" ]]; then
        NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
        echo "http://${NODE_IP}:${NODE_PORT}"
        return
    fi
    
    # Fallback to port-forward
    log_info "Using port-forward for testing..."
    kubectl port-forward svc/${SERVICE_NAME} 8080:80 -n ${NAMESPACE} &
    PORT_FORWARD_PID=$!
    sleep 5
    echo "http://localhost:8080"
}

# Run validation tests
run_validation_tests() {
    local service_url=$1
    log_info "Running validation tests against: ${service_url}"
    
    # Run the Python validation script
    python3 scripts/cluster_validation.py --url "${service_url}" --output cluster_test_results.json
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "All validation tests passed!"
    else
        log_error "Some validation tests failed"
    fi
    
    return $exit_code
}

# Performance testing
run_performance_tests() {
    local service_url=$1
    log_info "Running performance tests..."
    
    # Basic load test using curl
    log_info "Testing single request latency..."
    
    for i in {1..10}; do
        start_time=$(date +%s%3N)
        
        response=$(curl -s -w "%{http_code}" -X POST "${service_url}/predict" \
            -H "Content-Type: application/json" \
            -d '{
                "observations": [{
                    "speed": 25.0,
                    "steering": 0.0,
                    "sensors": [0.5, 0.5, 0.5, 0.5, 0.5]
                }],
                "deterministic": true
            }')
        
        end_time=$(date +%s%3N)
        latency=$((end_time - start_time))
        
        http_code=${response: -3}
        if [[ "$http_code" == "200" ]]; then
            echo "Request $i: ${latency}ms ✅"
        else
            echo "Request $i: HTTP $http_code ❌"
        fi
    done
}

# Check cluster resources
check_cluster_resources() {
    log_info "Checking cluster resources..."
    
    echo "Pods:"
    kubectl get pods -l app=carla-rl-serving -n ${NAMESPACE}
    
    echo -e "\nServices:"
    kubectl get svc -l app=carla-rl-serving -n ${NAMESPACE}
    
    echo -e "\nDeployment:"
    kubectl get deployment carla-rl-serving -n ${NAMESPACE}
    
    echo -e "\nPod logs (last 20 lines):"
    kubectl logs -l app=carla-rl-serving --tail=20 -n ${NAMESPACE}
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Kill port-forward if running
    if [[ -n "$PORT_FORWARD_PID" ]]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
    
    # Optionally delete resources
    if [[ "$1" == "--delete" ]]; then
        kubectl delete -f k8s/ -n ${NAMESPACE} --ignore-not-found=true
        log_success "Resources deleted"
    fi
}

# Main execution
main() {
    local delete_after_test=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --delete)
                delete_after_test=true
                shift
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --service-name)
                SERVICE_NAME="$2"
                shift 2
                ;;
            --image)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --url)
                TEST_URL="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --delete          Delete resources after testing"
                echo "  --namespace NAME  Kubernetes namespace (default: default)"
                echo "  --service-name    Service name (default: carla-rl-serving-service)"
                echo "  --image NAME      Docker image name (default: carla-rl-serving:latest)"
                echo "  --url URL         Test URL (default: auto-detect)"
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
    trap 'cleanup' EXIT
    
    # Execute test pipeline
    check_prerequisites
    build_and_load_image
    deploy_to_cluster
    
    # Get service URL
    SERVICE_URL=$(get_service_url)
    log_info "Service URL: ${SERVICE_URL}"
    
    # Wait a bit for service to be fully ready
    sleep 10
    
    # Run tests
    check_cluster_resources
    run_validation_tests "${SERVICE_URL}"
    run_performance_tests "${SERVICE_URL}"
    
    # Cleanup if requested
    if [[ "$delete_after_test" == true ]]; then
        cleanup --delete
    fi
    
    log_success "Cluster testing completed!"
}

# Run main function with all arguments
main "$@"
