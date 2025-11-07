#!/bin/bash

# Auto-scaling Framework Deployment Script
# This script helps deploy and manage the auto-scaling framework

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="auto-scaling-framework"
DOCKER_COMPOSE_FILE="docker-compose.yml"
K8S_NAMESPACE="auto-scaling-framework"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log "Prerequisites check passed!"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Build application image
    log "Building application image..."
    docker build -t auto-scaling-app:latest ./app/
    
    # Build scaling controller image
    log "Building scaling controller image..."
    docker build -t auto-scaling-controller:latest ./scaling-controller/
    
    log "Docker images built successfully!"
}

# Start services with Docker Compose
start_services() {
    log "Starting services with Docker Compose..."
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        docker-compose up -d
        log "Services started successfully!"
    else
        error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
}

# Stop services
stop_services() {
    log "Stopping services..."
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        docker-compose down
        log "Services stopped successfully!"
    else
        error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
}

# Restart services
restart_services() {
    log "Restarting services..."
    stop_services
    start_services
}

# Check service status
check_status() {
    log "Checking service status..."
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        docker-compose ps
    else
        error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
}

# Show logs
show_logs() {
    local service=${1:-""}
    
    if [ -z "$service" ]; then
        log "Showing logs for all services..."
        docker-compose logs -f
    else
        log "Showing logs for service: $service"
        docker-compose logs -f "$service"
    fi
}

# Health check
health_check() {
    log "Performing health checks..."
    
    local services=("app" "prometheus" "grafana" "alertmanager" "scaling-controller")
    local ports=(8080 9090 3000 9093 8081)
    local endpoints=("/health" "/-/healthy" "/api/health" "/-/healthy" "/health")
    
    for i in "${!services[@]}"; do
        local service="${services[$i]}"
        local port="${ports[$i]}"
        local endpoint="${endpoints[$i]}"
        
        info "Checking $service on port $port..."
        
        if curl -f -s "http://localhost:$port$endpoint" > /dev/null 2>&1; then
            log "✓ $service is healthy"
        else
            warn "✗ $service health check failed"
        fi
    done
}

# Deploy to Kubernetes
deploy_k8s() {
    log "Deploying to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Create namespace
    kubectl apply -f k8s/namespace.yml
    
    # Deploy all components
    kubectl apply -f k8s/
    
    log "Kubernetes deployment completed!"
    
    # Show deployment status
    kubectl get all -n "$K8S_NAMESPACE"
}

# Clean up
cleanup() {
    log "Cleaning up..."
    
    # Stop and remove containers
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        docker-compose down -v --remove-orphans
    fi
    
    # Remove images
    docker rmi auto-scaling-app:latest auto-scaling-controller:latest 2>/dev/null || true
    
    # Remove volumes
    docker volume prune -f
    
    log "Cleanup completed!"
}

# Load testing
run_load_test() {
    log "Running load test..."
    
    # Check if Node.js is available
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Please install Node.js first."
        exit 1
    fi
    
    # Install dependencies if needed
    if [ ! -d "scripts/node_modules" ]; then
        log "Installing load test dependencies..."
        cd scripts && npm install axios && cd ..
    fi
    
    # Run load test
    node scripts/load-test.js http://localhost:8080 "${1:-normal}"
}

# Show usage
show_usage() {
    echo "Auto-scaling Framework Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start           Start all services with Docker Compose"
    echo "  stop            Stop all services"
    echo "  restart         Restart all services"
    echo "  status          Show service status"
    echo "  logs [SERVICE]  Show logs (all services or specific service)"
    echo "  health          Perform health checks"
    echo "  build           Build Docker images"
    echo "  deploy-k8s      Deploy to Kubernetes"
    echo "  cleanup         Clean up all resources"
    echo "  load-test [TYPE] Run load test (normal|spike|stress|resource)"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start all services"
    echo "  $0 logs app                 # Show application logs"
    echo "  $0 load-test spike          # Run spike load test"
    echo "  $0 deploy-k8s               # Deploy to Kubernetes"
}

# Main script logic
main() {
    case "${1:-help}" in
        "start")
            check_prerequisites
            build_images
            start_services
            health_check
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "status")
            check_status
            ;;
        "logs")
            show_logs "$2"
            ;;
        "health")
            health_check
            ;;
        "build")
            check_prerequisites
            build_images
            ;;
        "deploy-k8s")
            deploy_k8s
            ;;
        "cleanup")
            cleanup
            ;;
        "load-test")
            run_load_test "$2"
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Run main function
main "$@" 