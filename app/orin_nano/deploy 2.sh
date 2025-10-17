#!/bin/bash

# mmoment Computer Vision Platform Deployment Script
# Manages the 3-container architecture:
# - Camera Service (Port 5002) - GPU-accelerated YOLOv8 + InsightFace
# - Biometric Security (Port 5003) - Encryption & secure purging
# - Solana Middleware (Port 5001) - Blockchain operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker compose.yml"
PROJECT_NAME="mmoment-platform"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if nvidia-docker is available
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        print_warning "nvidia-docker runtime not available or GPU not accessible"
        print_warning "Camera service may not have GPU acceleration"
    else
        print_success "NVIDIA Docker runtime is available"
    fi
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    check_docker
    
    # Check NVIDIA Docker
    check_nvidia_docker
    
    # Check available disk space
    available_space=$(df . | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 5242880 ]; then  # 5GB in KB
        print_warning "Less than 5GB of disk space available"
    fi
    
    # Check if ports are available
    for port in 5001 5002 5003; do
        if netstat -ln | grep -q ":$port "; then
            print_warning "Port $port is already in use"
        fi
    done
    
    print_success "System requirements check completed"
}

# Function to build all containers
build_containers() {
    print_status "Building mmoment platform containers..."
    
    # Build biometric security container first (no dependencies)
    print_status "Building biometric security container..."
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME build biometric-security
    
    # Build solana middleware
    print_status "Building solana middleware container..."
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME build solana-middleware
    
    # Build camera service (depends on the others)
    print_status "Building camera service container..."
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME build camera-service
    
    print_success "All containers built successfully"
}

# Function to start the platform
start_platform() {
    print_status "Starting mmoment computer vision platform..."
    
    # Start services in order
    print_status "Starting biometric security service..."
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d biometric-security
    
    print_status "Starting solana middleware..."
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d solana-middleware
    
    print_status "Starting camera service..."
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d camera-service
    
    print_success "Platform started successfully"
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    check_service_health
}

# Function to stop the platform
stop_platform() {
    print_status "Stopping mmoment platform..."
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down
    print_success "Platform stopped"
}

# Function to restart the platform
restart_platform() {
    print_status "Restarting mmoment platform..."
    stop_platform
    start_platform
}

# Function to check service health
check_service_health() {
    print_status "Checking service health..."
    
    services=("biometric-security:5003" "solana-middleware:5001" "camera-service:5002")
    
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if curl -f -s "http://localhost:$port/api/health" > /dev/null; then
            print_success "$name service is healthy (port $port)"
        else
            print_error "$name service is not responding (port $port)"
        fi
    done
}

# Function to show platform status
show_status() {
    print_status "mmoment Platform Status:"
    echo ""
    
    # Show container status
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
    echo ""
    
    # Show service endpoints
    print_status "Service Endpoints:"
    echo "🎥 Camera Service:      http://localhost:5002"
    echo "🔐 Biometric Security:  http://localhost:5003"
    echo "⛓️  Solana Middleware:   http://localhost:5001"
    echo ""
    
    # Check health
    check_service_health
}

# Function to show logs
show_logs() {
    service=${1:-}
    
    if [ -z "$service" ]; then
        print_status "Showing logs for all services..."
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs --tail=100 -f
    else
        print_status "Showing logs for $service..."
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs --tail=100 -f $service
    fi
}

# Function to clean up everything
cleanup() {
    print_status "Cleaning up mmoment platform..."
    
    # Stop containers
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down
    
    # Remove images
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down --rmi all
    
    # Remove volumes (WARNING: This deletes all data)
    read -p "Do you want to remove all volumes (this will delete all data)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v
        print_warning "All volumes and data have been removed"
    fi
    
    print_success "Cleanup completed"
}

# Function to run development mode
dev_mode() {
    print_status "Starting platform in development mode..."
    print_warning "This will mount source code volumes for live editing"
    
    # Start with development compose file if it exists
    if [ -f "docker compose.dev.yml" ]; then
        docker compose -f $COMPOSE_FILE -f docker compose.dev.yml -p $PROJECT_NAME up --build
    else
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME up --build
    fi
}

# Function to show help
show_help() {
    echo "mmoment Computer Vision Platform Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build all platform containers"
    echo "  start       Start the platform"
    echo "  stop        Stop the platform"
    echo "  restart     Restart the platform"
    echo "  status      Show platform status"
    echo "  logs        Show logs for all services"
    echo "  logs <service>  Show logs for specific service (camera-service, biometric-security, solana-middleware)"
    echo "  health      Check service health"
    echo "  dev         Start in development mode with live code mounting"
    echo "  cleanup     Stop platform and remove all containers/images"
    echo "  requirements Check system requirements"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build && $0 start    # Build and start platform"
    echo "  $0 logs camera-service  # Show camera service logs"
    echo "  $0 status               # Check platform status"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_requirements
        build_containers
        ;;
    "start")
        start_platform
        ;;
    "stop")
        stop_platform
        ;;
    "restart")
        restart_platform
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs $2
        ;;
    "health")
        check_service_health
        ;;
    "dev")
        dev_mode
        ;;
    "cleanup")
        cleanup
        ;;
    "requirements")
        check_requirements
        ;;
    "help"|*)
        show_help
        ;;
esac 