#!/bin/bash

# Media Cleanup Script for Jetson Camera Service
# Keeps only the most recent media files to prevent storage buildup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_SERVICE_DIR="$(dirname "$SCRIPT_DIR")/camera_service"

# Configuration
PHOTOS_DIR="$CAMERA_SERVICE_DIR/photos"
VIDEOS_DIR="$CAMERA_SERVICE_DIR/videos"
FACES_DIR="$CAMERA_SERVICE_DIR/faces"

# Keep only the last N files (configurable)
KEEP_PHOTOS=20
KEEP_VIDEOS=10
KEEP_FACES=50

# Log file
LOG_FILE="$CAMERA_SERVICE_DIR/logs/cleanup.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to cleanup directory
cleanup_directory() {
    local dir="$1"
    local keep_count="$2"
    local file_type="$3"
    
    if [ ! -d "$dir" ]; then
        log_message "Directory $dir does not exist, skipping"
        return
    fi
    
    # Count current files
    local current_count=$(find "$dir" -maxdepth 1 -type f | wc -l)
    
    if [ "$current_count" -le "$keep_count" ]; then
        log_message "$file_type: $current_count files (within limit of $keep_count)"
        return
    fi
    
    log_message "$file_type: Found $current_count files, keeping newest $keep_count"
    
    # Remove oldest files beyond the keep limit
    find "$dir" -maxdepth 1 -type f -printf '%T@ %p\n' | \
    sort -n | \
    head -n -"$keep_count" | \
    cut -d' ' -f2- | \
    while read -r file; do
        log_message "Removing old file: $(basename "$file")"
        rm -f "$file"
    done
    
    # Count after cleanup
    local after_count=$(find "$dir" -maxdepth 1 -type f | wc -l)
    log_message "$file_type: Cleanup complete, $after_count files remaining"
}

# Function to cleanup by age (alternative approach)
cleanup_by_age() {
    local dir="$1"
    local days="$2"
    local file_type="$3"
    
    if [ ! -d "$dir" ]; then
        return
    fi
    
    local old_files=$(find "$dir" -maxdepth 1 -type f -mtime +$days)
    local count=$(echo "$old_files" | grep -c .)
    
    if [ "$count" -gt 0 ]; then
        log_message "$file_type: Removing $count files older than $days days"
        echo "$old_files" | while read -r file; do
            if [ -n "$file" ]; then
                log_message "Removing old file: $(basename "$file")"
                rm -f "$file"
            fi
        done
    fi
}

# Main cleanup function
main() {
    log_message "=== Starting media cleanup ==="
    
    # Create directories if they don't exist
    mkdir -p "$PHOTOS_DIR" "$VIDEOS_DIR" "$FACES_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Cleanup by count (keep most recent N files)
    cleanup_directory "$PHOTOS_DIR" "$KEEP_PHOTOS" "Photos"
    cleanup_directory "$VIDEOS_DIR" "$KEEP_VIDEOS" "Videos"
    cleanup_directory "$FACES_DIR" "$KEEP_FACES" "Face embeddings"
    
    # Also cleanup very old files (older than 7 days) regardless of count
    cleanup_by_age "$PHOTOS_DIR" 7 "Photos"
    cleanup_by_age "$VIDEOS_DIR" 7 "Videos"
    cleanup_by_age "$FACES_DIR" 30 "Face embeddings"
    
    # Cleanup log files older than 30 days
    find "$(dirname "$LOG_FILE")" -name "*.log" -mtime +30 -delete 2>/dev/null
    
    log_message "=== Media cleanup completed ==="
}

# Run cleanup
main "$@" 