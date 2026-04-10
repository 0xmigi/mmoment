#!/bin/bash

# Set up virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set up environment variables if .env file exists
if [ -f ".env" ]; then
    echo "Loading environment from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found. Using default values."
fi

# Check if the middleware port is already in use
if lsof -Pi :${MIDDLEWARE_PORT:-5002} -sTCP:LISTEN -t >/dev/null ; then
    echo "Port ${MIDDLEWARE_PORT:-5002} is already in use. Stopping the existing process..."
    lsof -ti:${MIDDLEWARE_PORT:-5002} | xargs kill -9
fi

# Run the middleware
echo "Starting middleware server..."
python middleware.py 