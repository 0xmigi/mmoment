# CV Apps Service - Isolated container for competition apps
FROM ultralytics/ultralytics:latest-jetson-jetpack6

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code
COPY main.py .

# Copy apps directory (mounted at runtime for development)
# Production: COPY ../../apps /app/apps

# Port
EXPOSE 5004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5004/api/health || exit 1

# Run
CMD ["python3", "main.py"]
