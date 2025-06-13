# Biometric Security Container
# Handles facial embedding encryption, NFT packaging, and secure purging
# CPU-only container (no GPU needed)

FROM python:3.9-slim

# Security: Run as non-root user
RUN groupadd -r biouser && useradd -r -g biouser biouser

# Install system dependencies for secure operations
RUN apt-get update && apt-get install -y \
    secure-delete \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create secure directories with proper permissions
RUN mkdir -p /app/secure_data /app/temp_embeddings /app/logs && \
    chown -R biouser:biouser /app && \
    chmod -R 750 /app/secure_data /app/temp_embeddings

# Switch to non-root user
USER biouser

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5003/api/health || exit 1

# Start the biometric security service
CMD ["python", "main.py"] 