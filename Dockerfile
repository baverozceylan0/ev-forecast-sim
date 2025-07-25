FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy startup script
COPY start.sh .
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Default command (overridden by docker-compose)
CMD ["/bin/bash", "start.sh"]