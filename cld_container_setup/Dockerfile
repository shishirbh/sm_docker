# Unified SageMaker Container for Training and Serving


# Start from a slim Python base image
FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    python3-dev \
    build-essential \
    ca-certificates \
    curl \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*


# Set environment variables to prevent buffering of logs
ENV PYTHONUNBUFFERED=1

# Set up a working directory
WORKDIR /opt/ml

# Copy the requirements file first to leverage Docker cache
# Using the file path from your repository structure
COPY examples/requirements.txt /opt/ml/requirements.txt

# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

# Copy the framework source code and setup file
COPY src/ /opt/ml/src/
COPY setup.py /opt/ml/

# Install the custom framework package itself
RUN pip install --no-cache-dir .

# Copy the container scripts and handlers
COPY scripts/unified-entrypoint.py /home/model-server/unified-entrypoint.py
COPY handlers/ /home/model-server/

# Set permissions for the entrypoint
RUN chmod +x /home/model-server/unified-entrypoint.py

# Set the entrypoint for the container
ENTRYPOINT ["python", "/home/model-server/unified-entrypoint.py"]