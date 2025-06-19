# Unified SageMaker Container for Training and Serving
FROM ubuntu:20.04

LABEL maintainer="Custom ML Framework"
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    vim \
    openjdk-8-jdk-headless \
    libopencv-dev \
    openssh-client \
    openssh-server \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.11
ARG PYTHON_VERSION=3.11.6
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-$PYTHON_VERSION* && \
    ln -s /usr/local/bin/python3.11 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.11 /usr/bin/pip

# Upgrade pip and install base Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install SageMaker specific packages
RUN pip install --no-cache-dir \
    sagemaker-training==4.4.0 \
    sagemaker-inference==1.10.0 \
    multi-model-server==1.1.9 \
    retrying==1.3.4

# Install common ML frameworks (customize as needed)
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    scipy==1.11.1 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    requests==2.31.0 \
    boto3==1.28.0 \
    mxnet==1.9.1 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    tensorflow==2.13.0 \
    joblib==1.3.2

# Create necessary directories
RUN mkdir -p /opt/ml/code /opt/ml/model /home/model-server

# Copy the unified framework package
COPY code/unified_sagemaker_framework-1.0.0.tar.gz /unified_sagemaker_framework-1.0.0.tar.gz
RUN pip install --no-cache-dir /unified_sagemaker_framework-1.0.0.tar.gz && \
    rm /unified_sagemaker_framework-1.0.0.tar.gz

# Copy entrypoint and handler scripts
COPY scripts/unified-entrypoint.py /usr/local/bin/unified-entrypoint.py
COPY handlers/inference_handler.py /home/model-server/inference_handler.py
COPY handlers/training_handler.py /home/model-server/training_handler.py
RUN chmod +x /usr/local/bin/unified-entrypoint.py

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SAGEMAKER_TRAINING_MODULE=unified_sagemaker_framework.training:main \
    SAGEMAKER_SERVING_MODULE=unified_sagemaker_framework.serving:main

# Set working directory
WORKDIR /opt/ml

# Define entrypoint
ENTRYPOINT ["python", "/usr/local/bin/unified-entrypoint.py"]

# Default command
CMD ["serve"]