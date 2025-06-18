# Unified SageMaker Containers

This repository contains example Docker resources for custom training and serving on Amazon SageMaker.

## Project Layout

- **cld_container_setup** – Python package providing a unified framework for training and inference along with example handlers and tests.
- **training** – Dockerfile and build script for the training container.
- **serving** – Dockerfile and utilities for the inference container.

## Building and Pushing Images

Each component folder includes a `build_and_push.sh` script. A typical workflow is:

```bash
# From repository root
cd cld_container_setup && ./build_and_push.sh <account-id> <region> unified-sagemaker-container
```

See `cld_container_setup/quickstart.md` for detailed steps and additional examples.

## Running Unit Tests

The Python tests are located under `cld_container_setup/tests` and can be executed with:

```bash
pytest
```

Ensure required dependencies such as `numpy`, `boto3` and the SageMaker SDK are installed before running the tests.
