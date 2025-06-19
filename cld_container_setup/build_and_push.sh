#!/bin/bash
# Build and push unified SageMaker container to ECR

set -e

# Check arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <account-id> <region> <repository-name>"
    exit 1
fi

ACCOUNT_ID=$1
REGION=$2
REPO_NAME=$3

echo "Building unified SageMaker container..."
echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"
echo "Repository: $REPO_NAME"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build the framework package
echo "Building framework package..."
cd "$SCRIPT_DIR" # Change to script directory

python setup.py sdist --dist-dir dist # Output tarball to $SCRIPT_DIR/dist/
mkdir -p code
cp dist/unified_sagemaker_framework-1.0.0.tar.gz code/

# Build Docker image
echo "Building Docker image..."
# Ensure Docker build is run from SCRIPT_DIR where Dockerfile is located
docker build -f Dockerfile -t ${REPO_NAME} . # Context is $SCRIPT_DIR

# Tag for ECR
FULL_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"
docker tag ${REPO_NAME} ${FULL_NAME}

# Get ECR login
echo "Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Create repository if it doesn't exist
echo "Creating ECR repository if needed..."
aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} || \
    aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION}

# Push to ECR
echo "Pushing image to ECR..."
docker push ${FULL_NAME}

echo "Successfully pushed ${FULL_NAME}"

# Optional: Tag with version
VERSION=$(date +%Y%m%d-%H%M%S)
VERSIONED_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${VERSION}"
docker tag ${REPO_NAME} ${VERSIONED_NAME}
docker push ${VERSIONED_NAME}

echo "Also pushed with version tag: ${VERSIONED_NAME}"

# Clean up local artifacts
echo "Cleaning up local artifacts..."
rm -rf code
rm -rf dist
rm -rf unified_sagemaker_framework.egg-info # Common name for build artifacts from setuptools

echo "Build and push completed successfully!"