# Quick Start Guide - Unified SageMaker Container

## Prerequisites
- AWS CLI configured with appropriate credentials
- Docker installed and running
- Python 3.7+ with boto3 and sagemaker SDK
- ECR access permissions

## Step 1: Clone and Setup

```bash
# Clone the repository (or create the structure)
git clone <your-repo-url> unified-sagemaker-container
cd unified-sagemaker-container

# Make build script executable
chmod +x build_and_push.sh
```

## Step 2: Build and Push Container

```bash
# Get AWS account details
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=$(aws configure get region)
export ECR_REPOSITORY="unified-sagemaker-container"

# Build and push to ECR
./build_and_push.sh $AWS_ACCOUNT_ID $AWS_REGION $ECR_REPOSITORY
```

## Step 3: Quick Training Example

```python
import sagemaker
from sagemaker.estimator import Estimator

# Setup
role = sagemaker.get_execution_role()
container_uri = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPOSITORY}:latest"

# Create estimator
estimator = Estimator(
    image_uri=container_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    hyperparameters={
        'epochs': 10,
        'learning_rate': 0.001
    }
)

# Train
estimator.fit({'train': 's3://your-bucket/train-data'})
```

## Step 4: Quick Deployment

```python
# Deploy trained model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Make prediction
result = predictor.predict({'instances': [[1, 2, 3, 4]]})
print(result)

# Cleanup
predictor.delete_endpoint()
```

## Common Use Cases

### 1. Custom Training Script
```python
# Save as train.py
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()
    
    # Your training logic here
    train_model(args)

if __name__ == '__main__':
    main()
```

### 2. Custom Inference Handler
```python
# Save as custom_handler.py
from handlers.inference_handler import BaseModelHandler

class MyHandler(BaseModelHandler):
    def initialize(self, context):
        # Load your model
        pass
    
    def preprocess(self, request):
        # Process input
        pass
    
    def inference(self, model_input):
        # Run prediction
        pass
    
    def postprocess(self, output):
        # Format output
        pass
```

### 3. Multi-Model Endpoint
```python
# Deploy multiple models
model = Model(
    image_uri=container_uri,
    model_data=s3_model_prefix,
    role=role,
    env={'SAGEMAKER_MULTI_MODEL': 'true'}
)

# Create endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Invoke specific model
result = predictor.predict(
    data,
    target_model='model_v1.tar.gz'
)
```

## Environment Variables Reference

### Training
- `SAGEMAKER_TRAINING_HANDLER`: Custom training handler path
- `TRAINING_FRAMEWORK`: Framework name for metadata

### Serving
- `SAGEMAKER_MULTI_MODEL`: Enable multi-model ('true'/'false')
- `SAGEMAKER_MODEL_SERVER_WORKERS`: Workers per model
- `SAGEMAKER_HANDLER_CLASS`: Custom handler class

## Troubleshooting

### Container Build Fails
```bash
# Check Docker daemon
docker ps

# Clean Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -f Dockerfile -t test .
```

### Training Job Fails
```python
# Enable debug logging
estimator = Estimator(
    # ... other params ...
    environment={
        'PYTHONUNBUFFERED': '1',
        'SAGEMAKER_LOG_LEVEL': 'DEBUG'
    }
)
```

### Inference Errors
```python
# Test locally
from sagemaker.local import LocalSession

local_session = LocalSession()
local_session.config = {'local': {'local_code': True}}

estimator = Estimator(
    # ... other params ...
    sagemaker_session=local_session
)
```

## Next Steps
1. Review the [full documentation](README.md)
2. Check out [example notebooks](examples/notebook.ipynb)
3. Customize handlers for your use case
4. Add your preferred ML frameworks to the Dockerfile