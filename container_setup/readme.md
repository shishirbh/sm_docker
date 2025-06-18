# Unified SageMaker Container for Custom ML Models

This repository provides a unified Docker container solution that supports both training and inference for custom machine learning models on Amazon SageMaker.

## Features

- **Unified Container**: Single Docker image for both training and serving
- **Multi-Framework Support**: Built-in support for PyTorch, TensorFlow, scikit-learn, and MXNet
- **Extensible**: Easy to add custom training and inference logic
- **Multi-Model Endpoints**: Support for hosting multiple models on a single endpoint
- **Flexible Configuration**: Environment variables for customization

## Repository Structure

```
.
├── Dockerfile                    # Unified container definition
├── setup.py                      # Package setup file
├── build_and_push.sh            # Build and push script
├── src/
│   └── unified_sagemaker_framework/
│       ├── __init__.py
│       ├── training.py          # Training module
│       └── serving.py           # Serving module
├── scripts/
│   └── unified-entrypoint.py   # Container entrypoint
├── handlers/
│   ├── inference_handler.py    # Default inference handler
│   └── training_handler.py     # Custom training handler example
├── examples/
│   ├── train.py                # Example training script
│   ├── custom_model.py         # Example custom model
│   └── notebook.ipynb          # Example notebook
└── tests/
    └── test_container.py       # Container tests
```

## Quick Start

### 1. Build and Push the Container

```bash
# Set your AWS account details
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=$(aws configure get region)
export ECR_REPOSITORY="unified-sagemaker-container"

# Build and push
./build_and_push.sh $AWS_ACCOUNT_ID $AWS_REGION $ECR_REPOSITORY
```

### 2. Training Example

```python
import sagemaker
from sagemaker.estimator import Estimator

# Define the estimator
estimator = Estimator(
    image_uri=f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPOSITORY}:latest",
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m5.xlarge',
    hyperparameters={
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001
    },
    environment={
        'SAGEMAKER_TRAINING_HANDLER': '/home/model-server/training_handler.py'
    }
)

# Start training
estimator.fit({
    'train': 's3://your-bucket/train',
    'validation': 's3://your-bucket/validation'
})
```

### 3. Deployment Example

```python
# Deploy as endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    environment={
        'SAGEMAKER_MULTI_MODEL': 'false',  # Set to 'true' for multi-model
        'SAGEMAKER_HANDLER_CLASS': 'handlers.custom.MyModelHandler'  # Optional
    }
)

# Make predictions
result = predictor.predict(data)
```

## Customization

### Custom Training Script

Create your training script (e.g., `train.py`):

```python
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    # SageMaker passes data directories as environment variables
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()
    
    # Your training logic here
    train_model(args)
    
    # Save model to args.model_dir
    save_model(args.model_dir)

if __name__ == '__main__':
    main()
```

### Custom Inference Handler

Create a custom handler by extending `BaseModelHandler`:

```python
from handlers.inference_handler import BaseModelHandler

class MyCustomHandler(BaseModelHandler):
    def initialize(self, context):
        # Load your model
        model_dir = context.system_properties.get("model_dir")
        self.model = load_my_model(model_dir)
        self.initialized = True
    
    def preprocess(self, request):
        # Transform input data
        return preprocess_data(request)
    
    def inference(self, model_input):
        # Run inference
        return self.model.predict(model_input)
    
    def postprocess(self, inference_output):
        # Format output
        return format_output(inference_output)
```

## Environment Variables

### Training Configuration

- `SAGEMAKER_TRAINING_HANDLER`: Path to custom training handler
- `TRAINING_FRAMEWORK`: Framework name for metadata

### Serving Configuration

- `SAGEMAKER_MULTI_MODEL`: Enable multi-model endpoint ('true'/'false')
- `SAGEMAKER_MODEL_SERVER_WORKERS`: Number of workers per model
- `SAGEMAKER_INFERENCE_HANDLER`: Path to custom inference handler
- `SAGEMAKER_HANDLER_CLASS`: Python class for model handling
- `SAGEMAKER_MODEL_HANDLER`: Path to custom model loading logic

## Multi-Model Endpoint Support

To use multi-model endpoints:

```python
from sagemaker.model import Model
from sagemaker.multidatamodel import MultiDataModel

# Create base model
model = Model(
    image_uri=container_uri,
    model_data=model_artifacts_uri,
    role=role,
    env={'SAGEMAKER_MULTI_MODEL': 'true'}
)

# Create multi-data model
mme = MultiDataModel(
    name=model_name,
    model=model,
    model_data_prefix=s3_model_prefix
)

# Deploy
predictor = mme.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Invoke specific model
predictor.predict(data, target_model='model1.tar.gz')
```

## Supported Model Formats

The container automatically detects and loads:

- **PyTorch**: `.pt`, `.pth` files
- **TensorFlow**: `saved_model.pb` directories
- **scikit-learn**: `.pkl`, `.joblib` files
- **MXNet**: `-symbol.json` files with parameters
- **Custom**: Any format with custom handler

## Best Practices

1. **Model Artifacts**: Save all model files in the model directory during training
2. **Metadata**: Include training metadata for reproducibility
3. **Logging**: Use standard Python logging for debugging
4. **Error Handling**: Implement proper error handling in custom handlers
5. **Memory Management**: Raise `MemoryError` for OOM conditions in multi-model endpoints

## Extending the Container

To add new frameworks or features:

1. Update the Dockerfile to install required dependencies
2. Extend the inference handler to support new model formats
3. Add framework-specific logic to the training module
4. Update environment variable handling as needed

## Testing

```bash
# Local testing with SageMaker local mode
pip install sagemaker[local]

# Run tests
pytest tests/

# Test container locally
docker run -e SAGEMAKER_PROGRAM=train.py \
           -v $(pwd)/test_data:/opt/ml/input/data/train \
           $CONTAINER_URI train
```

## Troubleshooting

### Common Issues

1. **Container fails to start**: Check entrypoint logs
2. **Model loading fails**: Verify model format and handler
3. **OOM errors**: Adjust instance type or model server workers
4. **Inference errors**: Check preprocessing/postprocessing logic

### Debug Mode

Enable debug logging:

```python
estimator = Estimator(
    # ... other parameters ...
    environment={
        'PYTHONUNBUFFERED': '1',
        'SAGEMAKER_LOG_LEVEL': 'DEBUG'
    }
)
```

## License

This project is licensed under the MIT License.