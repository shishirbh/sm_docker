# Unified SageMaker Container - Example Usage
# This notebook demonstrates how to use the unified container for training and inference

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.predictor import Predictor
import json
import numpy as np

# Setup
session = sagemaker.Session()
role = get_execution_role()
account_id = boto3.client('sts').get_caller_identity()['Account']
region = session.boto_region_name
bucket = session.default_bucket()

# Container configuration
ecr_repository = 'unified-sagemaker-container'
container_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repository}:latest"

print(f"Role: {role}")
print(f"Bucket: {bucket}")
print(f"Container: {container_uri}")

# %% [markdown]
# ## 1. Prepare Training Data

# %%
# Create dummy training data
import pandas as pd
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
train_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
train_data['target'] = y

# Split into train and validation
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)

# Save to CSV
train_df.to_csv('train.csv', index=False)
val_df.to_csv('validation.csv', index=False)

# Upload to S3
prefix = 'unified-container-demo'
train_path = session.upload_data('train.csv', bucket=bucket, key_prefix=f'{prefix}/train')
val_path = session.upload_data('validation.csv', bucket=bucket, key_prefix=f'{prefix}/validation')

print(f"Training data: {train_path}")
print(f"Validation data: {val_path}")

# %% [markdown]
# ## 2. Training with Default Handler

# %%
# Create estimator with default training
estimator = Estimator(
    image_uri=container_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    base_job_name='unified-container-training',
    hyperparameters={
        'epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.01,
        'model_type': 'sklearn'  # or 'pytorch', 'tensorflow', etc.
    },
    environment={
        'PYTHONUNBUFFERED': '1'
    }
)

# Define input channels
from sagemaker.inputs import TrainingInput
train_input = TrainingInput(train_path, content_type='text/csv')
val_input = TrainingInput(val_path, content_type='text/csv')

# Start training
estimator.fit({'train': train_input, 'validation': val_input})

# %% [markdown]
# ## 3. Training with Custom Handler

# %%
# Create custom training script
custom_train_script = '''
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()
    
    # Load data
    train_df = pd.read_csv(os.path.join(args.train, 'train.csv'))
    X = train_df.drop('target', axis=1)
    y = train_df['target']
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    print(f"Model saved to {args.model_dir}")

if __name__ == '__main__':
    main()
'''

with open('custom_train.py', 'w') as f:
    f.write(custom_train_script)

# %%
# Use custom training script
from sagemaker.pytorch import PyTorch

custom_estimator = PyTorch(
    entry_point='custom_train.py',
    image_uri=container_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    base_job_name='unified-custom-training',
    hyperparameters={
        'n-estimators': 200,
        'max-depth': 15
    }
)

custom_estimator.fit({'train': train_input})

# %% [markdown]
# ## 4. Deploy Single Model Endpoint

# %%
# Deploy the trained model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='unified-container-endpoint'
)

# Test prediction
test_data = X[:5].tolist()
response = predictor.predict(json.dumps({'instances': test_data}))
print(f"Predictions: {response}")

# %% [markdown]
# ## 5. Deploy Multi-Model Endpoint

# %%
from sagemaker.multidatamodel import MultiDataModel

# Create multiple model artifacts
model_artifacts = []
for i in range(3):
    # Create variations of the model
    estimator_variant = Estimator(
        image_uri=container_uri,
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        hyperparameters={
            'epochs': 5 + i,
            'learning_rate': 0.01 * (i + 1)
        }
    )
    estimator_variant.fit({'train': train_input}, wait=False)
    model_artifacts.append(f"model_v{i}.tar.gz")

# %%
# Create multi-model endpoint
model_prefix = f"s3://{bucket}/{prefix}/models/"

mme_model = Model(
    image_uri=container_uri,
    model_data=model_prefix,
    role=role,
    env={
        'SAGEMAKER_MULTI_MODEL': 'true',
        'SAGEMAKER_MODEL_SERVER_WORKERS': '2'
    }
)

mme = MultiDataModel(
    name='unified-multi-model',
    model_model=mme_model,
    model_data_prefix=model_prefix
)

# Deploy MME
mme_predictor = mme.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='unified-mme-endpoint'
)

# %%
# Invoke specific models
for model_name in ['model_v0.tar.gz', 'model_v1.tar.gz', 'model_v2.tar.gz']:
    response = mme_predictor.predict(
        json.dumps({'instances': test_data}),
        target_model=model_name
    )
    print(f"Predictions from {model_name}: {response}")

# %% [markdown]
# ## 6. Custom Inference Handler

# %%
# Create custom inference handler
custom_handler = '''
from handlers.inference_handler import BaseModelHandler
import joblib
import numpy as np

class CustomInferenceHandler(BaseModelHandler):
    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        self.model = joblib.load(f"{model_dir}/model.joblib")
        self.initialized = True
    
    def preprocess(self, request):
        # Custom preprocessing
        data = request[0].get('body')
        if isinstance(data, bytes):
            data = json.loads(data.decode('utf-8'))
        return np.array(data['instances'])
    
    def inference(self, model_input):
        # Get predictions and probabilities
        predictions = self.model.predict(model_input)
        probabilities = self.model.predict_proba(model_input)
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def postprocess(self, inference_output):
        # Format output
        return {
            'predictions': inference_output['predictions'].tolist(),
            'probabilities': inference_output['probabilities'].tolist()
        }
'''

# Save handler
with open('custom_inference_handler.py', 'w') as f:
    f.write(custom_handler)

# %%
# Deploy with custom handler
custom_predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    environment={
        'SAGEMAKER_HANDLER_CLASS': 'custom_inference_handler.CustomInferenceHandler'
    }
)

# %% [markdown]
# ## 7. Batch Transform

# %%
# Create transformer for batch inference
transformer = estimator.transformer(
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/{prefix}/batch-output'
)

# Run batch transform
transformer.transform(
    data=train_path,
    content_type='text/csv',
    split_type='Line'
)

print(f"Batch transform output: {transformer.output_path}")

# %% [markdown]
# ## 8. Clean Up

# %%
# Delete endpoints
predictor.delete_endpoint()
mme_predictor.delete_endpoint()

# Clean up local files
import os
for file in ['train.csv', 'validation.csv', 'custom_train.py', 'custom_inference_handler.py']:
    if os.path.exists(file):
        os.remove(file)

print("Cleanup completed!")

# %% [markdown]
# ## Summary
# 
# This notebook demonstrated:
# 1. Training with the unified container using default handlers
# 2. Training with custom scripts
# 3. Deploying single model endpoints
# 4. Deploying multi-model endpoints
# 5. Using custom inference handlers
# 6. Running batch transforms
# 
# The unified container provides flexibility to handle various ML frameworks and custom requirements while maintaining a consistent interface for SageMaker.