"""
Example custom training handler that can be extended for specific use cases.
"""
import os
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def pre_training(training_env):
    """
    Custom pre-training setup.
    
    Args:
        training_env: SageMaker training environment object
    """
    logger.info("Custom pre-training setup...")
    
    # Example: Validate hyperparameters
    required_params = ['learning_rate', 'epochs', 'batch_size']
    for param in required_params:
        if param not in training_env.hyperparameters:
            logger.warning(f"Missing hyperparameter: {param}")
    
    # Example: Setup custom directories
    custom_dir = Path(training_env.output_data_dir) / 'custom_outputs'
    custom_dir.mkdir(parents=True, exist_ok=True)
    
    # Example: Log data channels
    logger.info("Available data channels:")
    for channel, path in training_env.channel_input_dirs.items():
        files = list(Path(path).glob('*'))
        logger.info(f"  {channel}: {len(files)} files")

def post_training(training_env):
    """
    Custom post-training cleanup and artifact handling.
    
    Args:
        training_env: SageMaker training environment object
    """
    logger.info("Custom post-training cleanup...")
    
    # Example: Save training metadata
    metadata = {
        'hyperparameters': training_env.hyperparameters,
        'job_name': training_env.job_name,
        'framework': os.environ.get('TRAINING_FRAMEWORK', 'custom'),
        'channels': list(training_env.channel_input_dirs.keys())
    }
    
    metadata_path = Path(training_env.model_dir) / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Example: Copy additional artifacts
    custom_outputs = Path(training_env.output_data_dir) / 'custom_outputs'
    if custom_outputs.exists():
        for file in custom_outputs.glob('*'):
            shutil.copy2(file, training_env.model_dir)

def train(training_env):
    """
    Custom training logic.
    
    Args:
        training_env: SageMaker training environment object
    """
    logger.info("Starting custom training logic...")
    
    # Get hyperparameters
    hp = training_env.hyperparameters
    epochs = int(hp.get('epochs', 10))
    batch_size = int(hp.get('batch_size', 32))
    learning_rate = float(hp.get('learning_rate', 0.001))
    
    logger.info(f"Training config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    # Load data from channels
    train_path = Path(training_env.channel_input_dirs.get('train', ''))
    val_path = Path(training_env.channel_input_dirs.get('validation', ''))
    
    # Example: Custom model training
    # This is where you would implement your actual training logic
    import time
    import random
    
    best_metric = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Simulate training
        time.sleep(1)  # Replace with actual training
        
        # Simulate metrics
        train_loss = random.uniform(0.1, 1.0) * (1 - epoch / epochs)
        val_loss = train_loss + random.uniform(0.05, 0.15)
        
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_metric:
            best_metric = val_loss
            # Save your model here
            save_model(training_env.model_dir, epoch, train_loss, val_loss)
    
    logger.info(f"Training completed. Best validation loss: {best_metric:.4f}")

def save_model(model_dir, epoch, train_loss, val_loss):
    """
    Save model artifacts.
    
    Args:
        model_dir: Directory to save model
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
    """
    # Example: Save a dummy model file
    model_path = Path(model_dir) / 'model.pkl'
    
    # In real implementation, serialize your actual model
    import pickle
    dummy_model = {
        'type': 'custom_model',
        'epoch': epoch,
        'metrics': {
            'train_loss': train_loss,
            'val_loss': val_loss
        },
        'weights': {}  # Your model weights would go here
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(dummy_model, f)
    
    logger.info(f"Model saved to {model_path}")