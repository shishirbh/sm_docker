"""
Serving module for unified SageMaker framework.
Handles model serving configuration and initialization.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def configure_mms():
    """Configure Multi Model Server settings."""
    # Set MMS configuration
    mms_config = {
        'default_response_timeout': os.environ.get('MMS_DEFAULT_RESPONSE_TIMEOUT', '60'),
        'default_workers_per_model': os.environ.get('SAGEMAKER_MODEL_SERVER_WORKERS', '1'),
        'job_queue_size': os.environ.get('MMS_JOB_QUEUE_SIZE', '100'),
        'load_models': os.environ.get('MMS_LOAD_MODELS', 'all'),
        'max_request_size': os.environ.get('MMS_MAX_REQUEST_SIZE', '6553600'),
        'max_response_size': os.environ.get('MMS_MAX_RESPONSE_SIZE', '6553600'),
        'model_store': os.environ.get('MMS_MODEL_STORE', '/opt/ml/model'),
    }
    
    # Write MMS config file if needed
    config_path = '/home/model-server/config.properties'
    if not os.path.exists(config_path):
        logger.info("Creating MMS configuration file...")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            for key, value in mms_config.items():
                f.write(f"{key}={value}\n")
    
    return config_path

def setup_model_handler():
    """Setup model handler for inference."""
    # Check if custom handler is specified
    custom_handler = os.environ.get('SAGEMAKER_INFERENCE_HANDLER')
    
    if custom_handler and os.path.exists(custom_handler):
        logger.info(f"Using custom inference handler: {custom_handler}")
        return custom_handler
    
    # Default handler
    default_handler = '/home/model-server/inference_handler.py:handle'
    logger.info(f"Using default inference handler: {default_handler}")
    return default_handler

def validate_model_artifacts():
    """Validate model artifacts are present."""
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory does not exist: {model_dir}")
        return False
    
    model_files = list(Path(model_dir).glob('*'))
    if not model_files:
        logger.warning(f"No model files found in: {model_dir}")
        return False
    
    logger.info(f"Found {len(model_files)} model files:")
    for file in model_files[:5]:  # Show first 5 files
        logger.info(f"  - {file.name}")
    
    return True

def setup_environment():
    """Setup serving environment."""
    # Add model directory to Python path if needed
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    
    # Set up custom module paths
    custom_modules = os.environ.get('SAGEMAKER_CUSTOM_MODULES', '')
    if custom_modules:
        for module_path in custom_modules.split(':'):
            if os.path.exists(module_path) and module_path not in sys.path:
                sys.path.insert(0, module_path)
    
    # Log environment
    logger.info("Serving environment configured:")
    logger.info(f"  Model directory: {model_dir}")
    logger.info(f"  Python path: {sys.path[:3]}...")  # Show first 3 paths

def main():
    """Main entry point for serving."""
    logger.info("Initializing unified serving framework...")
    
    try:
        # Setup environment
        setup_environment()
        
        # Configure MMS
        config_path = configure_mms()
        
        # Validate model artifacts
        if not validate_model_artifacts():
            logger.warning("Model validation failed, but continuing...")
        
        # Setup model handler
        handler = setup_model_handler()
        
        # Log configuration
        logger.info("Serving configuration:")
        logger.info(f"  MMS config: {config_path}")
        logger.info(f"  Model handler: {handler}")
        logger.info(f"  Multi-model: {os.environ.get('SAGEMAKER_MULTI_MODEL', 'false')}")
        
        logger.info("Serving initialization completed!")
        
    except Exception as e:
        logger.error(f"Serving initialization failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()