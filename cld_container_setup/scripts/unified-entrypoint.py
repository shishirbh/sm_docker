#!/usr/bin/env python
"""
Unified entrypoint for SageMaker container that handles both training and serving.
"""
import os
import sys
import json
import subprocess
import shlex
import logging
from retrying import retry
from subprocess import CalledProcessError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _retry_if_error(exception):
    """Retry condition for server start."""
    return isinstance(exception, (CalledProcessError, OSError))

@retry(stop_max_delay=50000, retry_on_exception=_retry_if_error)
def start_model_server():
    """Start the Multi Model Server for inference."""
    from sagemaker_inference import model_server
    
    # Configure model server settings
    if os.environ.get('SAGEMAKER_MULTI_MODEL', 'false').lower() == 'true':
        # Multi-model endpoint configuration
        logger.info("Starting Multi-Model Server...")
        os.environ.setdefault('SAGEMAKER_MODEL_SERVER_WORKERS', '1')
        handler_service = os.environ.get(
            'SAGEMAKER_INFERENCE_HANDLER',
            '/home/model-server/inference_handler.py:handle'
        )
    else:
        # Single model endpoint configuration
        logger.info("Starting Single Model Server...")
        handler_service = os.environ.get(
            'SAGEMAKER_INFERENCE_HANDLER',
            '/home/model-server/inference_handler.py:handle'
        )
    
    model_server.start_model_server(handler_service=handler_service)

def start_training():
    """Start the training process."""
    logger.info("Starting training...")
    
    # Import and run the training module
    from unified_sagemaker_framework import training
    training.main()

def execute_command(cmd):
    """Execute a shell command."""
    logger.info(f"Executing command: {cmd}")
    subprocess.check_call(shlex.split(cmd))

def main():
    """Main entrypoint logic."""
    if len(sys.argv) < 2:
        logger.error("No command specified. Use 'train' or 'serve'")
        sys.exit(1)
    
    command = sys.argv[1]
    logger.info(f"Container started with command: {command}")
    
    # Log environment for debugging
    logger.info("Environment variables:")
    for key, value in sorted(os.environ.items()):
        if key.startswith('SM_') or key.startswith('SAGEMAKER_'):
            logger.info(f"  {key}={value}")
    
    try:
        if command == "train":
            start_training()
        elif command == "serve":
            for var in ("SM_TRAINING_ENV", "SM_USER_ARGS"):
                os.environ.pop(var, None)
            start_model_server()
        else:
            # Execute custom command
            execute_command(" ".join(sys.argv[1:]))
        
        # Keep container running if needed
        if command == "serve":
            logger.info("Model server started. Keeping container alive...")
            subprocess.call(["tail", "-f", "/dev/null"])
            
    except Exception as e:
        logger.error(f"Error in container: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
