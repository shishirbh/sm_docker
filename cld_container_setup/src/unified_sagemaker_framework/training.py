"""
Training module for unified SageMaker framework.
Handles custom training scripts with configurable backends.
"""
import os
import sys
import json
import logging
import importlib.util
from pathlib import Path
from sagemaker_training import entry_point, environment, files, runner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedTrainingFramework:
    """Unified training framework that supports custom training scripts."""
    
    def __init__(self, training_env=None):
        self.training_env = training_env or environment.Environment()
        self.setup_paths()
        
    def setup_paths(self):
        """Setup necessary paths for training."""
        # Ensure model directory exists
        os.makedirs(self.training_env.model_dir, exist_ok=True)
        
        # Add code directory to Python path
        if self.training_env.module_dir not in sys.path:
            sys.path.insert(0, self.training_env.module_dir)
            
    def load_custom_handler(self):
        """Load custom training handler if specified."""
        handler_path = os.environ.get('SAGEMAKER_TRAINING_HANDLER')
        if handler_path and os.path.exists(handler_path):
            logger.info(f"Loading custom training handler from {handler_path}")
            spec = importlib.util.spec_from_file_location("custom_handler", handler_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return None
    
    def pre_training_hook(self):
        """Execute pre-training setup."""
        logger.info("Executing pre-training hook...")
        
        # Load hyperparameters
        hyperparameters = self.training_env.hyperparameters
        logger.info(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
        
        # Log input data configuration
        logger.info("Input data configuration:")
        for channel, path in self.training_env.channel_input_dirs.items():
            logger.info(f"  {channel}: {path}")
            
        # Execute custom pre-training logic
        custom_handler = self.load_custom_handler()
        if custom_handler and hasattr(custom_handler, 'pre_training'):
            custom_handler.pre_training(self.training_env)
    
    def post_training_hook(self):
        """Execute post-training cleanup."""
        logger.info("Executing post-training hook...")
        
        # Execute custom post-training logic
        custom_handler = self.load_custom_handler()
        if custom_handler and hasattr(custom_handler, 'post_training'):
            custom_handler.post_training(self.training_env)
        
        # Log model artifacts
        model_files = list(Path(self.training_env.model_dir).glob('*'))
        logger.info(f"Model artifacts saved: {[f.name for f in model_files]}")
    
    def train(self):
        """Execute the training process."""
        try:
            # Pre-training setup
            self.pre_training_hook()
            
            # Check if we should use custom training logic
            custom_handler = self.load_custom_handler()
            if custom_handler and hasattr(custom_handler, 'train'):
                logger.info("Using custom training logic...")
                custom_handler.train(self.training_env)
            else:
                # Use standard entry point execution
                logger.info("Using standard entry point execution...")
                self._run_user_script()
            
            # Post-training cleanup
            self.post_training_hook()
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
    
    def _run_user_script(self):
        """Run the user-provided training script."""
        # Get user script and module name
        user_script = self.training_env.user_entry_point
        module_name = self.training_env.module_name
        
        logger.info(f"Running user script: {user_script}")
        logger.info(f"Module name: {module_name}")
        
        # Prepare command line arguments
        cmd_args = self.training_env.to_cmd_args()
        env_vars = self.training_env.to_env_vars()
        
        # Execute the user script
        entry_point.run(
            self.training_env.module_dir,
            user_script,
            cmd_args,
            env_vars,
            runner_type=runner.ProcessRunnerType
        )

def main():
    """Main entry point for training."""
    logger.info("Starting unified training framework...")
    
    try:
        # Create and run training framework
        framework = UnifiedTrainingFramework()
        framework.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()