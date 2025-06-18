"""
Tests for the training module.
"""
import os
import sys
import json
import tempfile
import shutil
from unittest import TestCase, mock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_sagemaker_framework.training import UnifiedTrainingFramework


class TestUnifiedTrainingFramework(TestCase):
    """Test cases for UnifiedTrainingFramework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, 'model')
        self.code_dir = os.path.join(self.test_dir, 'code')
        self.input_dir = os.path.join(self.test_dir, 'input')
        
        # Create directories
        os.makedirs(self.model_dir)
        os.makedirs(self.code_dir)
        os.makedirs(self.input_dir)
        
        # Mock environment variables
        self.env_patcher = mock.patch.dict(os.environ, {
            'SM_MODEL_DIR': self.model_dir,
            'SM_MODULE_DIR': self.code_dir,
            'SM_INPUT_DIR': self.input_dir,
            'SM_OUTPUT_DATA_DIR': self.test_dir,
            'SM_CHANNEL_TRAIN': os.path.join(self.input_dir, 'train'),
            'SM_HP_EPOCHS': '10',
            'SM_HP_LEARNING_RATE': '0.001'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test framework initialization."""
        framework = UnifiedTrainingFramework()
        
        self.assertIsNotNone(framework.training_env)
        self.assertEqual(framework.training_env.model_dir, self.model_dir)
    
    def test_setup_paths(self):
        """Test path setup."""
        framework = UnifiedTrainingFramework()
        
        # Check model directory exists
        self.assertTrue(os.path.exists(self.model_dir))
        
        # Check code directory is in Python path
        self.assertIn(self.code_dir, sys.path)
    
    def test_load_custom_handler(self):
        """Test loading custom training handler."""
        # Create a mock handler file
        handler_path = os.path.join(self.test_dir, 'custom_handler.py')
        with open(handler_path, 'w') as f:
            f.write('''
def pre_training(env):
    return "pre_training_called"

def post_training(env):
    return "post_training_called"

def train(env):
    return "train_called"
''')
        
        os.environ['SAGEMAKER_TRAINING_HANDLER'] = handler_path
        
        framework = UnifiedTrainingFramework()
        handler = framework.load_custom_handler()
        
        self.assertIsNotNone(handler)
        self.assertTrue(hasattr(handler, 'pre_training'))
        self.assertTrue(hasattr(handler, 'post_training'))
        self.assertTrue(hasattr(handler, 'train'))
    
    def test_pre_training_hook(self):
        """Test pre-training hook execution."""
        framework = UnifiedTrainingFramework()
        
        # Should not raise exception
        framework.pre_training_hook()
    
    def test_post_training_hook(self):
        """Test post-training hook execution."""
        framework = UnifiedTrainingFramework()
        
        # Create a dummy model file
        model_file = os.path.join(self.model_dir, 'model.pkl')
        with open(model_file, 'w') as f:
            f.write('dummy model')
        
        # Should not raise exception
        framework.post_training_hook()
    
    @mock.patch('src.unified_sagemaker_framework.training.entry_point.run')
    def test_train_with_user_script(self, mock_run):
        """Test training with user script."""
        framework = UnifiedTrainingFramework()
        
        # Create dummy user script
        user_script = os.path.join(self.code_dir, 'train.py')
        with open(user_script, 'w') as f:
            f.write('print("training")')
        
        framework.training_env.user_entry_point = 'train.py'
        framework.training_env.module_name = 'train'
        
        framework.train()
        
        # Check entry_point.run was called
        mock_run.assert_called_once()
    
    def test_train_with_custom_handler(self):
        """Test training with custom handler."""
        # Create custom handler
        handler_path = os.path.join(self.test_dir, 'handler.py')
        with open(handler_path, 'w') as f:
            f.write('''
trained = False

def train(env):
    global trained
    trained = True
    
    # Save a dummy model
    import os
    model_path = os.path.join(env.model_dir, 'model.txt')
    with open(model_path, 'w') as f:
        f.write('trained model')
''')
        
        os.environ['SAGEMAKER_TRAINING_HANDLER'] = handler_path
        
        framework = UnifiedTrainingFramework()
        framework.train()
        
        # Check model was saved
        model_file = os.path.join(self.model_dir, 'model.txt')
        self.assertTrue(os.path.exists(model_file))
        
        with open(model_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, 'trained model')


class TestTrainingIntegration(TestCase):
    """Integration tests for training module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    @mock.patch.dict(os.environ, {
        'SM_MODEL_DIR': '/tmp/model',
        'SM_MODULE_DIR': '/tmp/code',
        'SM_INPUT_DIR': '/tmp/input'
    })
    def test_main_function(self):
        """Test main entry point."""
        from src.unified_sagemaker_framework.training import main
        
        # Mock the framework to avoid actual training
        with mock.patch('src.unified_sagemaker_framework.training.UnifiedTrainingFramework') as MockFramework:
            instance = MockFramework.return_value
            instance.train.return_value = None
            
            # Should not raise exception
            main()
            
            # Check framework was created and train was called
            MockFramework.assert_called_once()
            instance.train.assert_called_once()


if __name__ == '__main__':
    import unittest
    unittest.main()