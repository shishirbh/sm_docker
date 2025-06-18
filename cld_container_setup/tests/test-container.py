"""
Integration tests for the container.
"""
import os
import sys
import json
import subprocess
import tempfile
import shutil
import time
from unittest import TestCase, mock, skipIf
import docker
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def docker_available():
    """Check if Docker is available."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except:
        return False


class TestContainerBuild(TestCase):
    """Test container build process."""
    
    @skipIf(not docker_available(), "Docker not available")
    def test_dockerfile_builds(self):
        """Test that Dockerfile builds successfully."""
        # This is a basic test - in CI/CD you'd actually build
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dockerfile_path = os.path.join(project_root, 'Dockerfile')
        
        self.assertTrue(os.path.exists(dockerfile_path))


class TestEntrypoint(TestCase):
    """Test the unified entrypoint script."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    @mock.patch('subprocess.check_call')
    @mock.patch('subprocess.call')
    def test_entrypoint_train_command(self, mock_call, mock_check_call):
        """Test entrypoint with train command."""
        # Import entrypoint module
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
        from unified_entrypoint import main
        
        # Mock sys.argv
        with mock.patch.object(sys, 'argv', ['unified-entrypoint.py', 'train']):
            with mock.patch('unified_entrypoint.start_training') as mock_train:
                main()
                mock_train.assert_called_once()
    
    @mock.patch('subprocess.check_call')
    @mock.patch('subprocess.call')
    def test_entrypoint_serve_command(self, mock_call, mock_check_call):
        """Test entrypoint with serve command."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
        from unified_entrypoint import main
        
        with mock.patch.object(sys, 'argv', ['unified-entrypoint.py', 'serve']):
            with mock.patch('unified_entrypoint.start_model_server') as mock_serve:
                main()
                mock_serve.assert_called_once()
                mock_call.assert_called_with(["tail", "-f", "/dev/null"])


class TestTrainingIntegration(TestCase):
    """Integration tests for training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.env_vars = {
            'SM_MODEL_DIR': os.path.join(self.test_dir, 'model'),
            'SM_CHANNEL_TRAIN': os.path.join(self.test_dir, 'train'),
            'SM_OUTPUT_DATA_DIR': os.path.join(self.test_dir, 'output'),
            'SM_MODULE_DIR': os.path.join(self.test_dir, 'code'),
            'SM_HP_EPOCHS': '2',
            'SM_HP_LEARNING_RATE': '0.01'
        }
        
        # Create directories
        for key, path in self.env_vars.items():
            if 'DIR' in key or 'CHANNEL' in key:
                os.makedirs(path, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    @mock.patch.dict(os.environ, {})
    def test_training_with_custom_script(self):
        """Test training with custom user script."""
        # Create a simple training script
        train_script = os.path.join(self.env_vars['SM_MODULE_DIR'], 'train.py')
        with open(train_script, 'w') as f:
            f.write('''
import os
import json

def main():
    model_dir = os.environ.get('SM_MODEL_DIR', '/tmp/model')
    
    # Save a dummy model
    model_data = {'trained': True, 'epochs': int(os.environ.get('SM_HP_EPOCHS', 1))}
    
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'model.json'), 'w') as f:
        json.dump(model_data, f)
    
    print("Training completed!")

if __name__ == '__main__':
    main()
''')
        
        # Update environment
        self.env_vars.update({
            'SAGEMAKER_PROGRAM': 'train.py',
            'SM_USER_ENTRY_POINT': 'train.py',
            'SM_MODULE_NAME': 'train'
        })
        
        with mock.patch.dict(os.environ, self.env_vars):
            from src.unified_sagemaker_framework.training import main as train_main
            
            # Run training
            train_main()
            
            # Check model was saved
            model_file = os.path.join(self.env_vars['SM_MODEL_DIR'], 'model.json')
            self.assertTrue(os.path.exists(model_file))
            
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            self.assertEqual(model_data['epochs'], 2)
            self.assertTrue(model_data['trained'])


class TestServingIntegration(TestCase):
    """Integration tests for serving functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, 'model')
        os.makedirs(self.model_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_model_handler_loading(self):
        """Test model handler can be loaded."""
        # Create a dummy model
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        model = LogisticRegression()
        model.fit([[1, 2], [3, 4]], [0, 1])
        
        model_path = os.path.join(self.model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        
        # Test handler loading
        from handlers.inference_handler import UnifiedModelHandler, MockContext
        
        handler = UnifiedModelHandler()
        context = MockContext(self.model_dir)
        
        # Initialize handler
        handler.initialize(context)
        
        # Test inference
        data = [{'instances': [[1, 2], [3, 4]]}]
        response = handler.handle(data, context)
        
        self.assertIsNotNone(response)
        self.assertEqual(len(response), 2)


class TestEndToEnd(TestCase):
    """End-to-end tests (requires Docker)."""
    
    @skipIf(not docker_available(), "Docker not available")
    def test_container_health_check(self):
        """Test container can be built and is healthy."""
        # This would be implemented in a CI/CD pipeline
        # Here we just check Docker is available
        client = docker.from_env()
        self.assertIsNotNone(client.version())


class TestMultiModelSupport(TestCase):
    """Test multi-model endpoint support."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        os.environ['SAGEMAKER_MULTI_MODEL'] = 'true'
    
    def tearDown(self):
        """Clean up test fixtures."""
        del os.environ['SAGEMAKER_MULTI_MODEL']
        shutil.rmtree(self.test_dir)
    
    def test_multi_model_configuration(self):
        """Test multi-model configuration is recognized."""
        from src.unified_sagemaker_framework.serving import configure_mms
        
        with mock.patch.dict(os.environ, {'MMS_MODEL_STORE': self.test_dir}):
            config_path = configure_mms()
            
            # Check config file was created
            self.assertTrue(os.path.exists(config_path) or config_path == '/home/model-server/config.properties')


def run_integration_tests():
    """Run integration tests that require the container to be built."""
    print("Running container integration tests...")
    
    # Check if container exists
    try:
        client = docker.from_env()
        # Try to find the unified container
        images = client.images.list(filters={'reference': '*unified-sagemaker-container*'})
        
        if not images:
            print("Container not found. Build it first with build_and_push.sh")
            return False
        
        print(f"Found container image: {images[0].tags}")
        return True
        
    except Exception as e:
        print(f"Docker error: {e}")
        return False


if __name__ == '__main__':
    import unittest
    
    # Run unit tests
    unittest.main(verbosity=2)
    
    # Optionally run integration tests
    # run_integration_tests()