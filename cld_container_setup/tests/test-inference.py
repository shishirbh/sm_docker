"""
Tests for the inference handler module.
"""
import os
import sys
import json
import tempfile
import shutil
import pickle
from unittest import TestCase, mock
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handlers.inference_handler import (
    BaseModelHandler, DefaultModelHandler, UnifiedModelHandler, handle
)


class MockContext:
    """Mock context for testing."""
    def __init__(self, model_dir='/tmp/model'):
        self.system_properties = {
            'model_dir': model_dir,
            'gpu_id': None
        }


class TestBaseModelHandler(TestCase):
    """Test cases for BaseModelHandler."""
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        handler = BaseModelHandler()
        context = MockContext()
        
        with self.assertRaises(NotImplementedError):
            handler.initialize(context)
        
        with self.assertRaises(NotImplementedError):
            handler.preprocess([])
        
        with self.assertRaises(NotImplementedError):
            handler.inference([])
        
        with self.assertRaises(NotImplementedError):
            handler.postprocess([])


class TestDefaultModelHandler(TestCase):
    """Test cases for DefaultModelHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, 'model')
        os.makedirs(self.model_dir)
        self.handler = DefaultModelHandler()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_initialize_sklearn_model(self):
        """Test loading scikit-learn model."""
        # Create a dummy sklearn model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        
        import joblib
        model_path = os.path.join(self.model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        
        context = MockContext(self.model_dir)
        self.handler.initialize(context)
        
        self.assertTrue(self.handler.initialized)
        self.assertEqual(self.handler.model_type, 'sklearn')
        self.assertIsInstance(self.handler.model, LogisticRegression)
    
    def test_initialize_pytorch_model(self):
        """Test loading PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            
            # Create a dummy PyTorch model
            model = nn.Linear(10, 2)
            model_path = os.path.join(self.model_dir, 'model.pt')
            torch.save(model, model_path)
            
            context = MockContext(self.model_dir)
            self.handler.initialize(context)
            
            self.assertTrue(self.handler.initialized)
            self.assertEqual(self.handler.model_type, 'pytorch')
        except ImportError:
            self.skipTest("PyTorch not installed")
    
    def test_preprocess_json_input(self):
        """Test preprocessing JSON input."""
        request = [
            {'instances': [[1, 2, 3], [4, 5, 6]]},
            {'data': [7, 8, 9]},
            {'body': b'{"values": [10, 11, 12]}'}
        ]
        
        processed = self.handler.preprocess(request)
        
        self.assertEqual(len(processed), 4)  # 2 + 1 + 1
        self.assertEqual(processed[0], [1, 2, 3])
        self.assertEqual(processed[1], [4, 5, 6])
        self.assertEqual(processed[2], [7, 8, 9])
        self.assertIn('values', processed[3])
    
    def test_sklearn_inference(self):
        """Test sklearn model inference."""
        # Create and save a trained model
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        
        model_path = os.path.join(self.model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        
        # Initialize handler
        context = MockContext(self.model_dir)
        self.handler.initialize(context)
        
        # Run inference
        test_input = X[:5]
        output = self.handler.inference(test_input)
        
        self.assertEqual(output.shape[0], 5)
        self.assertTrue(np.all(np.isin(output, [0, 1])))
    
    def test_postprocess(self):
        """Test postprocessing output."""
        # Test numpy array
        np_output = np.array([[1, 2, 3], [4, 5, 6]])
        processed = self.handler.postprocess(np_output)
        self.assertEqual(processed, [[1, 2, 3], [4, 5, 6]])
        
        # Test list
        list_output = [1, 2, 3]
        processed = self.handler.postprocess(list_output)
        self.assertEqual(processed, [1, 2, 3])


class TestUnifiedModelHandler(TestCase):
    """Test cases for UnifiedModelHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, 'model')
        os.makedirs(self.model_dir)
        self.handler = UnifiedModelHandler()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_initialize_default_handler(self):
        """Test initialization with default handler."""
        context = MockContext(self.model_dir)
        self.handler.initialize(context)
        
        self.assertIsInstance(self.handler.handler, DefaultModelHandler)
    
    @mock.patch.dict(os.environ, {'SAGEMAKER_HANDLER_CLASS': 'handlers.inference_handler.DefaultModelHandler'})
    def test_initialize_custom_handler_class(self):
        """Test initialization with custom handler class."""
        context = MockContext(self.model_dir)
        self.handler.initialize(context)
        
        self.assertIsInstance(self.handler.handler, DefaultModelHandler)
    
    def test_handle_request(self):
        """Test handling inference request."""
        # Create a simple model
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        model = LogisticRegression()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model.fit(X, y)
        
        model_path = os.path.join(self.model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        
        # Prepare request
        context = MockContext(self.model_dir)
        data = [{'instances': [[1, 2], [3, 4]]}]
        
        # Handle request
        response = self.handler.handle(data, context)
        
        self.assertIsNotNone(response)
        self.assertEqual(len(response), 2)
    
    def test_handle_memory_error(self):
        """Test handling memory error."""
        # Create custom handler that raises MemoryError
        class MemoryErrorHandler(BaseModelHandler):
            def initialize(self, context):
                raise MemoryError("Out of memory")
            
            def preprocess(self, request):
                pass
            
            def inference(self, model_input):
                pass
            
            def postprocess(self, output):
                pass
        
        self.handler.handler = MemoryErrorHandler()
        
        context = MockContext(self.model_dir)
        data = [{'data': [1, 2, 3]}]
        
        with self.assertRaises(MemoryError):
            self.handler.handle(data, context)


class TestHandleFunction(TestCase):
    """Test the global handle function."""
    
    def test_handle_none_data(self):
        """Test handle function with None data."""
        result = handle(None, MockContext())
        self.assertIsNone(result)
    
    @mock.patch('handlers.inference_handler._handler')
    def test_handle_delegates_to_handler(self, mock_handler):
        """Test handle function delegates to handler."""
        data = [{'test': 'data'}]
        context = MockContext()
        expected_response = {'predictions': [0, 1]}
        
        mock_handler.handle.return_value = expected_response
        
        response = handle(data, context)
        
        mock_handler.handle.assert_called_once_with(data, context)
        self.assertEqual(response, expected_response)


if __name__ == '__main__':
    import unittest
    unittest.main()