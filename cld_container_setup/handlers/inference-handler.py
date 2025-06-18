"""
Unified inference handler for SageMaker endpoints.
Supports both single-model and multi-model endpoints with pluggable backends.
"""
import os
import json
import logging
import pickle
import glob
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModelHandler(ABC):
    """Base class for model handlers."""
    
    def __init__(self):
        self.initialized = False
        self.model = None
        self.model_artifacts = {}
        
    @abstractmethod
    def initialize(self, context):
        """Initialize the model handler."""
        pass
    
    @abstractmethod
    def preprocess(self, request):
        """Preprocess the input data."""
        pass
    
    @abstractmethod
    def inference(self, model_input):
        """Run inference on the model."""
        pass
    
    @abstractmethod
    def postprocess(self, inference_output):
        """Postprocess the inference output."""
        pass

class DefaultModelHandler(BaseModelHandler):
    """Default model handler implementation."""
    
    def initialize(self, context):
        """Initialize the model handler."""
        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        logger.info(f"Initializing model from directory: {model_dir}")
        
        # Load model based on available files
        self._load_model(model_dir)
        
    def _load_model(self, model_dir):
        """Load model from directory based on file type."""
        # Check for different model formats
        model_files = list(Path(model_dir).glob("*"))
        
        # Try loading PyTorch model
        pt_files = list(Path(model_dir).glob("*.pt")) + list(Path(model_dir).glob("*.pth"))
        if pt_files:
            self._load_pytorch_model(pt_files[0])
            return
            
        # Try loading TensorFlow/Keras model
        if (Path(model_dir) / "saved_model.pb").exists():
            self._load_tensorflow_model(model_dir)
            return
            
        # Try loading scikit-learn model
        pkl_files = list(Path(model_dir).glob("*.pkl")) + list(Path(model_dir).glob("*.joblib"))
        if pkl_files:
            self._load_sklearn_model(pkl_files[0])
            return
            
        # Try loading MXNet model
        symbol_files = list(Path(model_dir).glob("*-symbol.json"))
        if symbol_files:
            self._load_mxnet_model(model_dir)
            return
            
        # Load custom model handler if specified
        custom_handler = os.environ.get('SAGEMAKER_MODEL_HANDLER')
        if custom_handler:
            self._load_custom_handler(custom_handler, model_dir)
            return
            
        logger.warning("No recognized model format found. Implement custom handler.")
    
    def _load_pytorch_model(self, model_path):
        """Load PyTorch model."""
        try:
            import torch
            logger.info(f"Loading PyTorch model from {model_path}")
            self.model = torch.load(model_path, map_location='cpu')
            if hasattr(self.model, 'eval'):
                self.model.eval()
            self.model_type = 'pytorch'
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def _load_tensorflow_model(self, model_dir):
        """Load TensorFlow model."""
        try:
            import tensorflow as tf
            logger.info(f"Loading TensorFlow model from {model_dir}")
            self.model = tf.saved_model.load(model_dir)
            self.model_type = 'tensorflow'
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            raise
    
    def _load_sklearn_model(self, model_path):
        """Load scikit-learn model."""
        try:
            import joblib
            logger.info(f"Loading scikit-learn model from {model_path}")
            self.model = joblib.load(model_path)
            self.model_type = 'sklearn'
        except Exception as e:
            logger.error(f"Failed to load scikit-learn model: {e}")
            raise
    
    def _load_mxnet_model(self, model_dir):
        """Load MXNet model."""
        try:
            import mxnet as mx
            logger.info(f"Loading MXNet model from {model_dir}")
            
            # Find model prefix
            symbol_file = list(Path(model_dir).glob("*-symbol.json"))[0]
            prefix = str(symbol_file).replace('-symbol.json', '')
            
            # Load model
            sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
            ctx = mx.cpu()
            self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            
            # Load shapes if available
            shapes_file = f"{prefix}-shapes.json"
            if os.path.exists(shapes_file):
                with open(shapes_file) as f:
                    shapes = json.load(f)
                    data_shapes = [(s['name'], tuple(s['shape'])) for s in shapes]
                    self.model.bind(for_training=False, data_shapes=data_shapes)
                    self.model.set_params(arg_params, aux_params, allow_missing=True)
            
            self.model_type = 'mxnet'
        except Exception as e:
            logger.error(f"Failed to load MXNet model: {e}")
            raise
    
    def _load_custom_handler(self, handler_path, model_dir):
        """Load custom model handler."""
        spec = importlib.util.spec_from_file_location("custom_model", handler_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'load_model'):
            self.model = module.load_model(model_dir)
            self.model_type = 'custom'
    
    def preprocess(self, request):
        """Preprocess the input data."""
        # Handle different input formats
        processed_data = []
        
        for data in request:
            if isinstance(data, dict):
                # Handle JSON input
                if 'instances' in data:
                    processed_data.extend(data['instances'])
                elif 'data' in data:
                    processed_data.append(data['data'])
                elif 'body' in data:
                    # Handle raw body data
                    body = data['body']
                    if isinstance(body, (bytes, bytearray)):
                        # Try to decode as JSON
                        try:
                            decoded = json.loads(body.decode('utf-8'))
                            processed_data.append(decoded)
                        except:
                            # Keep as raw bytes
                            processed_data.append(body)
                    else:
                        processed_data.append(body)
                else:
                    processed_data.append(data)
            else:
                processed_data.append(data)
        
        return processed_data
    
    def inference(self, model_input):
        """Run inference on the model."""
        if self.model_type == 'pytorch':
            return self._pytorch_inference(model_input)
        elif self.model_type == 'tensorflow':
            return self._tensorflow_inference(model_input)
        elif self.model_type == 'sklearn':
            return self._sklearn_inference(model_input)
        elif self.model_type == 'mxnet':
            return self._mxnet_inference(model_input)
        elif self.model_type == 'custom':
            return self._custom_inference(model_input)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _pytorch_inference(self, model_input):
        """PyTorch inference."""
        import torch
        with torch.no_grad():
            # Convert input to tensor if needed
            if not isinstance(model_input[0], torch.Tensor):
                model_input = [torch.tensor(x) for x in model_input]
            # Stack inputs into batch
            batch = torch.stack(model_input)
            output = self.model(batch)
            return output.numpy()
    
    def _tensorflow_inference(self, model_input):
        """TensorFlow inference."""
        import tensorflow as tf
        # Convert to tensor
        input_tensor = tf.constant(model_input)
        output = self.model(input_tensor)
        return output.numpy()
    
    def _sklearn_inference(self, model_input):
        """Scikit-learn inference."""
        # Ensure input is 2D array
        input_array = np.array(model_input)
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)
        return self.model.predict(input_array)
    
    def _mxnet_inference(self, model_input):
        """MXNet inference."""
        import mxnet as mx
        from collections import namedtuple
        Batch = namedtuple('Batch', ['data'])
        self.model.forward(Batch([mx.nd.array(model_input)]))
        return self.model.get_outputs()[0].asnumpy()
    
    def _custom_inference(self, model_input):
        """Custom inference logic."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(model_input)
        elif callable(self.model):
            return self.model(model_input)
        else:
            raise ValueError("Custom model must be callable or have predict method")
    
    def postprocess(self, inference_output):
        """Postprocess the inference output."""
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(inference_output, np.ndarray):
            return inference_output.tolist()
        elif hasattr(inference_output, 'numpy'):
            return inference_output.numpy().tolist()
        else:
            return inference_output

class UnifiedModelHandler:
    """Unified handler that manages model loading and inference."""
    
    def __init__(self):
        self.handler = None
        
    def initialize(self, context):
        """Initialize the handler."""
        # Check for custom handler class
        custom_handler_class = os.environ.get('SAGEMAKER_HANDLER_CLASS')
        
        if custom_handler_class:
            # Load custom handler class
            module_name, class_name = custom_handler_class.rsplit('.', 1)
            module = importlib.import_module(module_name)
            handler_class = getattr(module, class_name)
            self.handler = handler_class()
        else:
            # Use default handler
            self.handler = DefaultModelHandler()
        
        self.handler.initialize(context)
    
    def handle(self, data, context):
        """Handle inference request."""
        try:
            # Initialize if needed
            if not self.handler.initialized:
                self.initialize(context)
            
            # Preprocess
            model_input = self.handler.preprocess(data)
            
            # Inference
            model_output = self.handler.inference(model_input)
            
            # Postprocess
            response = self.handler.postprocess(model_output)
            
            return response
            
        except MemoryError as e:
            logger.error(f"Out of memory error: {e}")
            raise
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            raise

# Global handler instance
_handler = UnifiedModelHandler()

def handle(data, context):
    """Entry point for MMS inference requests."""
    if data is None:
        return None
    return _handler.handle(data, context)