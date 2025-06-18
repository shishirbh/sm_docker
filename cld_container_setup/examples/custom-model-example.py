"""
Example custom model implementation showing how to create
a model that works with the unified container.
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union

class CustomModel:
    """Example custom model class."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the custom model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.preprocessor = None
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train the custom model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        # Example: Simple custom implementation
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Initialize preprocessor
        self.preprocessor = StandardScaler()
        X_scaled = self.preprocessor.fit_transform(X)
        
        # Initialize and train model
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=self.config.get('max_depth', 3),
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate metrics
        train_score = self.model.score(X_scaled, y)
        
        return {
            'train_accuracy': train_score,
            'n_estimators': self.model.n_estimators,
            'n_features': X.shape[1]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Probability array
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, model_dir: str) -> None:
        """Save model to directory.
        
        Args:
            model_dir: Directory to save model artifacts
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'custom_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'preprocessor': self.preprocessor,
                'config': self.config,
                'is_trained': self.is_trained
            }, f)
        
        # Save metadata
        metadata = {
            'model_type': 'custom_gradient_boosting',
            'version': '1.0.0',
            'config': self.config,
            'features': self.model.n_features_in_ if self.model else None
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, model_dir: str) -> 'CustomModel':
        """Load model from directory.
        
        Args:
            model_dir: Directory containing model artifacts
            
        Returns:
            Loaded CustomModel instance
        """
        model_path = os.path.join(model_dir, 'custom_model.pkl')
        
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Create new instance and restore state
        instance = cls(config=saved_data['config'])
        instance.model = saved_data['model']
        instance.preprocessor = saved_data['preprocessor']
        instance.is_trained = saved_data['is_trained']
        
        return instance


class CustomNeuralNetwork:
    """Example custom neural network using PyTorch."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """Initialize custom neural network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the neural network architecture."""
        import torch
        import torch.nn as nn
        
        layers = []
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
    
    def save_state(self, filepath: str):
        """Save model state."""
        import torch
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim
        }, filepath)
    
    def load_state(self, filepath: str):
        """Load model state."""
        import torch
        checkpoint = torch.load(filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])


# Example usage functions
def create_and_train_custom_model():
    """Example of creating and training a custom model."""
    # Generate sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15)
    
    # Create and train model
    config = {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 4
    }
    
    model = CustomModel(config)
    metrics = model.train(X, y)
    print(f"Training metrics: {metrics}")
    
    # Make predictions
    predictions = model.predict(X[:10])
    probabilities = model.predict_proba(X[:10])
    
    print(f"Sample predictions: {predictions}")
    print(f"Sample probabilities shape: {probabilities.shape}")
    
    return model


def integrate_with_sagemaker_handler():
    """Example of integrating custom model with SageMaker handler."""
    
    class CustomModelHandler:
        """Handler for custom model in SageMaker."""
        
        def __init__(self):
            self.model = None
        
        def load_model(self, model_dir: str):
            """Load the custom model."""
            self.model = CustomModel.load(model_dir)
            return self.model
        
        def predict(self, input_data: Union[np.ndarray, List]):
            """Make predictions using the custom model."""
            if isinstance(input_data, list):
                input_data = np.array(input_data)
            
            return self.model.predict_proba(input_data)
    
    return CustomModelHandler


if __name__ == "__main__":
    # Example usage
    print("Creating and training custom model...")
    model = create_and_train_custom_model()
    
    # Save model
    model.save('./temp_model')
    print("Model saved!")
    
    # Load model
    loaded_model = CustomModel.load('./temp_model')
    print("Model loaded!")
    
    # Clean up
    import shutil
    shutil.rmtree('./temp_model', ignore_errors=True)