#!/usr/bin/env python
"""
Example training script that demonstrates how to structure your training code
for the unified SageMaker container.
"""
import argparse
import os
import json
import logging
import sys
from pathlib import Path

# Example ML framework imports (customize as needed)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Example training script')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model-type', type=str, default='sklearn',
                       choices=['sklearn', 'pytorch', 'tensorflow', 'custom'],
                       help='Type of model to train')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    # Parse arguments
    args, _ = parser.parse_known_args()
    
    return args

def load_data(data_path):
    """Load training data from the specified path."""
    logger.info(f"Loading data from: {data_path}")
    
    # Example: Load CSV files
    csv_files = list(Path(data_path).glob('*.csv'))
    if csv_files:
        # Load first CSV file found
        df = pd.read_csv(csv_files[0])
        logger.info(f"Loaded data shape: {df.shape}")
        return df
    
    # Example: Load numpy files
    npy_files = list(Path(data_path).glob('*.npy'))
    if npy_files:
        data = np.load(npy_files[0])
        logger.info(f"Loaded numpy array shape: {data.shape}")
        return data
    
    # Add more data loading logic as needed
    raise ValueError(f"No supported data files found in {data_path}")

def preprocess_data(df):
    """Preprocess the data."""
    logger.info("Preprocessing data...")
    
    # Example preprocessing
    # Assume last column is target
    if isinstance(df, pd.DataFrame):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        X = df[:, :-1]
        y = df[:, -1]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_sklearn_model(X_train, y_train, X_val, y_val, args):
    """Train a scikit-learn model."""
    from sklearn.ensemble import RandomForestClassifier
    
    logger.info("Training RandomForest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Validation accuracy: {val_score:.4f}")
    
    return model, {'train_acc': train_score, 'val_acc': val_score}

def train_pytorch_model(X_train, y_train, X_val, y_val, args):
    """Train a PyTorch model."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    logger.info("Training PyTorch model...")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Define model
    class SimpleNet(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Initialize model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = SimpleNet(input_dim, num_classes)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return model, {'epochs': args.epochs, 'final_val_acc': val_acc}

def save_model(model, scaler, metrics, args):
    """Save the trained model and artifacts."""
    logger.info(f"Saving model to: {args.model_dir}")
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    if args.model_type == 'sklearn':
        import joblib
        # Save model
        joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
        # Save scaler
        joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.joblib'))
        
    elif args.model_type == 'pytorch':
        import torch
        # Save model
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
        # Save model architecture info
        model_info = {
            'input_dim': model.fc1.in_features,
            'num_classes': model.fc3.out_features
        }
        with open(os.path.join(args.model_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f)
    
    # Save metrics
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save training configuration
    config = vars(args)
    with open(os.path.join(args.model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Model saved successfully!")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    logger.info("Starting training with arguments:")
    logger.info(json.dumps(vars(args), indent=2))
    
    try:
        # Load data
        train_data = load_data(args.train)
        
        # Handle validation data
        if os.path.exists(args.validation) and os.listdir(args.validation):
            val_data = load_data(args.validation)
            X_train, y_train, scaler_train = preprocess_data(train_data)
            X_val, y_val, _ = preprocess_data(val_data)
            # Apply same scaling to validation
            X_val = scaler_train.transform(X_val[:, :-1])
        else:
            # Split training data
            if isinstance(train_data, pd.DataFrame):
                train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)
                X_train, y_train, scaler_train = preprocess_data(train_df)
                X_val, y_val, _ = preprocess_data(val_df)
                X_val = scaler_train.transform(X_val[:, :-1])
            else:
                X, y, scaler_train = preprocess_data(train_data)
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model based on type
        if args.model_type == 'sklearn':
            model, metrics = train_sklearn_model(X_train, y_train, X_val, y_val, args)
        elif args.model_type == 'pytorch':
            model, metrics = train_pytorch_model(X_train, y_train, X_val, y_val, args)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        # Save model
        save_model(model, scaler_train, metrics, args)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()