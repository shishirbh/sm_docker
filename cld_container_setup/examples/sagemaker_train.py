import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# SageMaker environment variables
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
SM_HYPERPARAMETERS = os.environ.get('SM_HYPERPARAMETERS', '{}')

def train():
    print("Starting training script...")

    # Load hyperparameters
    try:
        hyperparameters = eval(SM_HYPERPARAMETERS)
        print(f"Received hyperparameters: {hyperparameters}")
    except Exception as e:
        print(f"Could not evaluate hyperparameters: {e}. Using defaults.")
        hyperparameters = {}

    # Example hyperparameter
    n_estimators = hyperparameters.get('n_estimators', 100) # Just an example, LogisticRegression doesn't use it directly
    random_state = hyperparameters.get('random_state', 42)

    # --- Data Loading ---
    # Example: Load a CSV file. Replace with your actual data loading logic.
    # For this example, we'll create dummy data if no file is present.
    input_data_path = os.path.join(SM_CHANNEL_TRAINING, 'data.csv')

    if os.path.exists(input_data_path):
        print(f"Loading data from {input_data_path}")
        data = pd.read_csv(input_data_path)
        # Assuming last column is target and rest are features
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    else:
        print("No data.csv found in training channel. Generating dummy data.")
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100), name='target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")

    # --- Model Training ---
    print(f"Training LogisticRegression model... Random state: {random_state}")
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- Model Evaluation (Optional) ---
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.4f}")

    # --- Model Saving ---
    model_save_path = os.path.join(SM_MODEL_DIR, 'model.joblib')
    print(f"Saving model to {model_save_path}")
    joblib.dump(model, model_save_path)
    print("Model saved successfully.")

if __name__ == '__main__':
    # For local testing, you might want to parse arguments similarly to how SageMaker does
    # For SageMaker, it typically runs the script directly if it's the entry_point.

    # Example of how to parse custom arguments if needed
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-arg', type=str, default='default_value', help='A custom argument.')
    # Add any other arguments your script might need

    args, unknown = parser.parse_known_args() # Use parse_known_args to ignore SageMaker injected args if not defined
    print(f"Received custom args: {args}")

    train()
