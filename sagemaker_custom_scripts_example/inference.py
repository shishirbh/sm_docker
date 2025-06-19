import os
import joblib
import json
import pandas as pd
import numpy as np

# SageMaker environment variables
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

def model_fn(model_dir):
    """
    Loads the saved model from the model directory.
    """
    print(f"Loading model from {model_dir}...")
    model_path = os.path.join(model_dir, 'model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model

def input_fn(request_body, request_content_type):
    """
    Deserializes the input request body into an object for prediction.
    """
    print(f"Received request_content_type: {request_content_type}")
    if request_content_type == 'application/json':
        try:
            data = json.loads(request_body)
            if isinstance(data, dict) and 'instances' in data: # Common format for TF Serving or AI Platform
                 features = data['instances']
            elif isinstance(data, list): # List of records
                 features = data
            else: # Assuming simple JSON dict or list of lists for features
                 features = data

            # Convert to DataFrame for scikit-learn
            # This part might need adjustment based on the expected input structure
            if isinstance(features, dict) and not any(isinstance(i, list) for i in features.values()):
                 # single instance as dict
                 features = pd.DataFrame([features])
            elif isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
                 # list of instances as dicts
                 features = pd.DataFrame(features)
            elif isinstance(features, list) and len(features) > 0 and isinstance(features[0], list):
                 # list of lists
                 features = pd.DataFrame(features)
            else:
                 raise ValueError("Unsupported JSON structure for input_fn")

            print(f"Deserialized JSON to DataFrame with shape: {features.shape}")
            return features
        except Exception as e:
            raise ValueError(f"Error deserializing JSON: {e}")

    elif request_content_type == 'text/csv':
        try:
            # Assuming CSV has no header and features are comma-separated
            from io import StringIO
            data = pd.read_csv(StringIO(request_body), header=None)
            print(f"Deserialized CSV to DataFrame with shape: {data.shape}")
            return data
        except Exception as e:
            raise ValueError(f"Error deserializing CSV: {e}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Makes predictions against the loaded model.
    """
    print(f"Making prediction on input data with shape: {input_data.shape}")
    try:
        predictions = model.predict(input_data)
        # If your model outputs probabilities, you might want model.predict_proba(input_data)
        # For this example, predict directly returns class labels or regression values
        print(f"Predictions: {predictions}")
        return predictions
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

def output_fn(prediction_output, response_content_type):
    """
    Serializes the prediction output to the desired response content type.
    """
    print(f"Serializing prediction to response_content_type: {response_content_type}")
    if response_content_type == 'application/json':
        try:
            # Assuming prediction_output is a numpy array or list
            response_body = json.dumps(prediction_output.tolist())
            return response_body
        except Exception as e:
            raise ValueError(f"Error serializing prediction to JSON: {e}")
    elif response_content_type == 'text/csv':
        try:
            # Assuming prediction_output is a 1D numpy array or list
            # For multi-column output, adjust pd.DataFrame creation
            if isinstance(prediction_output, np.ndarray):
                prediction_output = prediction_output.flatten()
            response_body = pd.Series(prediction_output).to_csv(index=False, header=False)
            return response_body
        except Exception as e:
            raise ValueError(f"Error serializing prediction to CSV: {e}")
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")

# Example for local testing (optional)
if __name__ == '__main__':
    # Create dummy model directory and model for testing
    print("Running inference script in __main__ for local testing (example)...")
    os.makedirs('/tmp/model', exist_ok=True)

    # Create a dummy model for testing model_fn
    dummy_model_data = LogisticRegression()
    # Fit with some dummy data so it's a valid model
    dummy_model_data.fit(np.array([[1,2],[3,4]]), np.array([0,1]))
    joblib.dump(dummy_model_data, '/tmp/model/model.joblib')

    # Test model_fn
    print("\nTesting model_fn...")
    model = model_fn('/tmp/model')

    # Test input_fn with JSON
    print("\nTesting input_fn with JSON...")
    json_input_dict = {"instances": [{"feature1": 0.5, "feature2": 0.3}, {"feature1": 0.1, "feature2": 0.9}]} # Example with 'instances' key
    json_input_list_of_dict = [{"feature1": 0.5, "feature2": 0.3}, {"feature1": 0.1, "feature2": 0.9}]
    json_input_list_of_list = [[0.5, 0.3], [0.1, 0.9]]

    # Assuming the dummy model was trained on 2 features
    df_input_from_dict = input_fn(json.dumps(json_input_dict), 'application/json')
    print(f"Input from dict: \n{df_input_from_dict}")

    df_input_from_list_dict = input_fn(json.dumps(json_input_list_of_dict), 'application/json')
    print(f"Input from list of dicts: \n{df_input_from_list_dict}")

    df_input_from_list_list = input_fn(json.dumps(json_input_list_of_list), 'application/json')
    print(f"Input from list of lists: \n{df_input_from_list_list}")

    # Test predict_fn
    print("\nTesting predict_fn...")
    predictions = predict_fn(df_input_from_list_list, model) # Using data that matches dummy model (2 features)
    print(f"Predictions: {predictions}")

    # Test output_fn with JSON
    print("\nTesting output_fn with JSON...")
    json_output = output_fn(predictions, 'application/json')
    print(f"JSON output: {json_output}")

    # Test input_fn with CSV
    print("\nTesting input_fn with CSV...")
    csv_input = "0.5,0.3\n0.1,0.9"
    df_input_csv = input_fn(csv_input, 'text/csv')
    print(f"Input from CSV: \n{df_input_csv}")

    # Test predict_fn with CSV data
    print("\nTesting predict_fn with CSV data...")
    predictions_csv = predict_fn(df_input_csv, model)
    print(f"Predictions from CSV data: {predictions_csv}")

    # Test output_fn with CSV
    print("\nTesting output_fn with CSV...")
    csv_output = output_fn(predictions_csv, 'text/csv')
    print(f"CSV output: {csv_output}")

    # Clean up dummy model
    os.remove('/tmp/model/model.joblib')
    os.rmdir('/tmp/model')
    print("\nLocal testing example finished.")
