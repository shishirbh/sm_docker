import json
import logging
import os

import numpy as np
import joblib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir):
    """Load the model for inference."""
    model_path = os.path.join(model_dir, "model.joblib")
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    """Parse input data payload."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Perform prediction."""
    prediction = model.predict(input_data)
    return prediction


def output_fn(prediction, accept):
    """Format the prediction output."""
    if accept == "application/json":
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported accept type: {accept}")
