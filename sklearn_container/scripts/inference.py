import os
import io
import pandas as pd
import joblib

model = None

def model_fn(model_dir):
    global model
    if model is None:
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        return pd.read_csv(io.StringIO(request_body), header=None)
    raise ValueError('Unsupported content type: ' + request_content_type)

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    if accept == 'text/csv':
        result = ','.join(str(x) for x in prediction)
        return result, 'text/csv'
    raise ValueError('Unsupported accept type: ' + accept)
