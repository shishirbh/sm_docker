import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    args = parser.parse_args()

    data_path = os.path.join(args.train, 'train.csv')
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
