# Scikit-learn Tabular Container

This sample Docker image provides a lightweight environment for training and serving scikit-learn models on Amazon SageMaker.  The same image can be used for both training jobs and inference endpoints.

## Folder Structure

```
sklearn_container/
├── Dockerfile             # Container definition
├── entrypoint.sh          # Entrypoint handling train/serve
├── scripts/
│   ├── train.py           # Example training script
│   └── inference.py       # Example inference script
```

## Build

```bash
docker build -t sklearn-sagemaker sklearn_container
```

## Train Locally

```bash
docker run --rm \
  -e SM_MODE=train \
  -v $(pwd)/sklearn_container/scripts:/opt/ml/code \
  -v $(pwd)/data/train:/opt/ml/input/data/train \
  -v $(pwd)/model:/opt/ml/model \
  sklearn-sagemaker train.py
```

## Serve Locally

```bash
docker run --rm -p 8080:8080 \
  -e SM_MODE=serve \
  -v $(pwd)/sklearn_container/scripts:/opt/ml/code \
  -v $(pwd)/model:/opt/ml/model \
  sklearn-sagemaker inference.py
```

When used with SageMaker, specify the training or inference script as the `entry_point` argument of the estimator or model object.
