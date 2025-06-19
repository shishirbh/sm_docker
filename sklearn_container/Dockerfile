FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl vim && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        scikit-learn pandas numpy joblib \
        sagemaker-training sagemaker-inference

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["train.py"]
