# Example Custom Scripts for SageMaker

This directory provides example `train.py` and `inference.py` scripts for use with the custom SageMaker Docker image built from the `cld_container_setup` directory.

## Prerequisites

1.  **Docker Installed**: Ensure Docker is installed and running on your system.
2.  **AWS CLI Configured**: If you plan to push to ECR and use with SageMaker, ensure your AWS CLI is configured with necessary permissions for ECR, S3, and SageMaker.
3.  **Build the Docker Image**:
    *   Navigate to the `cld_container_setup` directory.
    *   Modify the `build_and_push.sh` script with your AWS Account ID, Region, and desired ECR Repository Name.
    *   Run the script: `bash build_and_push.sh <your-aws-account-id> <your-aws-region> <your-ecr-repo-name>`
    *   This will build the image and tag it as `<your-ecr-repo-name>:latest`. For local testing, you can use this local tag.

Let's assume `IMAGE_NAME="your-ecr-repo-name"` for the local testing commands below.

## Local Testing

Create a directory for local testing artifacts:
```bash
mkdir -p local_test/input/data/training
mkdir -p local_test/model_dir
mkdir -p local_test/output
```

### Local Training Test

1.  **Prepare Dummy Data (Optional)**:
    Create a dummy CSV file for training if you don't have your own:
    ```bash
    echo "feature1,feature2,feature3,feature4,feature5,target
1.0,2.0,1.5,2.5,1.8,0
3.0,4.0,3.5,4.5,3.8,1
2.0,3.0,2.5,3.5,2.8,0" > local_test/input/data/training/data.csv
    ```
    Our example `train.py` generates dummy data if `data.csv` is not found, but using a file is a good test.

2.  **Run Docker for Training**:
    Replace `IMAGE_NAME` with the actual name of your built Docker image.
    ```bash
    docker run --rm \
        -v $(pwd)/sagemaker_custom_scripts_example:/opt/ml/code \
        -v $(pwd)/local_test/input/data/training:/opt/ml/input/data/training \
        -v $(pwd)/local_test/model_dir:/opt/ml/model \
        -e SAGEMAKER_PROGRAM="train.py" \
        -e SM_MODEL_DIR="/opt/ml/model" \
        -e SM_CHANNEL_TRAINING="/opt/ml/input/data/training" \
        -e SM_HYPERPARAMETERS='{"random_state": 123, "n_estimators": 150}' \
        ${IMAGE_NAME} train
    ```
    After execution, your trained model (`model.joblib`) should be in `local_test/model_dir/`.

### Local Inference Test

1.  **Ensure Model Exists**: Make sure `model.joblib` is present in `local_test/model_dir/` from the training step.

2.  **Run Docker for Serving**:
    ```bash
    docker run --rm -d --name local_inference_server \
        -v $(pwd)/sagemaker_custom_scripts_example:/opt/ml/code \
        -v $(pwd)/local_test/model_dir:/opt/ml/model \
        -p 8080:8080 \
        -e SAGEMAKER_PROGRAM="inference.py" \
        -e SM_MODEL_DIR="/opt/ml/model" \
        -e SAGEMAKER_SUBMIT_DIRECTORY="/opt/ml/code" \
        -e SAGEMAKER_ENABLE_CLOUDWATCH_METRICS="false" \
        ${IMAGE_NAME} serve
    ```

3.  **Wait for Server and Send Request**:
    ```bash
    echo "Waiting for server to start..."
    sleep 15 # Adjust if necessary

    echo "Sending CSV inference request:"
    curl --request POST \
      --url http://localhost:8080/invocations \
      --header 'Content-Type: text/csv' \
      --data '0.5,0.3,0.4,0.2,0.1
0.1,0.9,0.8,0.7,0.6'
    echo "\n"

    echo "Sending JSON inference request (list of lists):"
    # Note: The dummy model in train.py expects 5 features.
    curl --request POST \
      --url http://localhost:8080/invocations \
      --header 'Content-Type: application/json' \
      --data '[[0.5,0.3,0.4,0.2,0.1],[0.1,0.9,0.8,0.7,0.6]]'
    echo "\n"

    # Example for JSON input with 'instances' key (if your input_fn is adapted)
    # Assuming model expects 5 features.
    # echo "Sending JSON inference request (dict with instances):"
    # curl --request POST \
    #  --url http://localhost:8080/invocations \
    #  --header 'Content-Type: application/json' \
    #  --data '{"instances": [[0.5,0.3,0.4,0.2,0.1],[0.1,0.9,0.8,0.7,0.6]]}'
    # echo "\n"

    ```

4.  **Stop the Server**:
    ```bash
    docker stop local_inference_server
    ```

## Using with SageMaker SDK

Once the image is pushed to ECR (e.g., `ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/REPO_NAME:latest`):

1.  **Upload Scripts to S3**:
    Package the contents of `sagemaker_custom_scripts_example` (train.py, inference.py, requirements.txt) into a `source.tar.gz` and upload to S3.
    ```bash
    tar -czvf source.tar.gz -C sagemaker_custom_scripts_example .
    aws s3 cp source.tar.gz s3://your-s3-bucket/path/to/source/
    ```
    Let `s3_source_dir = 's3://your-s3-bucket/path/to/source/source.tar.gz'`

2.  **SageMaker Training**:
    ```python
    from sagemaker.estimator import Estimator

    image_uri = 'YOUR_ECR_IMAGE_URI'
    role = 'YOUR_SAGEMAKER_EXECUTION_ROLE'
    s3_output_path = 's3://your-s3-bucket/path/to/training_output/'

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.m5.large', # Or any other instance type
        entry_point='train.py',       # Your training script
        source_dir=s3_source_dir,     # S3 URI to your source.tar.gz
        output_path=s3_output_path,
        hyperparameters={'random_state': 42, 'n_estimators': 200} # Example
    )

    # Define your training data input
    from sagemaker.inputs import TrainingInput
    s3_training_data = 's3://your-s3-bucket/path/to/training-data/' # CSV data
    inputs = {'training': TrainingInput(s3_uri=s3_training_data, content_type='text/csv')}

    estimator.fit(inputs)
    ```

3.  **SageMaker Inference**:
    ```python
    from sagemaker.model import Model

    image_uri = 'YOUR_ECR_IMAGE_URI'
    role = 'YOUR_SAGEMAKER_EXECUTION_ROLE'
    model_data = estimator.model_data # S3 URI to the model.tar.gz from training

    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        role=role,
        entry_point='inference.py',   # Your inference script
        source_dir=s3_source_dir      # S3 URI to your source.tar.gz
    )

    endpoint_name = 'my-custom-sklearn-endpoint'
    instance_type = 'ml.m5.large'

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )

    # Example prediction
    # payload = [[0.5,0.3,0.4,0.2,0.1],[0.1,0.9,0.8,0.7,0.6]] # for application/json
    payload_csv = "0.5,0.3,0.4,0.2,0.1\n0.1,0.9,0.8,0.7,0.6" # for text/csv

    # If using JSON:
    # from sagemaker.serializers import JSONSerializer
    # from sagemaker.deserializers import JSONDeserializer
    # predictor.serializer = JSONSerializer()
    # predictor.deserializer = JSONDeserializer()
    # result = predictor.predict(payload)

    # If using CSV:
    from sagemaker.serializers import CSVSerializer
    from sagemaker.deserializers import CSVDeserializer # Or JSONDeserializer if output is JSON
    predictor.serializer = CSVSerializer()
    # Assuming output is JSON, if it's CSV, adapt deserializer
    predictor.deserializer = JSONDeserializer(accept='application/json') # Our example inference.py outputs JSON

    result = predictor.predict(payload_csv)
    print(result)

    # Don't forget to delete the endpoint
    # predictor.delete_endpoint()
    ```
