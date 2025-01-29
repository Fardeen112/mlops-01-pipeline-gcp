PROJECT_ID = "involuted-tuner-441406-a9"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"} # @param {type:"string"}

BUCKET_URI = f"gs://mlops-01-pipeline/pipeline_root_1"  # @param {type:"string"}

from typing import NamedTuple

import kfp
from google.cloud import aiplatform
from kfp import compiler, dsl
from kfp.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                     OutputPath, component)

PIPELINE_ROOT = "{}/pipeline_root/shakespeare".format(BUCKET_URI)

aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)

@component(
    base_image="python:3.9",
    packages_to_install=[
        "scikit-learn",
        "pandas",
        "numpy",
        "google-cloud-storage"
    ]
)
def train(
    imported_dataset: Input[Dataset],
) -> NamedTuple(
    "Outputs",
    [
        ("output_message", str),
        ("model_accuracy", float),
    ],
):
    """Training step using scikit-learn logistic regression.
    Splits data into train/test sets, trains on training data,
    saves model and test indices to GCS.
    """
    import logging
    import os
    from google.cloud import storage
    import tempfile
    import numpy as np
    import json
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        import pandas as pd
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        logger.info("Successfully imported all required packages")
        
        # Load dataset
        logger.info(f"Loading dataset from: {imported_dataset.path}")
        df = pd.read_csv(imported_dataset.path)
        logger.info(f"Dataset shape: {df.shape}")
        
        # Split features and target
        X = df.drop(columns=["Diabetic", "PatientID"])
        y = df["Diabetic"]
        
        # Split the data and save test indices
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Get test indices for later use
        test_indices = X_test.index.tolist()
        
        logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Train logistic regression model
        lr_model = LogisticRegression(max_iter=1000)
        logger.info("Training logistic regression model...")
        lr_model.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy on training data
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model training completed. Validation accuracy: {accuracy:.4f}")
        
        # Save model and test indices to GCS
        bucket_name = "mlops-01-pipeline"
        model_folder = "models"
        model_filename = "model.joblib"
        indices_filename = "test_indices.json"
        
        gcs_model_path = f"{model_folder}/{model_filename}"
        gcs_indices_path = f"{model_folder}/{indices_filename}"
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            temp_model_path = os.path.join(temp_dir, model_filename)
            logger.info(f"Saving model to temporary file: {temp_model_path}")
            joblib.dump(lr_model, temp_model_path)
            
            # Save test indices
            temp_indices_path = os.path.join(temp_dir, indices_filename)
            with open(temp_indices_path, 'w') as f:
                json.dump({'test_indices': test_indices}, f)
            
            # Initialize GCS client
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            # Upload model to GCS
            model_blob = bucket.blob(gcs_model_path)
            logger.info(f"Uploading model to GCS: gs://{bucket_name}/{gcs_model_path}")
            model_blob.upload_from_filename(temp_model_path)
            
            # Upload test indices to GCS
            indices_blob = bucket.blob(gcs_indices_path)
            logger.info(f"Uploading test indices to GCS: gs://{bucket_name}/{gcs_indices_path}")
            indices_blob.upload_from_filename(temp_indices_path)
            
            logger.info("Model and test indices successfully uploaded to GCS")
        
        output_message = (
            f"Model trained successfully with validation accuracy: {accuracy:.4f}. "
            f"Model saved to: gs://{bucket_name}/{gcs_model_path}"
        )
        
        return (output_message, float(accuracy))
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise RuntimeError(f"Training failed: {str(e)}")
        
        
@component(
    base_image="python:3.9",
    packages_to_install=[
        "scikit-learn",
        "pandas", 
        "numpy",
        "google-cloud-storage"
    ]
)
def test(
    imported_dataset: Input[Dataset],
) -> NamedTuple(
    "Outputs",
    [
        ("metrics_gcs_path", str),
    ],
):
    """Test step for evaluating the trained model.
    Uses the same test split as training by loading saved test indices.
    """
    import logging
    import os
    import json
    from google.cloud import storage
    import tempfile
    from datetime import datetime
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        import pandas as pd
        import joblib
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        logger.info("Successfully imported all required packages")
        
        # Constants
        bucket_name = "mlops-01-pipeline"
        model_folder = "models"
        model_filename = "model.joblib"
        metrics_folder="metrics"
        indices_filename = "test_indices.json"
        
        gcs_model_path = f"{model_folder}/{model_filename}"
        gcs_indices_path = f"{model_folder}/{indices_filename}"
        
        # Load full dataset
        logger.info(f"Loading dataset from: {imported_dataset.path}")
        df = pd.read_csv(imported_dataset.path)
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download model and test indices from GCS
            temp_model_path = os.path.join(temp_dir, model_filename)
            temp_indices_path = os.path.join(temp_dir, indices_filename)
            
            # Download model
            model_blob = bucket.blob(gcs_model_path)
            logger.info(f"Downloading model from GCS: gs://{bucket_name}/{gcs_model_path}")
            model_blob.download_to_filename(temp_model_path)
            
            # Download test indices
            indices_blob = bucket.blob(gcs_indices_path)
            logger.info(f"Downloading test indices from GCS: gs://{bucket_name}/{gcs_indices_path}")
            indices_blob.download_to_filename(temp_indices_path)
            
            # Load model and test indices
            model = joblib.load(temp_model_path)
            with open(temp_indices_path, 'r') as f:
                test_indices = json.load(f)['test_indices']
            
            # Prepare test features and target using saved indices
            test_df = df.iloc[test_indices]
            X_test = test_df.drop(columns=["Diabetic", "PatientID"])
            y_test = test_df["Diabetic"]
            
            logger.info(f"Test set size: {len(test_indices)}")
            
            # Make predictions
            logger.info("Making predictions on test data")
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                "accuracy": str(accuracy_score(y_test, y_pred)),
                "precision": str(precision_score(y_test, y_pred)),
                "recall": str(recall_score(y_test, y_pred)),
                "f1_score": str(f1_score(y_test, y_pred)),
                "timestamp": str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create metrics filename using same base name as model
            # metrics_filename = model_filename.replace('.joblib', f'_metrics_{timestamp}.json')

            # Define the path where you want to store the metrics (e.g., in the 'metrics' folder)
            # metrics_folder = "metrics"
            # os.makedirs(metrics_folder, exist_ok=True)  # Create the 'metrics' folder if it doesn't exist

            # GCS Path where you will store the metrics
            # metrics_gcs_path = f"{metrics_folder}/{metrics_filename}"
            
            metrics_filename = model_filename.replace('.joblib', f'{timestamp}_metrics.json')
            metrics_gcs_path = f"{metrics_folder}/{metrics_filename}"
            
            # Save metrics to temporary JSON file
            temp_metrics_path = os.path.join(temp_dir, metrics_filename)
            with open(temp_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Upload metrics to GCS
            metrics_blob = bucket.blob(metrics_gcs_path)
            logger.info(f"Uploading metrics to GCS: gs://{bucket_name}/{metrics_gcs_path}")
            metrics_blob.upload_from_filename(temp_metrics_path)
            logger.info("Metrics successfully uploaded to GCS")
        
        return (f"gs://{bucket_name}/{metrics_gcs_path}",)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise RuntimeError(f"Testing failed: {str(e)}")

        
@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform",
    ]
)
def deploy_model(
    project: str = "your-project-id",  # Replace with your project ID
    location: str = "us-central1",     # Replace with your desired location
    description: str= "metrics path",
    bucket_name: str = "mlops-01-pipeline",
    model_filename: str = "model.joblib",
) -> NamedTuple(
    "Outputs",
    [
        ("endpoint_name", str),
        ("model_name", str),
    ],
):
    """
    Uploads a trained model to Vertex AI and deploys it to an endpoint.
    
    Args:
        project: GCP project ID
        location: GCP region
        bucket_name: GCS bucket name where model is stored
        model_filename: Name of the model file in GCS
    
    Returns:
        NamedTuple containing endpoint and model names
    """
    import logging
    from google.cloud import aiplatform
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize Vertex AI
        aiplatform.init(
            project=project,
            location=location,
        )
        
        # Construct GCS URI for the model
        gcs_model_uri = f"gs://{bucket_name}/models"
        logger.info(f"Model URI: {gcs_model_uri}")
        
        # Generate unique names for model and endpoint
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_display_name = f"diabetes_prediction_{timestamp}"
        endpoint_display_name = f"diabetes_endpoint_{timestamp}"
        
        logger.info(f"Uploading model: {model_display_name}")
        # Upload model to Vertex AI
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=gcs_model_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest",
            description=description
        )
        logger.info(f"Model uploaded successfully: {model.resource_name}")
        
        logger.info(f"Deploying model to endpoint: {endpoint_display_name}")
        # Deploy model to endpoint
        endpoint = model.deploy(
            deployed_model_display_name=endpoint_display_name,
            machine_type="n1-standard-2",
        )
        logger.info(f"Model deployed successfully to endpoint: {endpoint.resource_name}")
        
        return (endpoint.resource_name, model.resource_name)
        
    except Exception as e:
        logger.error(f"An error occurred during model deployment: {str(e)}")
        raise RuntimeError(f"Model deployment failed: {str(e)}")
        
@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="metadata-pipeline-v2",
)
def pipeline(message: str):
    importer = kfp.dsl.importer(
        artifact_uri="gs://mlops-01-pipeline/Dataset_diabetes-dev.csv",
        artifact_class=Dataset,
        reimport=False,
    )
    # preprocess_task = preprocess(message=message)
    train_task = train(
        imported_dataset=importer.output)
    
    test_task=test(imported_dataset=importer.output).after(train_task)
    # Deploy model
    deploy_task = deploy_model(
        project="involuted-tuner-441406-a9",  # Replace with your project ID
        location="us-central1",# Replace with your desired location
        description=test_task.outputs["metrics_gcs_path"]
    ).after(test_task)  # Ensure deployment happens after training

compiler.Compiler().compile(
    pipeline_func=pipeline, package_path="lightweight_pipeline.yaml"
)
DISPLAY_NAME = "shakespeare"

job = aiplatform.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="lightweight_pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={"message": "Hello, World"},
    enable_caching=False,
)


if __name__ == "__main__":
    job.run()