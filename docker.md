# Dockerization — Fraud Detection API

## Architecture

The application uses the following architecture:

Client → FastAPI → MLflow → Trained XGBoost Model → Prediction

The FastAPI application is located in:


src/app.py


The model was trained and tracked using MLflow from the analysis/training notebook.

## Docker Image

The Dockerfile packages the FastAPI serving application and its Python dependencies into a Docker image.

The image includes:

* Python 3.10
* FastAPI
* Pydantic
* MLflow
* pandas
* Uvicorn
* FastAPI application code

The training notebook and training dataset are not included because they are not required for serving predictions.

The trained model is accessed through MLflow rather than being stored directly inside the application directory.

## Build Command

On a system with Docker installed, the image would be created with:

bash
docker build -t fraud-detection-api .


## Run Command

The container would be started with:

bash
docker run -p 8000:8000 fraud-detection-api


The FastAPI application would then expose its API on port 8000.

## MLflow Configuration

The application currently uses an MLflow tracking server:

python
mlflow.set_tracking_uri("http://127.0.0.1:5000")


and loads the tracked XGBoost model using its MLflow run URI.

In a production containerized deployment, the MLflow server address would need to be configured so that it is reachable from the FastAPI container.

## Local Environment Limitation

Docker image building and container execution were not performed on the development machine because the current system does not support the required Docker environment.

Podman was also evaluated as an alternative but could not be successfully configured.

Therefore, this project documents the Docker containerization configuration and commands without claiming that the image was locally built or executed.

## Intended Deployment Flow


Client
   ↓
FastAPI Docker Container
   ↓
MLflow Server
   ↓
Trained XGBoost Model
   ↓
Fraud Probability
   ↓
Risk Classification

