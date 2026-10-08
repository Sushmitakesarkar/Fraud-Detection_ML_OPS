# Fraud Detection MLOps Platform

An end-to-end machine learning and MLOps project for detecting fraudulent credit card transactions.

The project demonstrates the complete workflow from model development and experiment tracking to API serving, monitoring, containerization, and CI/CD.

## Project Overview

The system:

Trains fraud detection models on an imbalanced credit card transaction dataset.

Evaluates models using fraud-focused metrics.

Tracks experiments and models using MLflow.

Serves the trained XGBoost model through a FastAPI REST API.

Implements basic application monitoring.

Provides Docker configuration for containerized deployment.

Uses GitHub Actions for automated testing, linting, and Docker image builds.

Documents a potential AWS production architecture.

# Architecture

                    Machine Learning
                         │
                         ▼
                 Model Training
                         │
                         ▼
                      MLflow
                         │
                         ▼
                   XGBoost Model
                         │
                         ▼
                    FastAPI API
                         │
              ┌──────────┼──────────┐
              │          │          │
           /health    /predict   /metrics
                         │
                         ▼
                    Monitoring
                         │
                         ▼
                       Docker
                         │
                         ▼
                    GitHub Actions
                         │
                 ┌───────┴────────┐
                 │                │
               Tests            Ruff
                 │                │
                 └───────┬────────┘
                         │
                    Docker Build

# Machine Learning
Used the Credit Card Fraud Detection dataset to build initial fraud detection models.

# Models
- Logistic Regression
- Random Forest
- XGBoost

### Key observation

The dataset is highly imbalanced, with fraudulent transactions representing only a very small fraction of all transactions.

Therefore, accuracy alone is not sufficient for evaluating a fraud detection system.


Fraud Model Evaluation

Because this dataset is highly imbalanced, accuracy alone is not a suitable metric for evaluating fraud detection performance.

Evaluation Metrics

We evaluated the models using:

Precision
Recall
F1 score
PR-AUC

Precision measures how many transactions predicted as fraud were actually fraudulent.(focus false positives)

Recall measures how many of the actual fraudulent transactions were successfully detected.(Focus false negatives)

F1 score combines precision and recall into a single metric.

PR-AUC evaluates the precision-recall trade-off across different classification thresholds and is particularly useful for highly imbalanced classification problems.


### WHY MIGHT A BANK PREFER RECALL OVER ACCURACY ?-
Recall focuses on all frauds that went undetected so for a bank false negatives means significant financial and customer consequences . For an imbalnced dataset accuracy could be great if he models flags all transactons legitimate but a recall would show how many frauds were missed .

However, maximizing recall can also increase false positives, causing legitimate transactions to be flagged. 
Therefore, a real fraud-detection system would need to consider recall together with precision, PR-AUC, threshold selection, and the costs of false positives and false negatives.

 Model Evaluation:
          Model           accuracy   precision    recall        f1        PR_AUC
0  Logistic Regression    0.973860   0.057288    0.918367    0.107849    0.717527
1        Decision Tree    0.999491   0.905882    0.785714    0.841530    0.862940
2        Random Forest    0.999491   0.896552    0.795918    0.843243    0.834176





# MLFLOW

Three classification models were trained and tracked using MLflow: Logistic Regression, Random Forest, and XGBoost. Their precision, recall, F1-score, and PR-AUC were compared because the dataset is highly imbalanced. The results show that model selection for fraud detection involves a trade-off between detecting fraudulent transactions and limiting false positives. The final model choice should therefore consider the application's relative cost of false negatives and false positives rather than relying on accuracy alone.

MLflow is used for experiment tracking and model management.

The FastAPI application loads the trained XGBoost model from an MLflow run.











#  FastAPI 

The trained fraud detection model was exposed through a FastAPI REST API.

### API Architecture


Client
   ↓
FastAPI
   ↓
MLflow Model
   ↓
XGBoost
   ↓
Fraud Probability
   ↓
Risk Classification


## Endpoints

## GET `/health`

Used to verify that the API is running.

Example response:

{
  "status": "healthy"
}


## POST `/predict`

Accepts transaction features and returns a fraud probability and risk classification.

The current model was trained using the ULB credit-card fraud dataset, whose prediction features are:

* `Time`
* `V1` through `V28`
* `Amount`

Therefore, the API request schema matches these model features.

Example response:

json
{
  "fraud_probability": 0.000009752601727086585,
  "risk": "Low",
  "threshold": 0.5
}




## GET /metrics

Returns basic in-memory application counters including:

Prediction count

Error count

High-risk prediction count

Low-risk prediction count

### Model Integration

The XGBoost model is loaded directly from the MLflow experiment using its run URI.


mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.xgboost.load_model(
    "runs:/<XGBOOST_RUN_ID>/xgboost_model"
)


### Risk Classification

The API converts the model's fraud probability into a risk category using a configurable threshold.

FRAUD_THRESHOLD = 0.5

if fraud_probability >= FRAUD_THRESHOLD:
    risk = "HIGH"
else:
    risk = "LOW"


The threshold is treated as a decision parameter rather than an inherent property of the model. Different thresholds can produce different precision and recall trade-offs.

### Monitoring

The application implements basic monitoring for:

Request logging

HTTP response status

Response time

Prediction counts

Error counts

High-risk and low-risk predictions




### API Documentation

FastAPI provides interactive API documentation at:


http://127.0.0.1:8000/docs


### Testing

The API was tested using:

* `GET /health`
* `POST /predict`
* Multiple test transactions from the fraud detection test set




TRAINING SIDE

data/
   ↓
notebooks/
   ↓
train models
   ↓
MLflow


SERVING SIDE

FastAPI
   ↓
MLflow
   ↓
trained XGBoost model
   ↓
prediction


#### Continuous Integration

This project uses GitHub Actions for continuous integration.

The workflow is triggered when code is pushed to the `main` branch or when a pull request targets `main`.

The CI pipeline performs three checks:

1. **Tests** — runs the project's pytest test suite.
2. **Linting** — checks Python code using Ruff.
3. **Docker build** — builds the FastAPI Docker image to verify that the Dockerfile and application dependencies can be packaged successfully.

The Docker image is built in the GitHub Actions runner. It is not pushed to a container registry at this stage.

The workflow is located at:

.github/workflows/ci.yml


### CI Flow
GitHub Actions is configured to run on pushes and pull requests.

The CI pipeline performs:

Checkout
   ↓
Install dependencies
   ↓
Run pytest
   ↓
Run Ruff
   ↓
Build Docker image

The Docker image is built by GitHub Actions but is not pushed to a container registry.

Workflow:

.github/workflows/ci.yml

#### AWS Architecture

A potential production deployment architecture is documented in:

AWS_ARCHITECTURE.md

The conceptual architecture is:

GitHub
   ↓
GitHub Actions
   ↓
Docker Image
   ↓
Amazon ECR
   ↓
Amazon EC2
   ↓
FastAPI
   ↓
ML Model

AWS deployment was not implemented. AWS services were studied to understand how the application could be deployed in a production environment.

##### Dataset

The original credit card fraud dataset is not included in the GitHub repository because the CSV exceeds GitHub's standard file-size limit.

The dataset is therefore kept locally and excluded through .gitignore.


