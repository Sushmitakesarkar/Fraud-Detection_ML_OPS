# Real-Time Fraud Detection MLOps Platform

# Fraud Detection Models

Used the Credit Card Fraud Detection dataset to build initial fraud detection models.

#Models
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





#MLFLOW

Three classification models were trained and tracked using MLflow: Logistic Regression, Random Forest, and XGBoost. Their precision, recall, F1-score, and PR-AUC were compared because the dataset is highly imbalanced. The results show that model selection for fraud detection involves a trade-off between detecting fraudulent transactions and limiting false positives. The final model choice should therefore consider the application's relative cost of false negatives and false positives rather than relying on accuracy alone.
















## Day 25 — FastAPI Prediction Service

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


### Endpoints

#### GET `/health`

Used to verify that the API is running.

Example response:

{
  "status": "healthy"
}


#### POST `/predict`

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


The probability shown above is an example response format; actual predictions depend on the transaction submitted to the model.

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

### API Documentation

FastAPI provides interactive API documentation at:


http://127.0.0.1:8000/docs


### Testing

The API was tested using:

* `GET /health`
* `POST /predict`
* Multiple test transactions from the fraud detection test set


Project architecture:

                    MLflow
               ┌──────────────┐
               │  Run         │
               │  XGBoost     │
               │  Model       │
               └──────┬───────┘
                      │
                      │ MODEL_URI
                      ↓
Client → FastAPI → MLflow Model
             ↓
        Prediction



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

Git Push
   ↓
GitHub Actions
   ↓
┌─────────────┐
│    Tests    │
└──────┬──────┘
       ↓
┌─────────────┐
│    Lint     │
└──────┬──────┘
       ↓
┌─────────────┐
│ Docker Build│
└──────┬──────┘
       ↓
    CI PASS

# Monitoring

## Application Monitoring

The Fraud Detection API implements basic application monitoring using Python logging and in-memory counters.

### Monitored Metrics

- Request method and endpoint
- HTTP response status
- Response time
- Total prediction count
- High-risk prediction count
- Low-risk prediction count
- Prediction/application error count

## Endpoints

### GET /health

Used to verify that the API is running.

### POST /predict

Returns the fraud probability and risk classification.

### GET /metrics

Returns the current in-memory application counters.

Example:

```json
{
  "prediction_count": 10,
  "error_count": 1,
  "high_risk_count": 2,
  "low_risk_count": 8
}
