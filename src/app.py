import logging
import time

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

#monitoring 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s -%(message)s"
)
logger=logging.getLogger(__name__)

prediction_count = 0
error_count = 0
high_risk_count = 0
low_risk_count = 0

#Fast API app
app=FastAPI(
    title="Fraud Detection API",
    description="API for fraud probability prediction",
    version="1.0.0"
)
@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = time.time()

    logger.info(
        "Request started: %s %s",
        request.method,
        request.url.path
    )

    response = await call_next(request)

    response_time = time.time() - start_time

    logger.info(
        "Request completed: %s %s | status=%s | response_time=%.4fs",
        request.method,
        request.url.path,
        response.status_code,
        response_time
    )

    return response

#Ml flow config
mlflow.set_tracking_uri("http://127.0.0.1:5000")

#Model config
MODEL_URI="runs:/c5bb92aad1d241189d60bc603451fe4b/xgboost_model"

#load model
model=mlflow.xgboost.load_model(MODEL_URI)


#REQUEST SCHEMA
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float 
    V3: float 
    V4: float 
    V5: float 
    V6: float 
    V7: float
    V8: float 
    V9: float 
    V10: float
    V11: float 
    V12: float
    V13: float 
    V14: float
    V15: float 
    V16: float 
    V17: float 
    V18: float
    V19: float
    V20: float 
    V21: float 
    V22: float 
    V23: float 
    V24: float 
    V25: float 
    V26: float
    V27: float
    V28: float
    Amount:float =Field(..., ge=0)

#Health checkpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

#Prediction endpoint
@app.post("/predict")
def predict(transaction:Transaction):
    global prediction_count, error_count, high_risk_count, low_risk_count

    try:

        #converting pydantic obj transaction into dict
        transaction_data=transaction.model_dump()
        #convert ths dicnto pandas dataframe cz my model in as trained using datframe
        input_data= pd.DataFrame([transaction_data])

        #CAlculating probability 
        fraud_probability= float(model.predict_proba(input_data)[0][1])
        fraud_threshold=0.5
        # Displaying risk 
        if fraud_probability >= fraud_threshold:
            risk="High"
            high_risk_count +=1
        else:
            risk="Low"
            low_risk_count +=1
        prediction_count +=1
        logger.info(
            "Prediction completed | probability=%.6f | risk=%s | total_predictions=%s",
            fraud_probability,
            risk,
            prediction_count

        )
        #return(fraud_probability,Risk)
        return { "fraud_probability": fraud_probability,                   #round(fraud_probability, 4),
                "risk": risk,
                "threshold": fraud_threshold}

    except Exception:
        error_count += 1

        logger.exception(
            "Prediction failed | total_errors=%s",
            error_count
        )

        raise

@app.get("/metrics")
def metrics():
    return {
        "prediction_count": prediction_count,
        "error_count": error_count,
        "high_risk_count": high_risk_count,
        "low_risk_count": low_risk_count
    }



