from fastapi import FastAPI
import json
import requests
from pydantic import BaseModel, Field
from typing import List
import mlflow
import pickle
import boto3
import numpy as np
import pandas as pd
from typing import Literal
import os

#uvicorn app:app --host 0.0.0.0 --port 8000

def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        #mlflow.set_tracking_uri('http://mlflow:5000')
        #mlflow.set_tracking_uri('http://localhost:5000') 
        os.environ["AWS_ACCESS_KEY_ID"] = "minio"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('C:/Users/franco/OneDrive/Escritorio/ESPECIALIZACIO EN INTELIGENCIA ARTIFICIAL/aprendizaje_maquina_II/amq2-service-ml/notebook_example/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    """try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()
"""
    return model_ml, version_model_ml#, data_dictionary


app = FastAPI()

class RainFeatures(BaseModel):
    Location: object = Field(
        description="Location in Australia",
    )
    MinTemp: float = Field(
        description="Minimum temperature [°C]",
        ge=-9,
        le=34,
    )
    MaxTemp: float = Field(
        description="Maximam temperature [°C]",
        ge=-5,
        le=49,
    )
    Rainfall: float = Field(
        description="amount of rain the day before [ml]",
        ge=0,
        le=371,
    )
    Evaporation: float = Field(
        description="Evaporation",
        ge=0,
        le=145,
    )
    Sunshine: float = Field(
        description="Sunshine",
        ge=0,
        le=15,
    )
    WindGustDir: object = Field(
        description="Wind gust direction [N, W, E, S]",
    )
    WindGustSpeed: float = Field(
        description="Wind gust speed",
        ge=6,
        le=135,
    )
    WindDir9am: object = Field(
        description="Wind direction at 9 a.m [N, W, E, S]",
    )
    WindDir3pm: object = Field(
        description="Wind direction at 3 p.m [N, W, E, S]",
    )
    WindSpeed9am: float = Field(
        description="Wind gust speed at 9 a.m",
        ge=0,
        le=130,
    )
    WindSpeed3pm: float = Field(
        description="Wind gust speed at 3 p.m",
        ge=0,
        le=87,
    )
    Humidity9am: float = Field(
        description="Humidity at 9 a.m",
        ge=0,
        le=100,
    )
    Humidity3pm: float = Field(
        description="Humidity at 3 p.m",
        ge=0,
        le=100,
    )
    Pressure9am: float = Field(
        description="Pressure at 9 a.m",
        ge=980,
        le=1041,
    )
    Pressure3pm: float = Field(
        description="Pressure at 3 p.m",
        ge=977,
        le=1040,
    )
    Cloud9am: float = Field(
        description="Cloud at 9 a.m",
        ge=0,
        le=9,
    )
    Cloud3pm: float = Field(
        description="Cloud at 3 p.m",
        ge=0,
        le=9,
    )
    Temp9am: float = Field(
        description="Temp at 9 a.m",
        ge=-8,
        le=41,
    )
    Temp3pm: float = Field(
        description="Temp at 3 p.m",
        ge=-6,
        le=47,
    )
    RainToday: object = Field(
        description="it rained yesterday? [Yes, No]",
    )
    Year: int = Field(
        description="Year",
        ge=2007,
        le=2017,
    )
    Month: int = Field(
        description="Month",
        ge=1,
        le=12,
    )
    Day: int = Field(
        description="Month",
        ge=1,
        le=31,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Location": "NorfolkIsland",
                    "MinTemp": 22.6,
                    "MaxTemp": 27.0,
                    "Rainfall": 1.0,
                    "Evaporation": 4.8,
                    "Sunshine": 4.6,
                    "WindGustDir": "N",
                    "WindGustSpeed": 44.0,
                    "WindDir9am": "N",
                    "WindDir3pm": "W",
                    "WindSpeed9am": 20.0,
                    "WindSpeed3pm": 13.0,
                    "Humidity9am": 88.0,
                    "Humidity3pm": 78.0,
                    "Pressure9am": 1015.8,
                    "Pressure3pm": 1014.2,
                    "Cloud9am": 7.0,
                    "Cloud3pm": 7.0,
                    "Temp9am": 24.6,
                    "Temp3pm": 25.6,
                    "RainToday": "No",
                    "Year": 2016,
                    "Month": 2,
                    "Day": 29
                }
            ]
        }
    }


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Model Service, you can predict if tomorrow rain or not."}


@app.post("/predict")
async def predict(data: RainFeatures):
    
    features = {
                "Location": data.Location, 
                "MinTemp": data.MinTemp, 
                "MaxTemp": data.MaxTemp, 
                "Rainfall": data.Rainfall, 
                "Evaporation": data.Evaporation, 
                "Sunshine": data.Sunshine, 
                "WindGustDir": data.WindGustDir, 
                "WindGustSpeed": data.WindGustSpeed, 
                "WindDir9am": data.WindDir9am, 
                "WindDir3pm": data.WindDir3pm, 
                "WindSpeed9am": data.WindSpeed9am,
                "WindSpeed3pm": data.WindSpeed3pm,
                "Humidity9am": data.Humidity9am, 
                "Humidity3pm": data.Humidity3pm, 
                "Pressure9am": data.Pressure9am, 
                "Pressure3pm": data.Pressure3pm, 
                "Cloud9am": data.Cloud9am, 
                "Cloud3pm": data.Cloud3pm, 
                "Temp9am": data.Temp9am, 
                "Temp3pm": data.Temp3pm, 
                "RainToday": data.RainToday, 
                "Year": data.Year, 
                "Month": data.Month, 
                "Day": data.Day
            }

    series = pd.Series(features)

    model, _ = load_model(model_name="registered_model", alias="Champion")

    prediction = model.predict(series.to_frame().transpose())

    if prediction[0] == 1:
        output = "Yes"
    else:
        output = "No"

    # Return the prediction result
    return {"¿Va a llover mañana?": output}
     
     

     
     
     
     
     
     
     
     
     
     
     
     
     
     