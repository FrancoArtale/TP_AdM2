from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import numpy as np
import os
import mlflow
import awswrangler as wr
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json
import boto3
import botocore.exceptions
import logging
from mlflow.data.pandas_dataset import PandasDataset

#https://www.toolify.ai/ai-news/install-python-packages-with-airflow-docker-179333 --> para instalar nuevos paquetes en airflow


# Funcion para calcular los rangos a descartar:
def find_skewed_boundaries(dfdt, variable, distance=1.5):
    IQR = dfdt[variable].quantile(0.75) - dfdt[variable].quantile(0.25)
    lower_boundary = dfdt[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = dfdt[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

# Función que convierte outliers en valores definidos por 'upper'
def censura(x, col, upper, lower):
    return np.where(x[col] > upper, upper,
                    np.where(x[col] < lower, lower, x[col]))


def Extract():

    # dirección para almacenar el archivo
    data_path = "s3://data/raw/Rawdata.csv"
    
    # Directorio donde se encuentra el script
    script_dir = os.path.dirname(__file__)

    # leemos el archivo
    df_path = os.path.join(script_dir, "weatherAUS.csv")
    dataframe = pd.read_csv(df_path, sep=",")

    # Save information of the dataset
    client = boto3.client('s3')

    data_dict = {}
    try:
        client.head_object(Bucket='data', Key='data_info/raw.json')
        result = client.get_object(Bucket='data', Key='data_info/raw.json')
        text = result["Body"].read().decode()
        data_dict = json.loads(text)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] != "404":
            # Something else has gone wrong.
            raise e

    categories_list = [i for i in dataframe.columns if dataframe[i].dtype == object]
    numeric_list = [i for i in dataframe.columns if dataframe[i].dtype != object]
    columns = [i for i in dataframe.columns]

    # Upload JSON String to an S3 Object
    data_dict['all_columns'] = columns
    data_dict['target_col'] = "RainTomorrow"
    data_dict['categorical_columns'] = categories_list
    data_dict['numeric_columns'] = numeric_list
    data_dict['columns_dtypes'] = {k: str(v) for k, v in dataframe.dtypes.to_dict().items()}

    category_dict = {}
    for category in categories_list:
        category_dict[category] = np.sort(dataframe[category].astype(str).unique()).tolist()

    data_dict['categories_values_per_categorical'] = category_dict

    data_dict['date'] = datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
    data_string = json.dumps(data_dict, indent=2)

    client.put_object(
        Bucket='data',  
        Key='data_info/raw.json',
        Body=data_string
    )


    # Hacemos seguimiento en MLflow
    mlflow.set_tracking_uri('http://mlflow:5000')

    try:
        experiment = mlflow.set_experiment("Raining_Tomorrow")
    except Exception as e:
        logging.error(f"Error setting experiment: {e}")
        raise
    
    try:
        with mlflow.start_run(run_name='ETL_run', experiment_id=experiment.experiment_id):
            mlflow.log_param("Fuente de los datos", "https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package")
            mlflow.log_param("ubicacion_archivos", "Archivo almacenado en el bucket de minio")
            mlflow.log_metric("numero_filas", dataframe.shape[0])
            mlflow.log_metric("datos_faltantes", dataframe.isnull().sum().sum())
            dataframe_descripcion = dataframe.describe()
            dataframe_descripcion.to_csv("descripcion_datos.csv")
            mlflow.log_artifact("descripcion_datos.csv")
    except Exception as e:
        logging.error(f"Error in MLflow run: {e}")
        raise
    

    # Almacenamos el archivo en Minio
    wr.s3.to_csv(df=dataframe, path=data_path, index=False)




def Transform():
    
    # Ruta de los datos crudos
    data_original_path = "s3://data/raw/Rawdata.csv"

    # Leemos los datos
    df = wr.s3.read_csv(data_original_path)

    data = df.copy()

    # Transformación de la columna de fecha
    data['Date'] = pd.to_datetime(data['Date'])  # Convertimos a tipo de dato 'datetime'
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data.drop('Date', axis=1, inplace=True)

    # Reducimos cardinalidad de variables de vientos:
    data['WindGustDir'] = data['WindGustDir'].astype(str).str[0]
    data['WindDir9am'] = data['WindDir9am'].astype(str).str[0]
    data['WindDir3pm']= data['WindDir3pm'].astype(str).str[0]

    # Se Eliminan filas con valores faltantes en la variable de salida ('RainTomorrow')
    data = data.dropna(subset=['RainTomorrow'])

    # Dividimos los datos en entrada (X) y salida (y)
    X = data.drop(columns=['RainTomorrow'])  # Entrada
    y = data['RainTomorrow']  # Salida

    # Definimos las columnas numéricas y categóricas
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    y = y.map({'Yes': 1, 'No': 0})

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Censura de outliers, solo variables con grandes cantidades
    col = ['Rainfall', 'Evaporation', 'WindSpeed9am', 'WindSpeed3pm']

    upper, lower = find_skewed_boundaries(X_train, col, 1.5)
    X_train[col] = censura(X_train, col, upper, lower)
    upper, lower = find_skewed_boundaries(X_test, col, 1.5)
    X_test[col] = censura(X_test, col, upper, lower)

    # Definimos las transformaciones para las columnas numéricas (podriamos usar mean o most_frequent)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Definimos las transformaciones para las columnas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combinamos las transformaciones en un preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train_procesado = preprocessor.fit_transform(X_train)
    X_test_procesado = preprocessor.fit_transform(X_test)

    df_xtrain = pd.DataFrame(X_train_procesado.toarray(), columns = preprocessor.get_feature_names_out())
    df_xtest = pd.DataFrame(X_test_procesado.toarray(), columns = preprocessor.get_feature_names_out())

    # al aplicar onehotencoder se genera una columna debido a los indices, la eliminamos:
    if 'num__Unnamed: 0' in df_xtrain.columns:
        df_xtrain.drop('num__Unnamed: 0', axis=1, inplace=True)
    if 'num__Unnamed: 0' in df_xtest.columns:
        df_xtest.drop('num__Unnamed: 0', axis=1, inplace=True)

    # Almacenamos los archivos en Minio
    wr.s3.to_csv(df=df_xtrain, path="s3://data/transform/X_train.csv", index=False)
    wr.s3.to_csv(df=df_xtest, path="s3://data/transform/X_test.csv", index=False)
    wr.s3.to_csv(df=y_train, path="s3://data/transform/y_train.csv", index=False)
    wr.s3.to_csv(df=y_test, path="s3://data/transform/y_test.csv", index=False)

def Load():

    # Leemos los datos
    X_train = wr.s3.read_csv("s3://data/transform/X_train.csv")
    X_test = wr.s3.read_csv("s3://data/transform/X_test.csv")
    y_train = wr.s3.read_csv("s3://data/transform/y_train.csv")
    y_test = wr.s3.read_csv("s3://data/transform/y_test.csv")

    print("X_train tiene la forma: {}, X_test: {}, y_train {}, y_test {}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

default_args = {
    'owner': "AdM2_students",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'start_date': datetime(2024, 6, 6),
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=15)
}

with DAG('ETL_DAG', default_args=default_args, schedule_interval=timedelta(days=1)) as dag:

    t1 = PythonOperator(task_id='Extract', python_callable=Extract)
    t2 = PythonOperator(task_id='Transform', python_callable=Transform)
    t3 = PythonOperator(task_id='Load', python_callable=Load)

    t1 >> t2 >> t3