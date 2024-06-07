import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mlflow
import os

def create_or_get_dataset() -> pd.DataFrame:

    """
    Not input parameters are required.

    We need to have the wheatherAUS.csv file in the same folder than  utils.py.

    Return a dataframe object.
    """
    
    # Directorio donde se encuentra el script
    script_dir = os.path.dirname(__file__)
    csv_file = os.path.join(script_dir, "weatherAUS.csv")

    #obtenemos el dataset
    data = pd.read_csv(csv_file, sep=',')
                        
    return data


def data_preparation(df: pd.DataFrame):
    """
    input: Dataframe

    output: return X_train, X_test, y_train, y_test, numeric_features, categorical_features
    """

    data = df.copy()

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


    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

 

def get_pipeline(model, numeric_features: list, categorical_features: list) -> Pipeline:

    """
        input:
            model: sklearn model
            numeric_features: numeric columns of dataset
            categorical_features: categorical columns of dataset
        output: 
            pipeline ready to train
    """    

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

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return model_pipeline



def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    