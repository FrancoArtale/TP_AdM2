from utils import create_or_get_dataset, data_preparation, get_pipeline, get_or_create_experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import mlflow
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def get_metrics_and_figures(y_test, y_pred, y_test_prob):
    
    # Calculamos la accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculamos la precisión
    precision = precision_score(y_test, y_pred)

    # Calculamos la curva de precision-recall
    fig_pr = plt.figure()
    pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_test_prob, ax=plt.gca())
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.close(fig_pr)

    #calculamos la curva Roc
    fig_roc = plt.figure()
    roc_display = RocCurveDisplay.from_predictions(y_test, y_test_prob, ax=plt.gca())
    plt.title("ROC Curve")
    plt.legend()
    plt.close(fig_roc)

    #calculamos la matriz de confución
    fig_cm = plt.figure()
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.legend()
    plt.close(fig_cm)

    return accuracy, precision, fig_pr, fig_cm, fig_roc



if __name__ == "__main__":

    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

    #obtenemos el dataset
    df = create_or_get_dataset()

    #Preparamos los datos
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = data_preparation(df=df)

    #creamos el modelo
    model = RandomForestClassifier()
    
    #creamos el pipeline
    model_pipeline = get_pipeline(model=model, numeric_features=numeric_features, categorical_features=categorical_features)

    #creamos el experimento
    experiment_id = get_or_create_experiment(experiment_name="hyp_test")

    # creamos la busca
    grid_search = GridSearchCV(
        estimator=model_pipeline, 
        param_grid= {
            'classifier__n_estimators': [10, 50, 100, 300],
            'classifier__max_depth': [2, 5, 10],
            'classifier__min_samples_split': [2, 5, 7],
            'classifier__min_samples_leaf': [1, 2]
        }, 
        cv=5, 
        scoring='roc_auc',
        n_jobs = -1,
        verbose=1
    )

    with mlflow.start_run(run_name="hyper", experiment_id=experiment_id) as run:

        # realizamos la busqueda
        grid_search.fit(X_train[:1000], y_train[:1000])

        # tomamos el mejor modelo
        best_model = grid_search.best_estimator_

        # entrenamos el mejor modelo
        best_model.fit(X_train[:1000], y_train[:1000])

        # hacemos predicciones
        y_pred = best_model.predict(X_test)
        y_test_prob = best_model.predict_proba(X_test)[:, 1]

        # calculamos las metricas
        accuracy, precision, fig_pr, fig_cm, fig_roc = get_metrics_and_figures(y_test, y_pred, y_test_prob)

        #almacenamos en mlflow
        mlflow.log_params(best_model["classifier"].get_params())

        mlflow.log_metrics({"Accuracy": accuracy, "Precision": precision})

        mlflow.sklearn.log_model(best_model, artifact_path="random_forest_classifier", registered_model_name='model_optimized')

        # log figures:
        mlflow.log_figure(fig_roc, "metrics/roc_curve.png")
        mlflow.log_figure(fig_cm, "metrics/cm.png")
        mlflow.log_figure(fig_pr, "metrics/pr.png")

        client = mlflow.MlflowClient()
    
        client.set_model_version_tag(name='model_optimized', version="1", key="Model_status", value="Model with a simple hyperparameters optimization")

        client.set_registered_model_alias(name='model_optimized', version="1", alias="Hyp")