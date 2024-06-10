from sklearn.base import BaseEstimator
import sys
# import mlflow
# import mlflow.sklearn

# from mlflow.tracking import MlflowClient

# MLFLOW_TRACKING_URI = "sqlite:///./mlflow/mlflow.db"


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export(
    model: BaseEstimator,
    # vec: dict,
    *args,
    **kwargs,
):

    print(sys.path)
    # client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # print(client.list_experiments())

    # with mlflow.start_run() as run:
    #     mlflow.sklearn.log_model(
    #         sk_model=model,
    #         artifact_path="sklearn-model",
    #         input_example=X_train,
    #         registered_model_name="sk-learn-random-forest-reg-model",
    #     )

    print("Hola")

    return None