from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils_homework.models.sklearn import load_class, train_model

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def training_simple(
    training_set: Dict[str, Union[Series, csr_matrix]],
    *args,
    **kwargs,
) -> Tuple[
    BaseEstimator,
    Dict,
]:
    print("Start training")
    X_train, y_train, dv = training_set['build']

    model_class = load_class("linear_model.LinearRegression")

    model, metrics, predictions = train_model(
        model_class(),
        X_train,
        y_train
    )

    print("Intercept:", model.intercept_)

    # Registry model
    # client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # client.list_experiments()
    # client.create_experiment(name="orchestration")

    # mlflow.sklearn.log_model(
    #     sk_model=model,
    #     artifact_path="sklearn-model",
    #     input_example=X_train,
    #     registered_model_name="sk-learn-linear-regression",
    # )


    return model, dv