from typing import List, Tuple

from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from mlops.utils_homework.data_preparation.encoders import vectorize_features
from mlops.utils_homework.data_preparation.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
def export(
    df_train: DataFrame, *args, **kwargs
) -> Tuple[
    csr_matrix,
    Series,
    BaseEstimator,
]:
    # df, df_train, df_val = data
    target = kwargs.get('target', 'duration')

    # X, _, _ = vectorize_features(select_features(df))
    # y: Series = df[target]

    X_train, X_val, dv = vectorize_features(
        select_features(df_train),
        None,
    )
    y_train = df_train[target]
    # y_val = df_val[target]

    # modReg = LinearRegression().fit(X_train, y_train)
    # print("Intercept:", modReg.intercept_)

    # return X_train, y_train, dv
    return X_train, y_train, dv
