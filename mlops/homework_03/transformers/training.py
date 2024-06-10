from sklearn.linear_model import LinearRegression

@transformer
def training_simple(
    training_set: Dict[str, Union[Series, csr_matrix]],
    model_class_name: str,
    *args,
    **kwargs,
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
    Callable[..., BaseEstimator],
]:
    print("Start")
    # X_train, y_train, _ = training_set['build']

    # model_class = load_class(model_class_name)

    # model, metrics, predictions = train_model(
    #     model_class(),
    #     X_train,
    #     y_train
    # )

    return "Hola Mundo"