import logging
import numpy as np
from math import sqrt
from flaml import AutoML
from pandas import Series, DataFrame
from typing import Any, Dict, Tuple
from sklearn.preprocessing import LabelEncoder


# Constants used to infer problem type
MULTICLASS_UPPER_LIMIT = 1000  # assume regression if dtype is numeric and unique label count is above this limit
LARGE_DATA_THRESHOLD = 1000
REGRESS_THRESHOLD_LARGE_DATA = 0.05
REGRESS_THRESHOLD_SMALL_DATA = 0.1


# Do not change these!
BINARY = "binary"
MULTICLASS = "multiclass"
CLASSIFICATION = "classification"   # our add-on since Flaml uses classification instead of subcategories
REGRESSION = "regression"
SOFTCLASS = "softclass"  # classification with soft-target (rather than classes, labels are probabilities of each class).
QUANTILE = "quantile"  # quantile regression (over multiple quantile levels, which are between 0.0 and 1.0)


# create logger
logger = logging.getLogger("automl_logger")


def infer_problem_type(y: Series, silent=False) -> str:
    """
    Identifies which type of prediction problem we are interested in (if user has not specified).
    Ie. binary classification, multi-class classification, or regression.

    This is built into AutoGluon - but Flaml does not auto-detect the problem type!
    """
    # treat None, NaN, INF, NINF as NA
    y = y.replace([np.inf, -np.inf], np.nan, inplace=False)
    y = y.dropna()
    num_rows = len(y)

    if num_rows == 0:
        raise ValueError("Label column cannot have 0 valid values")

    unique_values = y.unique()

    if num_rows > LARGE_DATA_THRESHOLD:
        regression_threshold = (
            REGRESS_THRESHOLD_LARGE_DATA  # if the unique-ratio is less than this, we assume multiclass classification, even when labels are integers
        )
    else:
        regression_threshold = REGRESS_THRESHOLD_SMALL_DATA

    unique_count = len(unique_values)
    if unique_count == 2:
        problem_type = BINARY
        reason = "only two unique label-values observed"
    elif y.dtype.name in ["object", "category", "string"]:
        problem_type = MULTICLASS
        reason = f"dtype of label-column == {y.dtype.name}"
    elif np.issubdtype(y.dtype, np.floating):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= regression_threshold) and (unique_count <= MULTICLASS_UPPER_LIMIT):
            try:
                can_convert_to_int = np.array_equal(y, y.astype(int))
                if can_convert_to_int:
                    problem_type = MULTICLASS
                    reason = "dtype of label-column == float, but few unique label-values observed and label-values can be converted to int"
                else:
                    problem_type = REGRESSION
                    reason = "dtype of label-column == float and label-values can't be converted to int"
            except:
                problem_type = REGRESSION
                reason = "dtype of label-column == float and label-values can't be converted to int"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == float and many unique label-values observed"
    elif np.issubdtype(y.dtype, np.integer):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= regression_threshold) and (unique_count <= MULTICLASS_UPPER_LIMIT):
            problem_type = MULTICLASS  # TODO: Check if integers are from 0 to n-1 for n unique values, if they have a wide spread, it could still be regression
            reason = "dtype of label-column == int, but few unique label-values observed"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == int and many unique label-values observed"
    else:
        raise NotImplementedError(f"label dtype {y.dtype} not supported!")
    if not silent:
        logger.log(25, f"AutoGluon infers your prediction problem is: '{problem_type}' (because {reason}).")

        # TODO: Move this outside of this function so it is visible even if problem type was not inferred.
        if problem_type in [BINARY, MULTICLASS]:
            if unique_count > 10:
                logger.log(20, f"\tFirst 10 (of {unique_count}) unique label values:  {list(unique_values[:10])}")
            else:
                logger.log(20, f"\t{unique_count} unique label values:  {list(unique_values)}")
        elif problem_type == REGRESSION:
            y_max = y.max()
            y_min = y.min()
            y_mean = y.mean()
            y_stddev = y.std()
            logger.log(20, f"\tLabel info (max, min, mean, stddev): ({y_max}, {y_min}, {round(y_mean, 5)}, {round(y_stddev, 5)})")

        logger.log(
            25,
            f"\tIf '{problem_type}' is not the correct problem_type, please manually specify the problem_type parameter during Predictor init "
            f"(You may specify problem_type as one of: {[BINARY, MULTICLASS, REGRESSION, QUANTILE]})",
        )
    return problem_type


def get_accuracy(task: str, model: AutoML, X_test: DataFrame, y_test: Series) -> str:
    """
    Given a model (via AutoML config) determine its accuracy based on the test sets
    X_test and y_test. The type of ML task is provided such that the correct evaluation
    of the model can be conducted: RMSE for regression and accuracy for classification.

    The final score is returned as a string to provide more context on the type of accuracy
    metric returned.
    """
    score = None
    if task == REGRESSION:
        mse = model.score(X_test, y_test, metric= "mse")
        rmse = round(sqrt(mse), 2)
        score = f"The rmse for this model is {rmse}"
    elif task == CLASSIFICATION:
        accuracy = round(model.score(X_test, y_test), 2)
        score = f"The accuracy for this model is {accuracy * 100}%"

    return score


def get_predictions(
        model: AutoML, X_test: DataFrame, y_test: Series, le: LabelEncoder, label_encoded: bool
    ) -> Tuple[list, list]:
    """
    Given the AutoML model object, we use the model to predict what the X_test values
    should be. These predictions are then combined with the y_test to return predicted
    and actual values.

    We decode the label column and predictions, if it was originally encoded!
    """
    predictions = model.predict(X_test)
    if label_encoded:
        predictions = le.inverse_transform(predictions)
        y_test = le.inverse_transform(y_test)

    return list(predictions), list(y_test)