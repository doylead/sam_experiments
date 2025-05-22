""" Sepearated functions from routing logic """

import logging
from numpy.polynomial.polynomial import polyfit, polyval
from sklearn.metrics import mean_squared_error
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def linear_regression(x, y):
    """
    Performs linear regression and calculates RMSE.

    Args:
        x (list): List of x-values.
        y (list): List of y-values.

    Returns:
        dict: {'params': [p0, p1], 'RMSE': rmse}
              p0: intercept, p1: slope
    """
    if not isinstance(x, list) or not isinstance(y, list):
        logger.error("Input arguments 'x' and 'y' must be lists.")
        raise ValueError("Input arguments 'x' and 'y' must be lists.")
    if not all(isinstance(val, (int, float)) for val in x) or not all(isinstance(val, (int, float)) for val in y):
        logger.error("Input lists 'x' and 'y' must contain numbers.")
        raise ValueError("Input lists 'x' and 'y' must contain numbers.")
    if len(x) != len(y):
        logger.error("Input lists 'x' and 'y' must have the same length.")
        raise ValueError("Input lists 'x' and 'y' must have the same length.")
    if len(x) < 2:
        logger.warning("Input lists 'x' and 'y' have less than 2 elements.  Returning default values.")
        return {'params': [0, 0], 'RMSE': 0}

    try:
        p1, p0 = polyfit(x, y, 1)
        y_predicted = polyval(x, [p0, p1])
        rmse = np.sqrt(mean_squared_error(y, y_predicted))
        logger.info("Linear regression successful. p0: %f, p1: %f, RMSE: %f", p0, p1, rmse)
        return {'params': [p0, p1], 'RMSE': rmse}
    except Exception as e:
        logger.error("Error during linear regression: %s", e)
        raise RuntimeError("Error during linear regression: %s", e)

def quadratic_regression(x, y):
    """
    Performs quadratic regression and calculates RMSE.

    Args:
        x (list): List of x-values.
        y (list): List of y-values.

    Returns:
        dict: {'params': [p0, p1, p2], 'RMSE': rmse}
        p0: intercept, p1: linear term, p2: quadratic term
    """
    if not isinstance(x, list) or not isinstance(y, list):
        logger.error("Input arguments 'x' and 'y' must be lists.")
        raise ValueError("Input arguments 'x' and 'y' must be lists.")
    if not all(isinstance(val, (int, float)) for val in x) or not all(isinstance(val, (int, float)) for val in y):
        logger.error("Input lists 'x' and 'y' must contain numbers.")
        raise ValueError("Input lists 'x' and 'y' must contain numbers.")
    if len(x) != len(y):
        logger.error("Input lists 'x' and 'y' must have the same length.")
        raise ValueError("Input lists 'x' and 'y' must have the same length.")
    if len(x) < 3:
        logger.warning("Input lists 'x' and 'y' have less than 3 elements.  Returning default values.")
        return {'params': [0, 0, 0], 'RMSE': 0}
    try:
        p2, p1, p0 = polyfit(x, y, 2)
        y_predicted = polyval(x, [p0, p1, p2])
        rmse = np.sqrt(mean_squared_error(y, y_predicted))
        logger.info("Quadratic regression successful. p0: %f, p1: %f, p2: %f, RMSE: %f",
                    p0, p1, p2, rmse)
        return {'params': [p0, p1, p2], 'RMSE': rmse}
    except Exception as e:
        logger.error("Error during quadratic regression: %s", e)
        raise RuntimeError("Error during quadratic regression: %s", e)
