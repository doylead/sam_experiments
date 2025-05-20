import os
import json
import logging
from numpy.polynomial.polynomial import polyfit, polyval
from sklearn.metrics import mean_squared_error
import numpy as np
from flask import Flask, request, jsonify

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set the logging level

IS_LAMBDA = 'AWS_LAMBDA_FUNCTION_NAME' in os.environ
if not IS_LAMBDA:
    app = Flask(__name__)



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
        logger.info(f"Linear regression successful. p0: {p0}, p1: {p1}, RMSE: {rmse}")
        return {'params': [p0, p1], 'RMSE': rmse}
    except Exception as e:
        logger.error(f"Error during linear regression: {e}")
        raise RuntimeError(f"Error during linear regression: {e}")



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
        logger.info(f"Quadratic regression successful. p0: {p0}, p1: {p1}, p2: {p2}, RMSE: {rmse}")
        return {'params': [p0, p1, p2], 'RMSE': rmse}
    except Exception as e:
        logger.error(f"Error during quadratic regression: {e}")
        raise RuntimeError(f"Error during quadratic regression: {e}")


if not IS_LAMBDA:
    @app.route('/linear', methods=['POST'])
    def linear():
        """
        Flask route for linear regression.
        """
        logger.info("Handling /linear request")
        try:
            data = request.get_json()
            if not data:
                logger.error("Request body is not JSON")
                return jsonify({'error': 'Request body must be JSON'}), 400
            x = data.get('x')
            y = data.get('y')
            if x is None or y is None:
                logger.error("Request body does not contain x or y")
                return jsonify({'error': 'Request body must contain \"x\" and \"y\" arrays.'}), 400
            result = linear_regression(x, y)
            return jsonify(result)
        except ValueError as e:
            logger.warning(f"ValueError: {e}")
            return jsonify({'error': str(e)}), 400
        except RuntimeError as e:
            logger.error(f"RuntimeError: {e}")
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            logger.exception(f"Exception in /linear: {e}")  # logs the full stack trace
            return jsonify({'error': 'Internal server error: ' + str(e)}), 500

    @app.route('/quadratic', methods=['POST'])
    def quadratic():
        """
        Flask route for quadratic regression.
        """
        logger.info("Handling /quadratic request")
        try:
            data = request.get_json()
            if not data:
                logger.error("Request body is not JSON")
                return jsonify({'error': 'Request body must be JSON'}), 400
            x = data.get('x')
            y = data.get('y')
            if x is None or y is None:
                logger.error("Request body does not contain x or y")
                return jsonify({'error': 'Request body must contain \"x\" and \"y\" arrays.'}), 400
            result = quadratic_regression(x, y)
            return jsonify(result)
        except ValueError as e:
            logger.warning(f"ValueError: {e}")
            return jsonify({'error': str(e)}), 400
        except RuntimeError as e:
            logger.error(f"RuntimeError: {e}")
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            logger.exception(f"Exception in /quadratic: {e}")  # logs the full stack trace
            return jsonify({'error': 'Internal server error: ' + str(e)}), 500



def handler(event, context):
    """
    Lambda handler for AWS.  This wraps the Flask app for Lambda.
    """
    # Configure logging for Lambda
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Lambda function started")
    try:
        body = json.loads(event['body'] or '{}') # Handle empty body
        x = body.get('x')
        y = body.get('y')
        logger.debug(f"Parsed body: x={x}, y={y}")

        if event['routeKey'] == 'POST /linear':
            result = linear_regression(x, y)
            logger.info(f"Linear result: {result}")
            response = {
                'statusCode': 200,
                'body': json.dumps(result),
                'headers': {'Content-Type': 'application/json'}
            }
        elif event['routeKey'] == 'POST /quadratic':
            result = quadratic_regression(x, y)
            logger.info(f"Quadratic result: {result}")
            response = {
                'statusCode': 200,
                'body': json.dumps(result),
                'headers': {'Content-Type': 'application/json'}
            }
        else:
            logger.warning(f"Route not found: {event['routeKey']}")
            response = {
                'statusCode': 404,
                'body': json.dumps({'error': 'Not Found'}),
                'headers': {'Content-Type': 'application/json'}
            }
        return response
    except (ValueError, KeyError) as e:
        logger.error(f"Value/Key Error: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON in request body: ' + str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
    except RuntimeError as e:
        logger.error(f"RuntimeError: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
    except Exception as e:
        logger.exception(f"General Exception: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal Server Error: ' + str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
