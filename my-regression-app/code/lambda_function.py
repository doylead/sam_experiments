"""Testing Python web server code that can be run locally using Flask or on AWS Lambda"""

import json
import logging
from flask import Flask, request, jsonify
from myfit import linear_regression, quadratic_regression

app = Flask(__name__)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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
        logger.warning("ValueError: %s", e)
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        logger.error("RuntimeError: %s", e)
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.exception("Exception in /linear: %s", e)
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
        logger.warning("ValueError: %s", e)
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        logger.error("RuntimeError: %s", e)
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.exception("Exception in /quadratic: %s", e)
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500

def handler(event, context):
    """
    Lambda handler for AWS.  This wraps the Flask app for Lambda.
    """
    # Configure logging for Lambda
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)

    logger.info("Lambda function started")
    try:
        body = json.loads(event['body'] or '{}') # Handle empty body
        x = body.get('x')
        y = body.get('y')
        logger.debug("Parsed body: x=%s, y=%s", x, y)

        if event['routeKey'] == 'POST /linear':
            result = linear_regression(x, y)
            logger.info("Linear result: %s", result)
            response = {
                'statusCode': 200,
                'body': json.dumps(result),
                'headers': {'Content-Type': 'application/json'}
            }
        elif event['routeKey'] == 'POST /quadratic':
            result = quadratic_regression(x, y)
            logger.info("Quadratic result: %s", result)
            response = {
                'statusCode': 200,
                'body': json.dumps(result),
                'headers': {'Content-Type': 'application/json'}
            }
        else:
            logger.warning("Route not found: %s", event['routeKey'])
            response = {
                'statusCode': 404,
                'body': json.dumps({'error': 'Not Found'}),
                'headers': {'Content-Type': 'application/json'}
            }
        return response
    except (ValueError, KeyError) as e:
        logger.error("Value/Key Error: %s", e)
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
    except json.JSONDecodeError as e:
        logger.error("JSON Decode Error: %s", e)
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON in request body: ' + str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
    except RuntimeError as e:
        logger.error("RuntimeError: %s", e)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
    except Exception as e:
        logger.exception("General Exception: %s", e)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal Server Error: ' + str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
