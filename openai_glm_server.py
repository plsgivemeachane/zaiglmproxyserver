from flask import Flask, request, jsonify, Response
import json
import time
import os
import logging
from dotenv import load_dotenv
from openai_glm_wrapper import OpenAIToGLMWrapper
from openai_glm_wrapper_v2 import OpenAIToGLMWrapperV2

# Add this import if SUPPORTED_MODELS exists in v2 wrapper
from openai_glm_wrapper_v2 import SUPPORTED_MODELS as SUPPORTED_MODELS_V2
from glm_hyper_think import GLMHyperthinkWrapper
from typing import Dict, Any, List
import threading
from collections import defaultdict
import random
from token_rotator import TokenRotator, load_tokens_from_file, initialize_token_rotator_from_config

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('glm_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Disable Flask's default logging in production
if not app.debug:
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

# Get timeout configuration from environment (default: 30 seconds)
timeout = int(os.getenv('GLM_TIMEOUT', 30))
logger.info(f"GLM effective request timeout set to {timeout} seconds")

# Initialize the GLM wrappers with timeout
wrapper = OpenAIToGLMWrapper(timeout=timeout)
wrapper_v2 = OpenAIToGLMWrapperV2(timeout=timeout)
wrapper_v3 = GLMHyperthinkWrapper(timeout=timeout)

# Initialize token rotator singleton
token_rotator = initialize_token_rotator_from_config()

# Rate limiting variables
last_successful_request_time = 0
RATE_LIMIT_SECONDS = 5

# Per-user rate limiting (thread-safe) - kept for other endpoints
user_rate_limits = defaultdict(float)  # user_id -> last_request_time
rate_limit_lock = threading.Lock()


def handle_chat_completions(wrapper, version='v1'):
    """Shared handler for chat completions across v1 and v2 endpoints."""
    global last_successful_request_time
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract required parameters
        messages = data.get('messages', [])
        model = data.get('model', '0727-360B-API')
        stream = data.get('stream', False)
        
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Validate messages format
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                return jsonify({"error": "Invalid message format. Each message must have 'role' and 'content'"}), 400
            if msg['role'] not in ['user', 'assistant', 'system']:
                return jsonify({"error": f"Invalid role: {msg['role']}. Must be 'user', 'assistant', or 'system'"}), 400
        
        # Get next token from rotator
        current_token = token_rotator.get_next_token()
        if not current_token:
            return jsonify({"error": "No tokens available"}), 503
        
        # Set the token for this request
        wrapper.set_authorization(current_token)
        
        # Log the full request for debugging
        logger.info(f"[{version.upper()}] Incoming request - Model: {model}, Stream: {stream}, Token: {current_token[:20]}...")
        logger.info(f"[{version.upper()}] Full messages: {json.dumps(messages, indent=2, ensure_ascii=False)}")
        logger.info(f"[{version.upper()}] Additional parameters: {json.dumps({k: v for k, v in data.items() if k not in ['messages', 'model', 'stream']}, indent=2, ensure_ascii=False)}")
        
        # Call the GLM wrapper
        if stream:
            # Streaming response
            def generate():
                try:
                    for chunk in wrapper.chat_completions(messages, model, stream=True):
                        yield chunk
                    # Record successful request
                    token_rotator.record_request(current_token)
                except Exception as e:
                    # Record error for this token
                    token_rotator.record_error(current_token)
                    # Re-raise the exception to be handled by Flask's error handlers
                    raise e
            
            return Response(generate(), mimetype='text/plain; charset=utf-8')
        else:
            # Non-streaming response
            response = wrapper.chat_completions(messages, model, stream=False)
            # Record successful request
            token_rotator.record_request(current_token)
            
            # Record token usage and cost if available (only for dict responses)
            if isinstance(response, dict) and 'usage' in response and 'pricing' in response:
                usage = response['usage']
                pricing = response['pricing']
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                cost = pricing.get('total_price_usd', 0.0)
                token_rotator.record_token_usage(current_token, input_tokens, output_tokens, cost)
            
            return Response(json.dumps(response), mimetype='application/json; charset=utf-8')
            
    except Exception as e:
        # Record error for this token if we have one
        if 'current_token' in locals():
            token_rotator.record_error(current_token)
        
        logger.error(f"Error in chat_completions {version}: {e}")
        
        # Handle specific error types with appropriate HTTP status codes
        from werkzeug.exceptions import GatewayTimeout, BadGateway
        if isinstance(e, GatewayTimeout):
            return jsonify({"error": {"message": str(e), "type": "timeout", "code": "gateway_timeout"}}), 504
        elif isinstance(e, BadGateway):
            return jsonify({"error": {"message": str(e), "type": "connection", "code": "bad_gateway"}}), 502
        else:
            return jsonify({"error": {"message": str(e), "type": "internal_error", "code": "internal_server_error"}}), 500


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint using GLM API."""
    return handle_chat_completions(wrapper, 'v1')

@app.route('/v2/chat/completions', methods=['POST'])
def chat_completions_v2():
    """OpenAI-compatible chat completions endpoint using GLM API V2 (answer only, no tags)."""
    return handle_chat_completions(wrapper_v2, 'v2')


@app.route('/v3/chat/completions', methods=['POST'])
def chat_completions_v3():
    """OpenAI-compatible chat completions endpoint with hyperthink reasoning (v3)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        messages = data.get('messages', [])
        model = data.get('model', '0727-360B-API')
        stream = data.get('stream', False)
        
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Get next token from rotator
        current_token = token_rotator.get_next_token()
        if not current_token:
            return jsonify({"error": "No available tokens"}), 503
        
        current_time = time.time()
        
        if stream:
            # Streaming response
            def generate():
                try:
                    for chunk in wrapper_v3.chat_completions(messages, model, stream=True):
                        if chunk:
                            yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    # Record successful request
                    token_rotator.record_request(current_token)
                except Exception as e:
                    # Record error for this token
                    token_rotator.record_error(current_token)
                    # Re-raise the exception to be handled by Flask's error handlers
                    raise e
            
            return Response(generate(), mimetype='text/plain; charset=utf-8')
        else:
            # Non-streaming response
            response = wrapper_v3.chat_completions(messages, model, stream=False)
            # Record successful request
            token_rotator.record_request(current_token)
            
            # Record token usage and cost if available (only for dict responses)
            if isinstance(response, dict) and 'usage' in response and 'pricing' in response:
                usage = response['usage']
                pricing = response['pricing']
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                cost = pricing.get('total_price_usd', 0.0)
                token_rotator.record_token_usage(current_token, input_tokens, output_tokens, cost)
            
            return Response(json.dumps(response), mimetype='application/json; charset=utf-8')
            
    except Exception as e:
        # Record error for this token if we have one
        if 'current_token' in locals():
            token_rotator.record_error(current_token)
        
        logger.error(f"Error in chat_completions v3: {e}")
        
        # Handle specific error types with appropriate HTTP status codes
        from werkzeug.exceptions import GatewayTimeout, BadGateway
        if isinstance(e, GatewayTimeout):
            return jsonify({"error": {"message": str(e), "type": "timeout", "code": "gateway_timeout"}}), 504
        elif isinstance(e, BadGateway):
            return jsonify({"error": {"message": str(e), "type": "connection", "code": "bad_gateway"}}), 502
        else:
            return jsonify({"error": {"message": str(e), "type": "internal_error", "code": "internal_server_error"}}), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available GLM models."""
    models_data = {
        "object": "list",
        "data": [
            {
                "id": "0727-360B-API",
                "object": "model",
                "created": 1677610602,
                "owned_by": "glm-api",
                "pricing": {
                    "input_token_price": 0.00003,
                    "output_token_price": 0.00006
                },
                "metadata": {
                    "provider": "GLM",
                    "model_type": "chat",
                    "context_length": 8192
                }
            }
        ]
    }
    return Response(json.dumps(models_data), mimetype='application/json; charset=utf-8')

@app.route('/v2/models', methods=['GET'])
def list_models_v2():
    """List available GLM models for v2 API."""
    models_data = {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1677610602,
                "owned_by": "glm-api",
                "pricing": {
                    "input_token_price": 0.00003,
                    "output_token_price": 0.00006
                },
                "metadata": {
                    "provider": "GLM",
                    "model_type": "chat",
                    "context_length": 8192
                }
            } for model_id in SUPPORTED_MODELS_V2
        ]
    }
    return Response(json.dumps(models_data), mimetype='application/json; charset=utf-8')

@app.route('/v3/models', methods=['GET'])
def list_models_v3():
    """List available GLM models for v3 API (hyperthink)."""
    models_data = {
        "object": "list",
        "data": [
            {
                "id": "0727-360B-API",
                "object": "model",
                "created": 1677610602,
                "owned_by": "glm-api",
                "pricing": {
                    "input_token_price": 0.00003,
                    "output_token_price": 0.00006
                },
                "metadata": {
                    "provider": "GLM",
                    "model_type": "chat",
                    "context_length": 8192,
                    "features": ["reasoning", "hyperthink"]
                }
            }
        ]
    }
    return Response(json.dumps(models_data), mimetype='application/json; charset=utf-8')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health_data = {"status": "healthy", "service": "openai-glm-wrapper"}
    return Response(json.dumps(health_data), mimetype='application/json; charset=utf-8')

@app.route('/tokens/stats', methods=['GET'])
def token_stats():
    """Get token rotation statistics."""
    try:
        stats = token_rotator.get_stats()
        return jsonify({
            "status": "success",
            "timestamp": int(time.time()),
            "data": stats
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500





@app.route('/conversation/save', methods=['POST'])
def save_conversation():
    """Save conversation to file."""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        filename = data.get('filename', 'glm_conversation.json')
        
        wrapper.save_conversation_to_file(messages, filename)
        success_data = {"message": f"Conversation saved to {filename}"}
        return Response(json.dumps(success_data), mimetype='application/json; charset=utf-8')
    except Exception as e:
        error_data = {"error": str(e)}
        return Response(json.dumps(error_data), mimetype='application/json; charset=utf-8', status=500)

@app.route('/conversation/load', methods=['POST'])
def load_conversation():
    """Load conversation from file."""
    try:
        data = request.get_json()
        filename = data.get('filename', 'glm_conversation.json')
        
        messages = wrapper.load_conversation_from_file(filename)
        messages_data = {"messages": messages}
        return Response(json.dumps(messages_data), mimetype='application/json; charset=utf-8')
    except Exception as e:
        error_data = {"error": str(e)}
        return Response(json.dumps(error_data), mimetype='application/json; charset=utf-8', status=500)



@app.errorhandler(404)
def not_found(error):
    error_data = {"error": "Endpoint not found"}
    return Response(json.dumps(error_data), mimetype='application/json; charset=utf-8', status=404)

@app.errorhandler(405)
def method_not_allowed(error):
    error_data = {"error": "Method not allowed"}
    return Response(json.dumps(error_data), mimetype='application/json; charset=utf-8', status=405)

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.getenv('PORT', 5000))
    
    logger.info("Starting OpenAI-to-GLM Wrapper API Server...")
    logger.info("Available endpoints:")
    logger.info("  POST /v1/chat/completions - OpenAI-compatible chat completions (with thinking tags) [BLOCKING]")
    logger.info("  POST /v2/chat/completions - OpenAI-compatible chat completions (answer only, no tags) [BLOCKING]")
    logger.info("  POST /v3/chat/completions - OpenAI-compatible chat completions with hyperthink reasoning [BLOCKING]")
    logger.info("  GET  /v1/models - List available GLM models")
    logger.info("  GET  /v2/models - List available GLM models for v2 API")
    logger.info("  GET  /v3/models - List available GLM models for v3 API (hyperthink)")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /tokens/stats - Token rotation statistics")
    logger.info("  POST /conversation/save - Save conversation to file")
    logger.info("  POST /conversation/load - Load conversation from file")
    logger.info(f"Server running on http://localhost:{port}")
    
    # Production-ready configuration
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
