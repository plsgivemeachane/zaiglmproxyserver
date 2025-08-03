from flask import Flask, request, jsonify, Response
import json
import time
import os
import logging
from dotenv import load_dotenv
from openai_glm_wrapper import OpenAIToGLMWrapper
from openai_glm_wrapper_v2 import OpenAIToGLMWrapperV2
from typing import Dict, Any
import threading
from collections import defaultdict


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

# Initialize the GLM wrappers
wrapper = OpenAIToGLMWrapper()
wrapper_v2 = OpenAIToGLMWrapperV2()

# Load GLM API token from environment variable
glm_token = os.getenv('GLM_API_TOKEN')
if glm_token:
    wrapper.set_authorization(glm_token)
    wrapper_v2.set_authorization(glm_token)
    logger.info("GLM API token loaded successfully")
else:
    logger.warning("GLM_API_TOKEN environment variable not set. Please set it in your .env file.")

# Rate limiting variables
last_successful_request_time = 0
RATE_LIMIT_SECONDS = 5

# Per-user rate limiting (thread-safe) - kept for other endpoints
user_rate_limits = defaultdict(float)  # user_id -> last_request_time
rate_limit_lock = threading.Lock()


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint using GLM API."""
    global last_successful_request_time
    
    try:
        # Check rate limiting
        current_time = time.time()
        time_since_last_request = current_time - last_successful_request_time
        
        if last_successful_request_time > 0 and time_since_last_request < RATE_LIMIT_SECONDS:
            remaining_time = RATE_LIMIT_SECONDS - time_since_last_request
            time.sleep(remaining_time)
            current_time = time.time()
        
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
        
        # Log the full request for debugging
        logger.info(f"[V1] Incoming request - Model: {model}, Stream: {stream}")
        logger.info(f"[V1] Full messages: {json.dumps(messages, indent=2, ensure_ascii=False)}")
        logger.info(f"[V1] Additional parameters: {json.dumps({k: v for k, v in data.items() if k not in ['messages', 'model', 'stream']}, indent=2, ensure_ascii=False)}")
        
        # Call the GLM wrapper
        if stream:
            # Streaming response
            def generate():
                try:
                    for chunk in wrapper.chat_completions(messages, model, stream=True):
                        yield chunk
                except Exception as e:
                    error_chunk = {
                        "id": f"chatcmpl-error-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"Error: {str(e)}"},
                            "finish_reason": "error"
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            last_successful_request_time = current_time
            return Response(generate(), mimetype='text/plain; charset=utf-8')
        else:
            # Non-streaming response
            response = wrapper.chat_completions(messages, model, stream=False)
            last_successful_request_time = current_time
            return Response(json.dumps(response), mimetype='application/json; charset=utf-8')
            
    except Exception as e:
        print(f"Error in chat_completions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/v2/chat/completions', methods=['POST'])
def chat_completions_v2():
    """OpenAI-compatible chat completions endpoint using GLM API V2 (answer only, no tags)."""
    global last_successful_request_time
    
    try:
        # Check rate limiting
        current_time = time.time()
        time_since_last_request = current_time - last_successful_request_time
        
        if last_successful_request_time > 0 and time_since_last_request < RATE_LIMIT_SECONDS:
            remaining_time = RATE_LIMIT_SECONDS - time_since_last_request
            # Wait for the remaining time instead of returning an error
            time.sleep(remaining_time)
            # Update the current time after sleeping
            current_time = time.time()
        
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
        
        # Log the full request for debugging
        logger.info(f"[V2] Incoming request - Model: {model}, Stream: {stream}")
        logger.info(f"[V2] Full messages: {json.dumps(messages, indent=2, ensure_ascii=False)}")
        logger.info(f"[V2] Additional parameters: {json.dumps({k: v for k, v in data.items() if k not in ['messages', 'model', 'stream']}, indent=2, ensure_ascii=False)}")

        # Call the GLM wrapper V2 (answer only, no tags)
        if stream:
            # Streaming response
            def generate():
                try:
                    for chunk in wrapper_v2.chat_completions(messages, model, stream=True):
                        yield chunk
                except Exception as e:
                    error_chunk = {
                        "id": f"chatcmpl-error-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"Error: {str(e)}"},
                            "finish_reason": "error"
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            last_successful_request_time = current_time
            return Response(generate(), mimetype='text/plain; charset=utf-8')
        else:
            # Non-streaming response
            response = wrapper_v2.chat_completions(messages, model, stream=False)
            last_successful_request_time = current_time
            return Response(json.dumps(response), mimetype='application/json; charset=utf-8')
            
    except Exception as e:
        print(f"Error in chat_completions_v2: {e}")
        return jsonify({"error": str(e)}), 500

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health_data = {"status": "healthy", "service": "openai-glm-wrapper"}
    return Response(json.dumps(health_data), mimetype='application/json; charset=utf-8')





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
    logger.info("  GET  /v1/models - List available GLM models")
    logger.info("  GET  /v2/models - List available GLM models for v2 API")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /conversation/save - Save conversation to file")
    logger.info("  POST /conversation/load - Load conversation from file")
    logger.info(f"Server running on http://localhost:{port}")
    
    # Production-ready configuration
    app.run(host='0.0.0.0', port=port, debug=True)