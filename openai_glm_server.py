from flask import Flask, request, jsonify, Response
import json
import time
import os
import logging
from dotenv import load_dotenv
from openai_glm_wrapper import OpenAIToGLMWrapper
from openai_glm_wrapper_v2 import OpenAIToGLMWrapperV2
from typing import Dict, Any, List
import threading
from collections import defaultdict
import random


class TokenRotator:
    """Manages rotation of GLM API tokens for load balancing and rate limiting."""
    
    def __init__(self, tokens: List[str]):
        self.tokens = tokens if tokens else []
        self.current_index = 0
        self.lock = threading.Lock()
        self.token_stats = {token: {'requests': 0, 'errors': 0, 'last_used': 0} for token in self.tokens}
        
    def get_next_token(self) -> str:
        """Get the next token in rotation."""
        if not self.tokens:
            return None
            
        with self.lock:
            token = self.tokens[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.tokens)
            self.token_stats[token]['last_used'] = time.time()
            return token
    
    def get_random_token(self) -> str:
        """Get a random token from the pool."""
        if not self.tokens:
            return None
        return random.choice(self.tokens)
    
    def record_request(self, token: str):
        """Record a successful request for a token."""
        if token in self.token_stats:
            with self.lock:
                self.token_stats[token]['requests'] += 1
    
    def record_error(self, token: str):
        """Record an error for a token."""
        if token in self.token_stats:
            with self.lock:
                self.token_stats[token]['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all tokens."""
        with self.lock:
            return {
                'total_tokens': len(self.tokens),
                'token_stats': dict(self.token_stats)
            }
    
    def add_token(self, token: str):
        """Add a new token to the rotation."""
        if token and token not in self.tokens:
            with self.lock:
                self.tokens.append(token)
                self.token_stats[token] = {'requests': 0, 'errors': 0, 'last_used': 0}
    
    def remove_token(self, token: str):
        """Remove a token from the rotation."""
        if token in self.tokens:
            with self.lock:
                self.tokens.remove(token)
                if token in self.token_stats:
                    del self.token_stats[token]
                # Reset index if needed
                if self.current_index >= len(self.tokens) and self.tokens:
                    self.current_index = 0


def load_tokens_from_file(filename: str = 'token.json') -> List[str]:
    """Load tokens from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tokens = data.get('tokens', [])
            logger.info(f"Loaded {len(tokens)} tokens from {filename}")
            return tokens
    except FileNotFoundError:
        logger.error(f"Token file {filename} not found")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in token file {filename}")
        return []
    except Exception as e:
        logger.error(f"Error loading tokens from {filename}: {e}")
        return []


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

# Get timeout configuration from environment (default: 120 seconds)
timeout = int(os.getenv('GLM_TIMEOUT', 120))

# Initialize the GLM wrappers with timeout
wrapper = OpenAIToGLMWrapper()
wrapper_v2 = OpenAIToGLMWrapperV2(timeout=timeout)

# Load tokens from token.json file
tokens = load_tokens_from_file('token.json')

# Fallback to environment variable if no tokens in file
if not tokens:
    glm_token = os.getenv('GLM_API_TOKEN')
    if glm_token:
        tokens = [glm_token]
        logger.info("Using GLM API token from environment variable")
    else:
        logger.warning("No tokens found in token.json and GLM_API_TOKEN environment variable not set.")

# Initialize token rotator
token_rotator = TokenRotator(tokens)

if tokens:
    logger.info(f"Token rotation initialized with {len(tokens)} tokens")
else:
    logger.error("No tokens available for GLM API. Please add tokens to token.json or set GLM_API_TOKEN environment variable.")

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
        
        # Get next token from rotator
        current_token = token_rotator.get_next_token()
        if not current_token:
            return jsonify({"error": "No tokens available"}), 503
        
        # Set the token for this request
        wrapper.set_authorization(current_token)
        
        # Log the full request for debugging
        logger.info(f"[V1] Incoming request - Model: {model}, Stream: {stream}, Token: {current_token[:20]}...")
        logger.info(f"[V1] Full messages: {json.dumps(messages, indent=2, ensure_ascii=False)}")
        logger.info(f"[V1] Additional parameters: {json.dumps({k: v for k, v in data.items() if k not in ['messages', 'model', 'stream']}, indent=2, ensure_ascii=False)}")
        
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
            try:
                response = wrapper.chat_completions(messages, model, stream=False)
                # Record successful request
                token_rotator.record_request(current_token)
                last_successful_request_time = current_time
                return Response(json.dumps(response), mimetype='application/json; charset=utf-8')
            except Exception as e:
                # Record error for this token
                token_rotator.record_error(current_token)
                raise e
            
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
        
        # Get next token from rotator
        current_token = token_rotator.get_next_token()
        if not current_token:
            return jsonify({"error": "No tokens available"}), 503
        
        # Set the token for this request
        wrapper_v2.set_authorization(current_token)
        
        # Log the full request for debugging
        logger.info(f"[V2] Incoming request - Model: {model}, Stream: {stream}, Token: {current_token[:20]}...")
        logger.info(f"[V2] Full messages: {json.dumps(messages, indent=2, ensure_ascii=False)}")
        logger.info(f"[V2] Additional parameters: {json.dumps({k: v for k, v in data.items() if k not in ['messages', 'model', 'stream']}, indent=2, ensure_ascii=False)}")

        # Call the GLM wrapper V2 (answer only, no tags)
        if stream:
            # Streaming response
            def generate():
                try:
                    for chunk in wrapper_v2.chat_completions(messages, model, stream=True):
                        yield chunk
                    # Record successful request
                    token_rotator.record_request(current_token)
                except Exception as e:
                    # Record error for this token
                    token_rotator.record_error(current_token)
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
            try:
                response = wrapper_v2.chat_completions(messages, model, stream=False)
                # Record successful request
                token_rotator.record_request(current_token)
                last_successful_request_time = current_time
                return Response(json.dumps(response), mimetype='application/json; charset=utf-8')
            except Exception as e:
                # Record error for this token
                token_rotator.record_error(current_token)
                raise e
            
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
    logger.info("  GET  /v1/models - List available GLM models")
    logger.info("  GET  /v2/models - List available GLM models for v2 API")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /tokens/stats - Token rotation statistics")
    logger.info("  POST /conversation/save - Save conversation to file")
    logger.info("  POST /conversation/load - Load conversation from file")
    logger.info(f"Server running on http://localhost:{port}")
    
    # Production-ready configuration
    app.run(host='0.0.0.0', port=port, debug=True)