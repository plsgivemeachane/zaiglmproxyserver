import json
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Generator
from glm_api_client import GLMAPIClient
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('glm_server.log'),
        logging.StreamHandler()
    ]
)
# Create logger for this module
logger = logging.getLogger(__name__)

class OpenAIToGLMWrapperV2:
    """OpenAI-compatible wrapper for GLM API that converts requests to GLM format."""
    
    def __init__(self, base_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"):
        """Initialize the GLM wrapper."""
        self.glm_client = GLMAPIClient(base_url)
        
    def set_authorization(self, token: str):
        """Set the authorization token for GLM API."""
        self.glm_client.set_authorization(token)
        
    def _generate_uuid(self) -> str:
        """Generate a UUID for message IDs."""
        generated_uuid = str(uuid.uuid4())
        return generated_uuid
    
    def _create_user_message(self, content: str, parent_id: Optional[str] = None, 
                        model: str = "0727-360B-API") -> Dict[str, Any]:
        """Create a user message in GLM API format."""
        # GLM API uses simple message format: {"role": "user", "content": "..."}
        message = {
            "role": "user",
            "content": content
        }
        
        return message
    
    def _create_assistant_message(self, content: str, parent_id: Optional[str] = None,
                            model: str = "0727-360B-API") -> Dict[str, Any]:
        """Create an assistant message in GLM API format."""
        # GLM API uses simple message format: {"role": "assistant", "content": "..."}
        message = {
            "role": "assistant",
            "content": content
        }
        
        return message
    
    def _convert_openai_messages(self, messages: List[Dict[str, Any]], model: str = "0727-360B-API") -> List[Dict[str, Any]]:
        """Convert OpenAI format messages to GLM API format."""
        converted_messages = []
        
        for i, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", "")
            
            # Handle both string and list content (for multimodal messages)
            if isinstance(content, list):
                # Extract text content from multimodal message
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                    elif isinstance(item, str):
                        text_content += item
                content = text_content
            elif not isinstance(content, str):
                content = str(content)
            
            if role == "user":
                converted_msg = {"role": "user", "content": content}
            elif role == "assistant":
                converted_msg = {"role": "assistant", "content": content}
            elif role == "system":
                # Convert system message to user message with system prefix
                system_content = f"System: {content}"
                converted_msg = {"role": "user", "content": system_content}
            else:
                # Skip unknown roles
                continue
                
            converted_messages.append(converted_msg)
            
        return converted_messages
    
    def chat_completions(self, messages: List[Dict[str, Any]], model: str = "0727-360B-API", 
                        stream: bool = False, **kwargs) -> Dict[str, Any]:
        """OpenAI-compatible chat completions endpoint."""
        
        # Convert OpenAI messages to GLM format
        glm_messages = self._convert_openai_messages(messages, model)
        
        if not glm_messages:
            logger.error("No valid messages found after conversion")
            raise ValueError("No valid messages found in the conversation")
        
    
        
        if stream:
            # Return streaming response
            def generate_stream():
                full_content = ""
                first_content_received = False
                thinking_token_sent = False
                
                # Pre-calculate prompt tokens for usage in each chunk
                try:
                    prompt_text = " ".join([str(msg["content"]) for msg in glm_messages])
                    prompt_tokens = max(1, len(prompt_text) // 4)
                except Exception as e:
                    logger.warning(f"Failed to calculate prompt tokens: {e}")
                    prompt_tokens = 1
                
                for response in self.glm_client.send_messages(glm_messages, model=model):
                    content = response.get("content", "")
                    phase = response.get("phase", "")
                    
                    # Handle thinking phase - send "Think..." token just once
                    if phase == "thinking" and not thinking_token_sent:
                        thinking_token_sent = True
                        thinking_chunk = {
                            "id": f"chatcmpl-{self._generate_uuid()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": "Think..."
                                    },
                                    "finish_reason": None
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": 0,  # Don't count thinking token
                                "total_tokens": prompt_tokens
                            },
                            "pricing": {
                                "input_token_price": 0.00003,
                                "output_token_price": 0.00006,
                                "total_price_usd": 0.0  # No cost for thinking token
                            },
                            "metadata": {
                                "cost_usd": 0.0,
                                "estimated_price": "$0.000000",
                                "provider": "Zhipu (GLM-4.5)"
                            }
                        }
                        
                        # yield f"data: {json.dumps(thinking_chunk)}\n\n"
                    
                    # Capture content from answer phase (and handle edge cases)
                    if phase == "answer" and content:
                        # Handle potential leading space issue on first content
                        if not first_content_received and content.startswith(" "):
                            content = content.lstrip()
                        if not first_content_received:
                            first_content_received = True
                        full_content += content
                        
                        # Calculate current usage for this chunk
                        try:
                            completion_tokens = max(1, len(full_content) // 4)
                            total_tokens = prompt_tokens + completion_tokens
                            cost_usd = round((prompt_tokens * 0.00003) + (completion_tokens * 0.00006), 6)
                        except Exception as e:
                            logger.warning(f"Failed to calculate tokens for chunk: {e}")
                            completion_tokens = 1
                            total_tokens = prompt_tokens + 1
                            cost_usd = 0.0
                        
                        chunk = {
                            "id": f"chatcmpl-{self._generate_uuid()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": content
                                    },
                                    "finish_reason": None
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            },
                            "pricing": {
                                "input_token_price": 0.00003,
                                "output_token_price": 0.00006,
                                "total_price_usd": cost_usd
                            },
                            "metadata": {
                                "cost_usd": cost_usd,
                                "estimated_price": f"${cost_usd:.6f}",
                                "provider": "Zhipu (GLM-4.5)"
                            }
                        }
                        
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                # Calculate realistic fake token usage for streaming
                try:
                    prompt_text = " ".join([str(msg["content"]) for msg in glm_messages])
                    prompt_tokens = max(1, len(prompt_text) // 4)
                    completion_tokens = max(1, len(full_content) // 4)
                    total_tokens = prompt_tokens + completion_tokens
                    
                    # Calculate cost safely
                    cost_usd = round((prompt_tokens * 0.00003) + (completion_tokens * 0.00006), 6)
                    estimated_price = f"${cost_usd:.6f}"
                except Exception as e:
                    logger.warning(f"Failed to calculate final streaming tokens: {e}")
                    # Fallback values if calculation fails
                    prompt_tokens = 1
                    completion_tokens = 1
                    total_tokens = 2
                    cost_usd = 0.0
                    estimated_price = "$0.000000"
                
                # Send final chunk with usage information
                final_chunk = {
                    "id": f"chatcmpl-{self._generate_uuid()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    "pricing": {
                        "input_token_price": 0.00003,
                        "output_token_price": 0.00006,
                        "total_price_usd": cost_usd
                    },
                    "metadata": {
                        "cost_usd": cost_usd,
                        "estimated_price": estimated_price,
                        "provider": "Zhipu (GLM-4.5)"
                    }
                }
                
                yield "data: [DONE]\n\n"
                yield f"data: {json.dumps(final_chunk)}\n\n"
            
            return generate_stream()
        
        else:
            # Non-streaming response
            full_content = ""
            first_content_received = False
            thinking_token_sent = False
            
            for response in self.glm_client.send_messages(glm_messages, model=model):
                content = response.get("content", "")
                phase = response.get("phase", "")
                
                # Handle thinking phase - add "Think..." token just once
                if phase == "thinking" and not thinking_token_sent:
                    thinking_token_sent = True
                    # full_content += "Think..."  # Add thinking token to content
                
                # Capture content from answer phase (and handle edge cases)
                if phase == "answer" and content:
                    # Handle potential leading space issue on first content
                    if not first_content_received and content.startswith(" "):
                        content = content.lstrip()
                    if not first_content_received:
                        first_content_received = True
                    full_content += content
            
            # Calculate realistic fake token usage with error handling
            try:
                prompt_text = " ".join([str(msg["content"]) for msg in glm_messages])
                
                # More realistic token estimation (roughly 4 characters per token)
                prompt_tokens = max(1, len(prompt_text) // 4)
                completion_tokens = max(1, len(full_content) // 4)
                total_tokens = prompt_tokens + completion_tokens
                
                cost_usd = round((prompt_tokens * 0.00003) + (completion_tokens * 0.00006), 6)
                estimated_price = f"${cost_usd:.6f}"
            except Exception as e:
                logger.warning(f"Failed to calculate final non-streaming tokens: {e}")
                # Fallback values if calculation fails
                prompt_tokens = 1
                completion_tokens = 1
                total_tokens = 2
                cost_usd = 0.0
                estimated_price = "$0.000000"
            
            # Return OpenAI-compatible response
            response = {
                "id": f"chatcmpl-{self._generate_uuid()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "pricing": {
                    "input_token_price": 0.00003,
                    "output_token_price": 0.00006,
                    "total_price_usd": cost_usd
                },
                "metadata": {
                    "cost_usd": cost_usd,
                    "estimated_price": estimated_price,
                    "provider": "Zhipu (GLM-4.5)"
                }
            }
            return response
    
    def save_conversation_to_file(self, messages: List[Dict[str, Any]], filename: str = "glm_conversation.json"):
        """Save conversation messages to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"messages": messages}, f, indent=2, ensure_ascii=False)
    
    def load_conversation_from_file(self, filename: str = "glm_conversation.json") -> List[Dict[str, Any]]:
        """Load conversation messages from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                messages = data.get("messages", [])
                return messages
        except FileNotFoundError:
            logger.warning(f"File {filename} not found. Starting with empty conversation.")
            print(f"File {filename} not found. Starting with empty conversation.")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {filename}. Starting with empty conversation.")
            print(f"Invalid JSON in {filename}. Starting with empty conversation.")
            return []

