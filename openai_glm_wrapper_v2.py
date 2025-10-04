import json
import uuid
import time
import logging
import tiktoken
from typing import Dict, Any, List, Optional, Generator
from glm_api_client import GLMAPIClient, RateLimitError
from werkzeug.exceptions import BadGateway, GatewayTimeout, TooManyRequests

# Create logger for this module
logger = logging.getLogger(__name__)

class OpenAIToGLMWrapperV2:
    """OpenAI-compatible wrapper for GLM API that converts requests to GLM format."""
    
    def __init__(self, base_url: str = "https://chat.z.ai/api/chat/completions", timeout: int = 120, max_retries: int = 3):
        """Initialize the GLM wrapper.
        
        Args:
            base_url: Base URL for the GLM API (deprecated, kept for compatibility)
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retry attempts for rate limiting (default: 3)
        """
        self.glm_client = GLMAPIClient(timeout=timeout, max_retries=max_retries)
        self.max_retries = max_retries
        
    def _num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string using cl100k_base encoding."""
        try:
            # Use cl100k_base encoding directly since GLM models aren't in tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(string))
            return num_tokens
        except Exception:
            # Fallback to character-based estimation if tiktoken fails
            return max(1, len(string) // 4)
    
    def _count_message_tokens(self, message: Dict[str, Any]) -> int:
        """Count tokens in a message."""
        content = message.get("content", "")
        return self._num_tokens_from_string(content)
    
    def middle_out(self, messages: List[Dict[str, Any]], max_messages: Optional[int] = None, 
                   max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Apply middle-out compression to a list of messages.

        :param messages: List of dicts [{"role": "user", "content": "..."}, ...]
        :param max_messages: Max number of messages allowed (e.g. 1000 for Claude)
        :param max_tokens: Max token budget (optional)
        :return: Compressed list of messages
        """
        logger.debug(f"middle_out called with {len(messages)} messages, max_messages={max_messages}, max_tokens={max_tokens}")
        if not messages:
            logger.debug("No messages provided, returning as is.")
            return messages
        
        compressed_messages = messages.copy()

        # 1. Handle max_messages case
        if max_messages is not None and len(compressed_messages) > max_messages:
            half = max_messages // 2
            logger.debug(f"Truncating messages: keeping {half} from start and {half} from end.")
            compressed_messages = compressed_messages[:half] + compressed_messages[-half:]
            logger.debug(f"Message count after max_messages truncation: {len(compressed_messages)}")

        # 2. Handle max_tokens case
        if max_tokens is not None:
            total_tokens = sum(self._count_message_tokens(m) for m in compressed_messages)
            logger.debug(f"Total tokens before token trimming: {total_tokens}")
            while total_tokens > max_tokens and len(compressed_messages) > 2:
                mid = len(compressed_messages) // 2
                mid_message = compressed_messages[mid]
                mid_tokens = self._count_message_tokens(mid_message)
                # If removing part of message would suffice
                if total_tokens - mid_tokens < max_tokens:
                    # Calculate how many tokens to keep from the mid message
                    tokens_to_remove = total_tokens - max_tokens
                    content = mid_message.get("content", "")
                    encoding = tiktoken.get_encoding("cl100k_base")
                    tokens = encoding.encode(content)
                    if tokens_to_remove < len(tokens) - 4:  # At least one token and room for ellipsis
                        keep_tokens = len(tokens) - tokens_to_remove
                        truncated = encoding.decode(tokens[:max(1, keep_tokens - 3)]) + "..."
                        compressed_messages[mid]["content"] = truncated
                        logger.debug(f"Truncated message at index {mid} to fit token budget. New length: {self._count_message_tokens(compressed_messages[mid])} tokens")
                        total_tokens = sum(self._count_message_tokens(m) for m in compressed_messages)
                        break
                logger.debug(f"Removing message at index {mid} to reduce tokens.")
                del compressed_messages[mid]
                total_tokens = sum(self._count_message_tokens(m) for m in compressed_messages)
                logger.debug(f"Total tokens after removal: {total_tokens}, messages left: {len(compressed_messages)}")

        logger.debug(f"Returning {len(compressed_messages)} messages after middle_out compression.")
        return compressed_messages
    def set_authorization(self, token: str):
        """Set the authorization token for GLM API."""
        self.glm_client.set_authorization(token)
        logger.info("Authorization token updated for GLM API client")
        
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
                # system_content = f"System: {content}"
                converted_msg = {"role": "system", "content": content}
            else:
                # Skip unknown roles
                continue
                
            converted_messages.append(converted_msg)
            
        return converted_messages
    
    def chat_completions(self, messages: List[Dict[str, Any]], model: str = "0727-360B-API",
                        stream: bool = False, max_messages: Optional[int] = None, 
                        max_tokens: Optional[int] = None, **kwargs) -> Dict[str, Any] | Generator[str, None, None]:
        """OpenAI-compatible chat completions endpoint.
        
        Args:
            messages: List of message dictionaries
            model: Model name to use
            stream: Whether to stream the response
            max_messages: Maximum number of messages (applies middle-out compression)
            max_tokens: Maximum token budget (applies middle-out compression)
            **kwargs: Additional arguments
        """
        
        # Apply middle-out compression if limits are specified or enforce hardcoded max tokens
        hardcoded_max_tokens = 131072
        effective_max_tokens = min(max_tokens, hardcoded_max_tokens) if max_tokens is not None else hardcoded_max_tokens
        
        if max_messages is not None or max_tokens is not None or effective_max_tokens < float('inf'):
            original_count = len(messages)
            messages = self.middle_out(messages, max_messages=max_messages, max_tokens=effective_max_tokens)
            if len(messages) != original_count:
                logger.info(f"Applied middle-out compression: {original_count} -> {len(messages)} messages")
        
        # Convert OpenAI messages to GLM format
        glm_messages = self._convert_openai_messages(messages, model)
        
        if not glm_messages:
            logger.error("No valid messages found after conversion")
            raise ValueError("No valid messages found in the conversation")
        
        logger.info(f"Processing {len(glm_messages)} messages with model {model} (stream={stream})")
        
    
        
        if stream:
            # Return streaming response
            def generate_stream():
                full_content = ""
                first_content_received = False
                thinking_token_sent = False
                
                # Pre-calculate prompt tokens for usage in each chunk
                try:
                    prompt_text = " ".join([str(msg["content"]) for msg in glm_messages])
                    prompt_tokens = max(1, self._num_tokens_from_string(prompt_text))
                except Exception as e:
                    logger.warning(f"Failed to calculate prompt tokens: {e}")
                    prompt_tokens = 1
                
                try:
                    response_count = 0
                    last_response_time = time.time()
                    
                    for response in self.glm_client.send_messages(glm_messages, model=model):
                        response_count += 1
                        current_time = time.time()
                        
                        # Log response metrics for rate limiting analysis
                        if response_count == 1:
                            logger.debug(f"First response received after {current_time - last_response_time:.2f}s")
                        
                        last_response_time = current_time
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
                                            "content": ""
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
                            
                            yield f"data: {json.dumps(thinking_chunk)}\n\n"
                            logger.debug("Sent thinking phase indicator")
                        
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
                                completion_tokens = max(1, self._num_tokens_from_string(full_content))
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
                except RateLimitError as e:
                    logger.error(f"Rate limiting detected in streaming mode: {e}")
                    # Convert to HTTP 429 for OpenAI compatibility
                    raise TooManyRequests(f"Rate limited: {e}")
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"GLM API error in streaming mode: {error_msg}")
                    
                    if "rate limit" in error_msg.lower() or "429" in error_msg:
                        raise TooManyRequests(f"Rate limited: {error_msg}")
                    elif "timed out" in error_msg.lower():
                        raise GatewayTimeout(f"GLM API request timed out: {error_msg}")
                    elif "connection" in error_msg.lower():
                        raise BadGateway(f"GLM API connection failed: {error_msg}")
                    else:
                        raise BadGateway(f"GLM API error: {error_msg}")
                
                # Validate that we received some response
                if response_count == 0:
                    logger.warning("No responses received from GLM API in streaming mode")
                    raise BadGateway("GLM API returned no response data")
                
                logger.debug(f"Streaming completed with {response_count} response chunks")
                
                # Calculate realistic token usage for streaming
                try:
                    prompt_text = " ".join([str(msg["content"]) for msg in glm_messages])
                    prompt_tokens = max(1, self._num_tokens_from_string(prompt_text))
                    completion_tokens = max(1, self._num_tokens_from_string(full_content))
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
                
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                logger.debug("Streaming response completed successfully")
            
            return generate_stream()
        
        else:
            # Non-streaming response
            full_content = ""
            first_content_received = False
            thinking_token_sent = False
            
            try:
                response_count = 0
                start_time = time.time()
                
                for response in self.glm_client.send_messages(glm_messages, model=model):
                    response_count += 1
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
            except RateLimitError as e:
                logger.error(f"Rate limiting detected in non-streaming mode: {e}")
                # Convert to HTTP 429 for OpenAI compatibility
                raise TooManyRequests(f"Rate limited: {e}")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"GLM API error in non-streaming mode: {error_msg}")
                
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    raise TooManyRequests(f"Rate limited: {error_msg}")
                elif "timed out" in error_msg.lower():
                    raise GatewayTimeout(f"GLM API request timed out: {error_msg}")
                elif "connection" in error_msg.lower():
                    raise BadGateway(f"GLM API connection failed: {error_msg}")
                else:
                    raise BadGateway(f"GLM API error: {error_msg}")
            
            # Validate that we received some response
            if response_count == 0:
                logger.warning("No responses received from GLM API in non-streaming mode")
                raise BadGateway("GLM API returned no response data")
            
            end_time = time.time()
            logger.debug(f"Non-streaming completed in {end_time - start_time:.2f}s with {response_count} response chunks")
            
            # Validate that we have actual content
            if not full_content.strip():
                logger.warning("GLM API returned empty content")
                raise BadGateway("GLM API returned empty response content")
            
            # Calculate realistic token usage with error handling
            try:
                prompt_text = " ".join([str(msg["content"]) for msg in glm_messages])
                
                # Use tiktoken for accurate token estimation
                prompt_tokens = max(1, self._num_tokens_from_string(prompt_text))
                completion_tokens = max(1, self._num_tokens_from_string(full_content))
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

