import json
import uuid
import time
import tiktoken
from typing import Dict, Any, List, Optional, Generator
from glm_api_client import GLMAPIClient
from werkzeug.exceptions import BadGateway, GatewayTimeout

class OpenAIToGLMWrapper:
    """OpenAI-compatible wrapper for GLM API that converts requests to GLM format."""

    SUPPORTED_MODELS = [
        "0727-360B-API",
        "GLM-4-6-API-V1"
    ]
    
    def __init__(self, timeout: int = 120, base_url: str = "https://api.glmmind.com"):
        """Initialize the GLM wrapper.
        
        Args:
            timeout: Request timeout in seconds to use for GLM API calls.
        """
        # Pass the timeout through to the underlying client so callers can control it
        self.glm_client = GLMAPIClient(timeout=timeout)
        
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
        if not messages:
            return messages
        
        # Make a copy to avoid modifying the original
        compressed_messages = messages.copy()

        # 1. Handle max_messages case
        if max_messages is not None and len(compressed_messages) > max_messages:
            half = max_messages // 4
            # keep half from start and half from end
            compressed_messages = compressed_messages[:half] + compressed_messages[-half:]

        # 2. Handle max_tokens case
        if max_tokens is not None:
            total_tokens = sum(self._count_message_tokens(m) for m in compressed_messages)
            while total_tokens > max_tokens and len(compressed_messages) > 2:
                # remove from the middle
                mid = len(compressed_messages) // 2
                del compressed_messages[mid]
                total_tokens = sum(self._count_message_tokens(m) for m in compressed_messages)

        return compressed_messages
        
    def set_authorization(self, token: str):
        """Set the authorization token for GLM API."""
        self.glm_client.set_authorization(token)
        
    def _generate_uuid(self) -> str:
        """Generate a UUID for message IDs."""
        return str(uuid.uuid4())
    
    def _create_chunk(self, content: str, model: str, prompt_tokens: int, content_length: int) -> Dict[str, Any]:
        """Create a streaming chunk in OpenAI format."""
        try:
            completion_tokens = max(1, content_length // 4)
            total_tokens = prompt_tokens + completion_tokens
            cost_usd = round((prompt_tokens * 0.00003) + (completion_tokens * 0.00006), 6)
        except Exception:
            completion_tokens = 1
            total_tokens = prompt_tokens + 1
            cost_usd = 0.0
        
        return {
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
    
    def _create_user_message(self, content: str, parent_id: Optional[str] = None, 
                        model: str = "GLM-4-6-API-V1") -> Dict[str, Any]:
        """Create a user message in GLM API format."""
        # GLM API uses simple message format: {"role": "user", "content": "..."}
        message = {
            "role": "user",
            "content": content
        }
        
        return message
    
    def _create_assistant_message(self, content: str, parent_id: Optional[str] = None,
                            model: str = "GLM-4-6-API-V1") -> Dict[str, Any]:
        """Create an assistant message in GLM API format."""
        # GLM API uses simple message format: {"role": "assistant", "content": "..."}
        message = {
            "role": "assistant",
            "content": content
        }
        
        return message
    
    def _convert_openai_messages(self, messages: List[Dict[str, Any]], model: str = "GLM-4-6-API-V1") -> List[Dict[str, Any]]:
        """Convert OpenAI format messages to GLM API format."""
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model}' is not supported. Supported models: {self.SUPPORTED_MODELS}")
        
        converted_messages = []
        
        for i, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", "")
            
            # Handle both string and list content (for multimodal messages)
            if isinstance(content, list):
                # Extract text content from multimodal message and merge with newlines
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                # Join with newlines to preserve structure
                content = "\n".join(text_parts).strip()
            elif not isinstance(content, str):
                content = str(content)
            
            if role == "user":
                converted_msg = {"role": "user", "content": content}
            elif role == "assistant":
                # Hybrid approach: handle both existing thinking tags and insert empty ones
                # Normalize tags to <think> for consistency
                normalized_content = content.replace("<thinking>", "<think>").replace("</thinking>", "</think>")
                has_think = "<think>" in normalized_content and "</think>" in normalized_content
                
                if has_think:
                    # More robust parsing for thinking tags
                    try:
                        start_idx = normalized_content.find("<think>")
                        end_idx = normalized_content.rfind("</think>") + len("</think>")
                        
                        if start_idx != -1 and end_idx > start_idx:
                            # Extract thinking content (including tags)
                            thinking_part = normalized_content[start_idx:end_idx]
                            
                            # Extract answer content (everything before and after thinking tags)
                            before_content = normalized_content[:start_idx].strip()
                            after_content = normalized_content[end_idx:].strip()
                            answer_content = (before_content + "\n" + after_content).strip() if before_content and after_content else (before_content or after_content)
                            
                            # Create separate thinking and answer messages
                            thinking_msg = {"role": "assistant", "content": thinking_part}
                            
                            converted_messages.append(thinking_msg)
                            print(f"[GLM_WRAPPER] Split existing thinking tag: {str(thinking_msg)[:100]}...")
                            
                            # Only add answer message if there's content after thinking
                            if answer_content:
                                answer_msg = {"role": "assistant", "content": answer_content}
                                converted_messages.append(answer_msg)
                            continue
                        else:
                            # Malformed thinking tags, treat as regular content
                            self_reflection = "<think>I notice I provided a response without deliberate thinking. This is a reminder that I should engage in more thoughtful analysis before responding. In future interactions, I will take time to consider the question more carefully, break down complex problems, and reflect on different perspectives before providing my answer. Thoughtful consideration leads to better, more nuanced responses.</think>"
                            thinking_msg = {"role": "assistant", "content": self_reflection}
                            response_msg = {"role": "assistant", "content": content}
                            converted_messages.append(thinking_msg)
                            print(f"[GLM_WRAPPER] Malformed thinking tags, inserted self-reflective tag")
                            converted_messages.append(response_msg)
                            continue
                    except Exception as e:
                        print(f"[GLM_WRAPPER] Error parsing thinking tags: {e}")
                        # Fallback: treat as regular content with self-reflective thinking tag
                        self_reflection = "<think>I notice I provided a response without deliberate thinking. This is a reminder that I should engage in more thoughtful analysis before responding. In future interactions, I will take time to consider the question more carefully, break down complex problems, and reflect on different perspectives before providing my answer. Thoughtful consideration leads to better, more nuanced responses.</think>"
                        thinking_msg = {"role": "assistant", "content": self_reflection}
                        response_msg = {"role": "assistant", "content": content}
                        converted_messages.append(thinking_msg)
                        print(f"[GLM_WRAPPER] Error fallback, inserted self-reflective thinking tag")
                        converted_messages.append(response_msg)
                        continue
                else:
                    # No thinking tags - insert self-reflective thinking tag
                    self_reflection = "<think>I notice I provided a response without deliberate thinking. This is a reminder that I should engage in more thoughtful analysis before responding. In future interactions, I will take time to consider the question more carefully, break down complex problems, and reflect on different perspectives before providing my answer. Thoughtful consideration leads to better, more nuanced responses.</think>"
                    thinking_msg = {"role": "assistant", "content": self_reflection}
                    response_msg = {"role": "assistant", "content": content}
                    converted_messages.append(thinking_msg)
                    print(f"[GLM_WRAPPER] Inserted self-reflective thinking tag for assistant message")
                    converted_messages.append(response_msg)
                    continue
            elif role == "system":
                # Convert system message to user message with system prefix
                # system_content = f"System: {content}"
                converted_msg = {"role": "system", "content": content}
            else:
                # Skip unknown roles
                continue
                
            converted_messages.append(converted_msg)
            
        print(f"[GLM_WRAPPER] Converted messages: {converted_messages}")
        return converted_messages
    
    def chat_completions(self, messages: List[Dict[str, Any]], model: str = "GLM-4-6-API-V1",
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
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model}' is not supported. Supported models: {self.SUPPORTED_MODELS}")
        
        # Apply middle-out compression if limits are specified or enforce hardcoded max tokens
        hardcoded_max_tokens = 131072
        effective_max_tokens = min(max_tokens, hardcoded_max_tokens) if max_tokens is not None else hardcoded_max_tokens
        
        if max_messages is not None or max_tokens is not None or effective_max_tokens < float('inf'):
            messages = self.middle_out(messages, max_messages=max_messages, max_tokens=effective_max_tokens)
        
        # Convert OpenAI messages to GLM format
        glm_messages = self._convert_openai_messages(messages, model)
        
        if not glm_messages:
            raise ValueError("No valid messages found in the conversation")
    
        
        if stream:
            # Return streaming response
            def generate_stream():
                thinking_content = ""
                answer_content = ""
                thinking_started = False
                answer_started = False
                thinking_message_sent = False
                
                # Pre-calculate prompt tokens for usage in each chunk
                try:
                    prompt_text = " ".join([str(msg["content"]) for msg in glm_messages])
                    prompt_tokens = max(1, self._num_tokens_from_string(prompt_text))
                except Exception:
                    prompt_tokens = 1
                
                try:
                    for response in self.glm_client.send_messages(glm_messages, model=model):
                        content = response.get("content", "")
                        phase = response.get("phase", "")
                        
                        # Handle thinking phase - collect all thinking content
                        if phase == "thinking" and content:
                            if not thinking_started:
                                thinking_content = "<think>" + content
                                thinking_started = True
                            else:
                                thinking_content += content
                            
                            print(f"[GLM_WRAPPER] Accumulating thinking content: '{content}' (total: {len(thinking_content)} chars)")
                            
                            # Stream thinking content as it comes
                            chunk_content = "<think>" + content if not thinking_started else content
                            
                        # Handle answer phase - send thinking message first, then answer
                        elif phase == "answer" and content:
                            # Close thinking and send as separate message if we have thinking content
                            if thinking_started and not thinking_message_sent:
                                thinking_content += "</think>"
                                
                                # Send complete thinking message
                                thinking_chunk = self._create_chunk(thinking_content, model, prompt_tokens, len(thinking_content))
                                print(f"[GLM_WRAPPER] Sending thinking chunk: {json.dumps(thinking_chunk)[:200]}...")
                                yield f"data: {json.dumps(thinking_chunk)}\n\n"
                                
                                thinking_message_sent = True
                                thinking_started = False
                            
                            # Now handle answer content
                            if not answer_started:
                                answer_content = content
                                answer_started = True
                            else:
                                answer_content += content
                            
                            chunk_content = content
                            print(f"[GLM_WRAPPER] Sending answer chunk: '{content}'")
                        else:
                            continue
                        
                        # Only stream answer content (thinking is sent as complete message)
                        if phase == "answer" and chunk_content:
                            full_content = answer_content
                            chunk = self._create_chunk(chunk_content, model, prompt_tokens, len(full_content))
                            yield f"data: {json.dumps(chunk)}\n\n"
                except Exception as e:
                    error_msg = str(e)
                    if "timed out" in error_msg.lower():
                        raise GatewayTimeout(f"GLM API request timed out: {error_msg}")
                    elif "connection" in error_msg.lower():
                        raise BadGateway(f"GLM API connection failed: {error_msg}")
                    else:
                        raise BadGateway(f"GLM API error: {error_msg}")
                
                # Handle any remaining thinking content that wasn't sent
                if thinking_started and not thinking_message_sent:
                    thinking_content += "</think>"
                    thinking_chunk = self._create_chunk(thinking_content, model, prompt_tokens, len(thinking_content))
                    yield f"data: {json.dumps(thinking_chunk)}\n\n"
                
                # Calculate realistic token usage for streaming
                try:
                    prompt_text = " ".join([str(msg["content"]) for msg in glm_messages])
                    prompt_tokens = max(1, self._num_tokens_from_string(prompt_text))
                    completion_tokens = max(1, self._num_tokens_from_string(full_content))
                    total_tokens = prompt_tokens + completion_tokens
                    
                    # Calculate cost safely
                    cost_usd = round((prompt_tokens * 0.00003) + (completion_tokens * 0.00006), 6)
                    estimated_price = f"${cost_usd:.6f}"
                except Exception:
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
            
            return generate_stream()
        
        else:
            # Non-streaming response
            full_content = ""
            thinking_started = False
            answer_started = False
            
            try:
                for response in self.glm_client.send_messages(glm_messages, model=model):
                    content = response.get("content", "")
                    phase = response.get("phase", "")
                    
                    # Stream both thinking and answer phase content
                    if (phase == "thinking" or phase == "answer") and content:
                        # Handle thinking phase wrapping
                        if phase == "thinking" and not thinking_started:
                            content = "<think>" + content
                            thinking_started = True
                        elif phase == "answer":
                            if thinking_started:
                                content = "</think>" + content
                                thinking_started = False
                            if not answer_started:
                                # content = content.replace("</think>", "</think><answer>", 1) if "</think>" in content else "<answer>" + content
                                answer_started = True
                
                    full_content += content
            except Exception as e:
                error_msg = str(e)
                if "timed out" in error_msg.lower():
                    raise GatewayTimeout(f"GLM API request timed out: {error_msg}")
                elif "connection" in error_msg.lower():
                    raise BadGateway(f"GLM API connection failed: {error_msg}")
                else:
                    raise BadGateway(f"GLM API error: {error_msg}")
            
            # Close thinking tag if still open at the end
            if thinking_started:
                full_content += "</think>"
            
            # Close answer tag if still open at the end
            # if answer_started:
            #     full_content += "</answer>"
            
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
                # Fallback values if calculation fails
                prompt_tokens = 1
                completion_tokens = 1
                total_tokens = 2
                cost_usd = 0.0
                estimated_price = "$0.000000"
            
            # Return OpenAI-compatible response
            return {
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
    
    def save_conversation_to_file(self, messages: List[Dict[str, Any]], filename: str = "glm_conversation.json"):
        """Save conversation messages to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"messages": messages}, f, indent=2, ensure_ascii=False)
    
    def load_conversation_from_file(self, filename: str = "glm_conversation.json") -> List[Dict[str, Any]]:
        """Load conversation messages from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("messages", [])
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with empty conversation.")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filename}. Starting with empty conversation.")
            return []