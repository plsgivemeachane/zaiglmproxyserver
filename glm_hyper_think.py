import xml.etree.ElementTree as ET
import re
from typing import Dict, Any, Optional, List, Generator
import uuid
from glm_api_client import GLMAPIClient
from reasoning_module import ReasoningModule
from token_rotator import get_token_rotator, initialize_token_rotator_from_config
import json
import time


class ToolXMLParser:
    """
    A parser for XML-formatted tool commands as specified in tool_use.md.
    Handles parsing and execution of 'thoughts' and 'attempt_completion' commands.
    """
    
    def __init__(self):
        self.supported_tools = ['thoughts', 'attempt_completion']
        self.parsed_commands = []
    
    def parse_xml_tools(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse XML-formatted tool commands from text.
        Captures only the first supported tool found.
        
        Args:
            text (str): Input text containing XML tool commands
            
        Returns:
            List[Dict[str, Any]]: List containing at most one parsed tool command
        """
        commands = []
        
        # Find all XML tool blocks using regex
        # Pattern matches: <tool_name>...content...</tool_name>
        pattern = r'<(\w+)>([\s\S]*?)</\1>'
        matches = re.findall(pattern, text)
        
        # Only process the first supported tool found
        for tool_name, content in matches:
            if tool_name in self.supported_tools:
                parsed_tool = self._parse_tool_content(tool_name, content)
                if parsed_tool:
                    commands.append(parsed_tool)
                    break  # Stop after finding the first supported tool
        
        self.parsed_commands.extend(commands)
        return commands
    
    def _parse_tool_content(self, tool_name: str, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse the content within a tool XML block.
        
        Args:
            tool_name (str): Name of the tool
            content (str): XML content within the tool tags
            
        Returns:
            Optional[Dict[str, Any]]: Parsed tool data or None if parsing fails
        """
        try:
            # Wrap content in a root element for proper XML parsing
            xml_content = f"<root>{content}</root>"
            root = ET.fromstring(xml_content)
            
            tool_data = {
                'tool_name': tool_name,
                'parameters': {}
            }
            
            # Extract all parameter elements
            for child in root:
                tool_data['parameters'][child.tag] = child.text or ''
            
            return tool_data
            
        except ET.ParseError as e:
            print(f"Error parsing XML for tool '{tool_name}': {e}")
            return None
    
    def execute_command(self, command: Dict[str, Any]) -> Any:
        """
        Execute a parsed command based on its tool type.
        
        Args:
            command (Dict[str, Any]): Parsed command data
            
        Returns:
            Any: Result of command execution
        """
        tool_name = command.get('tool_name')
        parameters = command.get('parameters', {})
        
        if tool_name == 'thoughts':
            return self._execute_thoughts(parameters)
        elif tool_name == 'attempt_completion':
            return self._execute_attempt_completion(parameters)
        else:
            raise ValueError(f"Unsupported tool: {tool_name}")
    
    def _execute_thoughts(self, parameters: Dict[str, str]) -> str:
        """
        Execute thoughts command - capture and store generated thoughts output.
        
        Args:
            parameters (Dict[str, str]): Command parameters containing thoughts and mode
            
        Returns:
            str: Captured thoughts
        """
        # Extract the generated thoughts and mode from parameters
        thoughts = parameters.get('content', '')
        mode = parameters.get('mode', 'default')
        
        # Store the thoughts for later use
        self._store_thoughts(thoughts, mode)
        
        print(f"[THOUGHTS] {thoughts}")
        return thoughts
    
    def _execute_attempt_completion(self, parameters: Dict[str, str]) -> str:
        """
        Execute attempt_completion command - finalize task result.
        
        Args:
            parameters (Dict[str, str]): Command parameters
            
        Returns:
            str: Completion result
        """
        result = parameters.get('result', '')
        
        if not result:
            raise ValueError("attempt_completion requires 'result' parameter")
        
        completion_message = f"[COMPLETION] {result}"
        print(completion_message)
        return completion_message
    
    def _store_thoughts(self, thoughts: str, mode: str = 'default'):
        """
        Store captured thoughts and mode for later use.
        
        Args:
            thoughts (str): The captured thoughts text
            mode (str): The mode associated with the thoughts
        """
        if not hasattr(self, 'stored_thoughts'):
            self.stored_thoughts = []
        
        self.stored_thoughts.append({
            'thoughts': thoughts,
            'mode': mode,
            'timestamp': self._get_timestamp()
        })
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp for storing thoughts.
        
        Returns:
            str: Current timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_stored_thoughts(self) -> List[Dict[str, str]]:
        """
        Retrieve all stored thoughts.
        
        Returns:
            List[Dict[str, str]]: List of stored thoughts with metadata
        """
        return getattr(self, 'stored_thoughts', [])
    
    def process_text(self, text: str) -> List[Any]:
        """
        Complete workflow: parse XML tools from text and execute them.
        
        Args:
            text (str): Input text containing XML tool commands
            
        Returns:
            List[Any]: Results from executed commands
        """
        commands = self.parse_xml_tools(text)
        results = []
        
        for command in commands:
            try:
                result = self.execute_command(command)
                results.append(result)
            except Exception as e:
                error_msg = f"Error executing {command.get('tool_name', 'unknown')}: {e}"
                print(error_msg)
                results.append(error_msg)
        
        return results
    
    def get_parsed_commands(self) -> List[Dict[str, Any]]:
        """
        Get all parsed commands from the current session.
        
        Returns:
            List[Dict[str, Any]]: List of all parsed commands
        """
        return self.parsed_commands
    
    def clear_commands(self):
        """
        Clear all stored parsed commands.
        """
        self.parsed_commands.clear()

class GLMHyperthinkWrapper: 
    """OpenAI-compatible wrapper for GLM API that converts requests to GLM format."""
    
    def __init__(self, base_url: str = "https://chat.z.ai/api/chat/completions", timeout: int = 120):
        """Initialize the GLM wrapper.
        
        Args:
            base_url: Base URL for the GLM API (deprecated, kept for compatibility)
            timeout: Request timeout in seconds (default: 120)
        """
        self.glm_client = GLMAPIClient(timeout=timeout)
        self.reasoning_module = ReasoningModule(self.glm_client)
        self.token_rotator = get_token_rotator()
        
        # Initialize token rotator if not already initialized
        if not self.token_rotator.tokens:
            self.token_rotator = initialize_token_rotator_from_config()
        
    def set_authorization(self, token: str):
        """Set the authorization token for GLM API."""
        self.glm_client.set_authorization(token)
    
    def _get_next_token(self) -> str:
        """Get the next token from the token rotator."""
        token = self.token_rotator.get_next_token()
        if token:
            self.glm_client.set_authorization(token)
            return token
        else:
            raise ValueError("No tokens available for GLM API. Please configure tokens in token.json or set GLM_API_TOKEN environment variable.")
    
    def record_token_usage(self, token: str, success: bool = True, input_tokens: int = 0, output_tokens: int = 0, cost: float = 0.0):
        """Record token usage statistics."""
        if success:
            self.token_rotator.record_request(token)
        else:
            self.token_rotator.record_error(token)
        
        if input_tokens > 0 or output_tokens > 0:
            self.token_rotator.record_token_usage(token, input_tokens, output_tokens, cost)
        
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
                converted_msg = {"role": "system", "content": content}
            else:
                # Skip unknown roles
                continue
                
            converted_messages.append(converted_msg)
            
        return converted_messages
    
    def chat_completions(self, messages: List[Dict[str, Any]], model: str = "0727-360B-API",
                        stream: bool = False, **kwargs) -> Dict[str, Any] | Generator[str, None, None]:
        """OpenAI-compatible chat completions endpoint."""
        
        # Get next token from rotator
        current_token = self._get_next_token()
        
        # Convert OpenAI messages to GLM format
        glm_messages = self._convert_openai_messages(messages, model)
        
        if not glm_messages:
            self.record_token_usage(current_token, success=False)
            raise ValueError("No valid messages found in the conversation")
        
    
        
        if stream:
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
                
                try:
                    # Use reasoning module for the reasoning phase
                    reasoning_result = self.reasoning_module.run_complete_reasoning(glm_messages, model)
                    
                    if reasoning_result["success"] and reasoning_result["final_message"]:
                        print(f"=== Reasoning Complete - Passing to Answer Phase ===")
                        print(f"Reasoning iterations: {reasoning_result['iterations']}")
                        print(f"Final message role: {reasoning_result['final_message']['role']}")
                        print(f"Final message content length: {len(reasoning_result['final_message']['content'])}")
                        glm_messages.append(reasoning_result["final_message"])
                        print(f"Total messages for answer phase: {len(glm_messages)}")
                    else:
                        # Fallback if reasoning fails
                        print("Reasoning failed or incomplete, proceeding with original messages")
                        
                    # print(glm_messages)
                    
                    for response in self.glm_client.send_messages(glm_messages, model=model, reasoning=False):
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
            
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                    # Record successful token usage for streaming
                    completion_tokens = max(1, len(full_content) // 4)
                    total_tokens = prompt_tokens + completion_tokens
                    cost_usd = (prompt_tokens * 0.00003) + (completion_tokens * 0.00006)
                    
                    self.record_token_usage(
                        current_token,
                        success=True,
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                        cost=cost_usd
                    )
                        
                except Exception as e:
                    # Record failed token usage
                    self.record_token_usage(current_token, success=False)
                    # Send error response
                    error_chunk = {
                        "id": f"chatcmpl-{self._generate_uuid()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"Error: {str(e)}"},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return generate_stream()
        
        else:
            # Non-streaming response
            try:
                full_content = ""
                first_content_received = False
                thinking_token_sent = False
                
                # Use reasoning module for the reasoning phase
                reasoning_result = self.reasoning_module.run_complete_reasoning(glm_messages, model)
                
                if reasoning_result["success"] and reasoning_result["final_message"]:
                    print(f"=== Reasoning Complete - Passing to Answer Phase ===")
                    print(f"Reasoning iterations: {reasoning_result['iterations']}")
                    print(f"Final message role: {reasoning_result['final_message']['role']}")
                    print(f"Final message content length: {len(reasoning_result['final_message']['content'])}")
                    glm_messages.append(reasoning_result["final_message"])
                    print(f"Total messages for answer phase: {len(glm_messages)}")
                else:
                    # Fallback if reasoning fails
                    print("Reasoning failed or incomplete, proceeding with original messages")
            
                for response in self.glm_client.send_messages(glm_messages, model=model, reasoning=False):
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
                
                # Record successful token usage
                self.record_token_usage(
                    current_token, 
                    success=True, 
                    input_tokens=prompt_tokens, 
                    output_tokens=completion_tokens, 
                    cost=cost_usd
                )
                
                return response
                
            except Exception as e:
                # Record failed token usage
                self.record_token_usage(current_token, success=False)
                # Return error response
                return {
                    "id": f"chatcmpl-{self._generate_uuid()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
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



# Example usage and testing
if __name__ == "__main__":
    # Initialize GLM wrapper
    wrapper = GLMHyperthinkWrapper()
    wrapper.set_authorization("eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjRjNjMzZTRlLWVmNWQtNGI3OS04ZDA2LWZiY2Q5ZjQyNjAxMCIsImVtYWlsIjoicmFraXM5MDc2QGdtYWlsLmNvbSJ9.wpviZL1aEir2pJ1vQ524fAiCDcdT2mFo7U9qHtnnuvqXXcF3fMYLpa5i4wxahp_BiwbsEUgTQZ04obkd5Zu2qA")  # Replace with actual token
    
    # Test conversation with system and user messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant focused on reasoning and problem-solving."
        },
        {
            "role": "user",
            "content": "What is the value of pi"
        }
    ]
    
    print("Testing GLM Hyperthink Wrapper:")
    print("=" * 40)
    
    # Test streaming response
    print("\nTesting Streaming Response:")
    try:
        for chunk in wrapper.chat_completions(messages, stream=True):
            if chunk.startswith("data: "):
                chunk_data = json.loads(chunk[6:])
                content = chunk_data["choices"][0]["delta"].get("content", "")
                if content:
                    print(content, end="", flush=True)
    except Exception as e:
        print(f"\nError during streaming: {e}")
    
    # Test non-streaming response
    # print("\n\nTesting Non-streaming Response:")
    # try:
    #     response = wrapper.chat_completions(messages, stream=False)
    #     print(f"Response: {response}")
    #     print(f"Model: {response.get('model')}")
    #     print(f"Usage: {response.get('usage')}")
    # except Exception as e:
    #     print(f"Error during non-streaming call: {e}")
