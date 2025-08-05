import json
import requests
import time
import uuid
from typing import Dict, Any, Optional, Generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GLMAPIClient:
    def __init__(self, authorization_token: str = None, timeout: int = 120):
        """Initialize the GLM API client with authorization token and timeout.
        
        Args:
            authorization_token: Bearer token for API authentication
            timeout: Request timeout in seconds (default: 120)
        """
        self.base_url = "https://chat.z.ai/api/chat/completions"
        # self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.authorization_token = authorization_token
        self.timeout = timeout
        self.session = requests.Session()
        
        # GLM API Client initialized
        
        # Set default headers
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US",
            "content-type": "application/json; charset=utf-8",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-fe-version": "prod-fe-1.0.51"
        }
        
        if self.authorization_token:
            self.headers["authorization"] = f"Bearer {self.authorization_token}"
            
        self.session.headers.update(self.headers)
    
    def set_authorization(self, token: str):
        """Set or update the authorization token."""
        self.authorization_token = token
        self.headers["authorization"] = f"Bearer {token}"
        self.session.headers.update({"authorization": f"Bearer {token}"})
    
    def send_message(self, message: str, chat_id: str = None, model: str = "0727-360B-API") -> Generator[Dict[str, Any], None, None]:
        """Send a single message and get streaming response."""
        messages = [{"role": "user", "content": message}]
        return self.send_messages(messages, chat_id, model)
    
    def send_messages(self, messages: list, chat_id: str = None, model: str = "0727-360B-API") -> Generator[Dict[str, Any], None, None]:
        """Send multiple messages and get streaming response."""
        
        # Generate chat_id if not provided
        if not chat_id:
            chat_id = str(uuid.uuid4())
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        payload = {
            "stream": True,
            "model": model,
            "messages": messages,
            "params": {},
            "tool_servers": [],
            "features": {
                "image_generation": False,
                "code_interpreter": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "features": [
                    {"type": "mcp", "server": "vibe-coding", "status": "hidden"},
                    {"type": "mcp", "server": "ppt-maker", "status": "hidden"},
                    {"type": "mcp", "server": "image-search", "status": "hidden"}
                ]
            },
            "chat_id": chat_id,
            "id": request_id
        }
        
        try:
            
            response = self.session.post(
                self.base_url,
                json=payload,
                stream=True,
                headers={"referrer": f"https://chat.z.ai/c/{chat_id}"},
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                # print(line)
                line = line.decode('utf-8')
                if line.strip():
                    # Remove 'data: ' prefix if present (similar to Qwen client)
                    if line.startswith('data: '):
                        line = line[6:]  # Remove 'data: ' prefix
                    
                    # Skip empty lines after stripping
                    if not line.strip():
                        continue
                    
                    # Handle special SSE messages
                    if line.strip() == '[DONE]':
                        break
                    
                    try:
                        
                        # print(line)
                        
                        # Parse JSON response
                        data = json.loads(line)
                        
                        # print(data)
                        
                        # Extract relevant information
                        if data.get("type") == "chat:completion" and "data" in data:
                            completion_data = data["data"]
                            # Handle both delta_content and edit_content
                            content = completion_data.get("delta_content", "") or completion_data.get("edit_content", "")
                            
                            # If this is edit_content, remove thinking details section
                            if completion_data.get("edit_content") and "</details>" in content:
                                # Find the end of the details tag and cut everything before it
                                details_end = content.find("</details>")
                                if details_end != -1:
                                    content = content[details_end + len("</details>"):].lstrip()
                            
                            phase = completion_data.get("phase", "unknown")
                            
                            result = {
                                "content": content,
                                "phase": phase,
                                "type": data.get("type"),
                                "raw_data": data
                            }
                            
                            yield result
                            
                    except json.JSONDecodeError:
                        continue
            
        except requests.exceptions.Timeout as e:
            raise Exception(f"GLM API request timed out after {self.timeout} seconds. The server may be overloaded or the request is taking too long to process: {e}")
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"GLM API connection failed. Please check your internet connection: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"GLM API request failed: {e}")
    
    def send_message_text_only(self, message: str, chat_id: str = None, model: str = "0727-360B-API") -> Generator[str, None, None]:
        """Send a message and get only text content (simplified)."""
        for response in self.send_message(message, chat_id, model):
            if response.get("content"):
                content = response["content"]
                yield content
    
    def chat_interactive(self, chat_id: str = None):
        """Start an interactive chat session."""
        if not chat_id:
            chat_id = str(uuid.uuid4())
        
        print("GLM-4.5 Interactive Chat (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("\nGLM-4.5: ", end="", flush=True)
                
                full_response = ""
                current_phase = None
                
                for response in self.send_message(user_input, chat_id):
                    content = response.get("content", "")
                    phase = response.get("phase", "")
                    
                    # Handle phase transitions
                    if phase != current_phase:
                        if phase == "thinking":
                            print("\n[Thinking...]", end="", flush=True)
                        elif phase == "answer":
                            print("\n\nGLM-4.5: ", end="", flush=True)
                        current_phase = phase
                    
                    # Only show answer phase content to user
                    if phase == "answer" and content:
                        print(content, end="", flush=True)
                        full_response += content
                
                print()  # New line after response
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")

def main():
    """Example usage"""
    # Initialize client
    client = GLMAPIClient()
    
    # Start interactive chat
    client.chat_interactive()

if __name__ == "__main__":
    main()