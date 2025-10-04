import json
import requests
import time
import uuid
import random
from typing import Dict, Any, Optional, Generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    """Custom exception for rate limiting scenarios."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after

class GLMAPIClient:
    def __init__(self, authorization_token: Optional[str] = None, timeout: int = 120, max_retries: int = 3):
        """Initialize the GLM API client with authorization token and timeout.
        
        Args:
            authorization_token: Bearer token for API authentication
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retry attempts for rate limiting (default: 3)
        """
        self.base_url = "https://chat.z.ai/api/chat/completions"
        self.authorization_token = authorization_token
        self.timeout = timeout
        self.max_retries = max_retries
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
    
    def _calculate_backoff_delay(self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff delay with jitter.
        
        Args:
            attempt: Current retry attempt (0-based)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            
        Returns:
            Delay in seconds with exponential backoff and jitter
        """
        # Exponential backoff: base_delay * (2 ^ attempt)
        delay = base_delay * (2 ** attempt)
        # Cap at max_delay
        delay = min(delay, max_delay)
        # Add jitter (±25% randomization)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)
    
    def _is_rate_limited(self, response: requests.Response) -> bool:
        """Check if the response indicates rate limiting.
        
        Args:
            response: HTTP response object
            
        Returns:
            True if rate limited, False otherwise
        """
        # Check for HTTP 429 (Too Many Requests)
        if response.status_code == 429:
            return True
        
        # Check for other rate limiting indicators
        if response.status_code in [502, 503, 504]:  # Bad Gateway, Service Unavailable, Gateway Timeout
            return True
            
        # Check response headers for rate limiting indicators
        rate_limit_headers = ['x-ratelimit-remaining', 'x-rate-limit-remaining', 'ratelimit-remaining']
        for header in rate_limit_headers:
            if header in response.headers and response.headers[header] == '0':
                return True
                
        return False
    
    def _get_retry_after(self, response: requests.Response) -> Optional[int]:
        """Extract retry-after value from response headers.
        
        Args:
            response: HTTP response object
            
        Returns:
            Retry-after delay in seconds, or None if not specified
        """
        retry_after = response.headers.get('retry-after')
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                # Handle date format (not common for rate limiting)
                pass
        return None
    
    def _validate_streaming_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate that streaming response data is complete and valid.
        
        Args:
            response_data: Parsed response data
            
        Returns:
            True if response is valid, False otherwise
        """
        # Check for required fields
        if not isinstance(response_data, dict):
            return False
            
        # Check for completion type
        if response_data.get("type") != "chat:completion":
            return False
            
        # Check for data field
        if "data" not in response_data:
            return False
            
        data = response_data["data"]
        if not isinstance(data, dict):
            return False
            
        # Check for phase information
        phase = data.get("phase")
        if phase not in ["thinking", "answer", "unknown"]:
            logger.warning(f"Unexpected phase in response: {phase}")
            
        return True
    
    def send_message(self, message: str, chat_id: Optional[str] = None, model: str = "0727-360B-API") -> Generator[Dict[str, Any], None, None]:
        """Send a single message and get streaming response."""
        messages = [{"role": "user", "content": message}]
        return self.send_messages(messages, chat_id, model)
    
    def send_messages(self, messages: list, chat_id: Optional[str] = None, model: str = "0727-360B-API", reasoning: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Send multiple messages and get streaming response with retry logic."""
        
        # Generate chat_id if not provided
        if not chat_id:
            chat_id = str(uuid.uuid4())
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        payload = {
            "stream": True,
            "model": model,
            "messages": messages,
            "params": {
                "include_reasoning": True,
                "reasoning": {
                    "effort": "high"
                },
                "thinking": {
                    "type": "enabled"
                },
            },
            "include_reasoning": True,
            "thinking": {
                "type": "enabled"
            },
            "reasoning": {
                "effort": "high"
            },
            "tool_servers": [],
            "features": {
                "image_generation": False,
                "code_interpreter": False,
                "web_search": False,
                "auto_web_search": False,
                "enable_thinking": reasoning,
                "flags": [],
            },
            "chat_id": chat_id,
            "id": request_id
        }
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Sending GLM API request (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response = self.session.post(
                    self.base_url,
                    json=payload,
                    stream=True,
                    headers={"referrer": f"https://chat.z.ai/c/{chat_id}"},
                    timeout=self.timeout
                )
                
                # Check for rate limiting before raising for status
                if self._is_rate_limited(response):
                    retry_after = self._get_retry_after(response)
                    logger.warning(f"Rate limited (HTTP {response.status_code}). Retry after: {retry_after}s")
                    
                    if attempt < self.max_retries:
                        # Use retry-after if provided, otherwise use exponential backoff
                        if retry_after:
                            delay = min(retry_after, 60)  # Cap at 60 seconds
                        else:
                            delay = self._calculate_backoff_delay(attempt)
                        
                        logger.info(f"Waiting {delay:.2f}s before retry {attempt + 2}")
                        time.sleep(delay)
                        continue
                    else:
                        raise RateLimitError(
                            f"Rate limited after {self.max_retries + 1} attempts. HTTP {response.status_code}",
                            retry_after
                        )
                
                response.raise_for_status()
                
                # Track response metrics for validation
                total_chunks = 0
                valid_chunks = 0
                json_errors = 0
                
                # Process streaming response
                for line in response.iter_lines():
                    line = line.decode('utf-8')
                    if line.strip():
                        total_chunks += 1
                        
                        # Remove 'data: ' prefix if present
                        if line.startswith('data: '):
                            line = line[6:]
                        
                        # Skip empty lines after stripping
                        if not line.strip():
                            continue
                        
                        # Handle special SSE messages
                        if line.strip() == '[DONE]':
                            logger.debug(f"Stream completed. Total chunks: {total_chunks}, Valid: {valid_chunks}, JSON errors: {json_errors}")
                            break
                        
                        try:
                            # Parse JSON response
                            data = json.loads(line)
                            
                            # Validate response structure
                            if not self._validate_streaming_response(data):
                                logger.warning(f"Invalid response structure: {data}")
                                continue
                            
                            valid_chunks += 1
                            
                            # Extract relevant information
                            if data.get("type") == "chat:completion" and "data" in data:
                                completion_data = data["data"]
                                content = completion_data.get("delta_content", "") or completion_data.get("edit_content", "")
                                
                                # If this is edit_content, remove thinking details section
                                if completion_data.get("edit_content") and "</details>" in content:
                                    last_details_end = content.rfind("</details>")
                                    if last_details_end != -1:
                                        content = content[last_details_end + len("</details>"):].lstrip()
                                
                                phase = completion_data.get("phase", "unknown")
                                
                                result = {
                                    "content": content,
                                    "phase": phase,
                                    "type": data.get("type"),
                                    "raw_data": data
                                }
                                
                                yield result
                                
                        except json.JSONDecodeError as e:
                            json_errors += 1
                            logger.warning(f"JSON decode error (chunk {total_chunks}): {e}")
                            logger.debug(f"Problematic line: '{line[:200]}{'...' if len(line) > 200 else ''}'")
                            
                            # If too many JSON errors, this might indicate a problem
                            if json_errors > 10 and json_errors / total_chunks > 0.5:
                                logger.error(f"High JSON error rate: {json_errors}/{total_chunks}. Possible response corruption.")
                            continue
                
                # If we get here, the request succeeded
                logger.info(f"GLM API request completed successfully on attempt {attempt + 1}")
                return
                
            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response and self._is_rate_limited(e.response):
                    # This should have been caught above, but handle it here too
                    retry_after = self._get_retry_after(e.response)
                    logger.warning(f"HTTP error indicates rate limiting: {e}")
                    
                    if attempt < self.max_retries:
                        delay = retry_after if retry_after else self._calculate_backoff_delay(attempt)
                        logger.info(f"Retrying after {delay:.2f}s due to rate limiting")
                        time.sleep(delay)
                        continue
                    else:
                        raise RateLimitError(f"Rate limited after {self.max_retries + 1} attempts: {e}", retry_after)
                else:
                    # Non-rate-limiting HTTP error, don't retry
                    logger.error(f"HTTP error (non-retryable): {e}")
                    raise Exception(f"GLM API HTTP error: {e}")
                    
            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Request timeout on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries:
                    delay = self._calculate_backoff_delay(attempt, base_delay=2.0)
                    logger.info(f"Retrying after {delay:.2f}s due to timeout")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"GLM API request timed out after {self.max_retries + 1} attempts and {self.timeout} seconds per attempt: {e}")
                    
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries:
                    delay = self._calculate_backoff_delay(attempt, base_delay=1.5)
                    logger.info(f"Retrying after {delay:.2f}s due to connection error")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"GLM API connection failed after {self.max_retries + 1} attempts: {e}")
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.error(f"Request exception on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Retrying after {delay:.2f}s due to request exception")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"GLM API request failed after {self.max_retries + 1} attempts: {e}")
        
        # If we get here, all retries failed
        if last_exception:
            raise Exception(f"GLM API request failed after {self.max_retries + 1} attempts. Last error: {last_exception}")
        else:
            raise Exception(f"GLM API request failed after {self.max_retries + 1} attempts with unknown error")
    
    def send_message_text_only(self, message: str, chat_id: Optional[str] = None, model: str = "0727-360B-API") -> Generator[str, None, None]:
        """Send a message and get only text content (simplified)."""
        for response in self.send_message(message, chat_id, model):
            if response.get("content"):
                content = response["content"]
                yield content
    
    def chat_interactive(self, chat_id: Optional[str] = None):
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