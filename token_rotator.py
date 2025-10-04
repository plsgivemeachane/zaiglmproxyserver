import json
import time
import threading
import random
import os
import logging
from typing import Dict, Any, List, Optional


class TokenRotator:
    """Manages rotation of GLM API tokens for load balancing and rate limiting."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, tokens: Optional[List[str]] = None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TokenRotator, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, tokens: Optional[List[str]] = None):
        """Initialize the TokenRotator singleton."""
        if self._initialized:
            return
            
        self.tokens = tokens if tokens else []
        self.current_index = 0
        self.rotation_lock = threading.Lock()
        self.token_stats = {
            token: {
                'requests': 0,
                'errors': 0,
                'last_used': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_cost': 0.0
            } for token in self.tokens
        }
        self._initialized = True
        
    def get_next_token(self) -> str | None:
        """Get the next token in rotation."""
        if not self.tokens:
            return None
            
        with self.rotation_lock:
            token = self.tokens[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.tokens)
            self.token_stats[token]['last_used'] = time.time()
            return token
    
    def get_random_token(self) -> str | None:
        """Get a random token from the pool."""
        if not self.tokens:
            return None
        return random.choice(self.tokens)
    
    def record_request(self, token: str | None):
        """Record a successful request for a token."""
        if token and token in self.token_stats:
            with self.rotation_lock:
                self.token_stats[token]['requests'] += 1
    
    def record_error(self, token: str | None):
        """Record an error for a token."""
        if token and token in self.token_stats:
            with self.rotation_lock:
                self.token_stats[token]['errors'] += 1
    
    def record_token_usage(self, token: str | None, input_tokens: int, output_tokens: int, cost: float = 0.0):
        """Record token usage and cost for a token."""
        if token and token in self.token_stats:
            with self.rotation_lock:
                self.token_stats[token]['input_tokens'] += input_tokens
                self.token_stats[token]['output_tokens'] += output_tokens
                self.token_stats[token]['total_cost'] += cost
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all tokens."""
        with self.rotation_lock:
            # Calculate overall statistics
            total_input_tokens = sum(stats['input_tokens'] for stats in self.token_stats.values())
            total_output_tokens = sum(stats['output_tokens'] for stats in self.token_stats.values())
            total_tokens_count = total_input_tokens + total_output_tokens
            total_spending = sum(stats['total_cost'] for stats in self.token_stats.values())
            
            return {
                'total_tokens': len(self.tokens),
                'overall_stats': {
                    'total_input_tokens': total_input_tokens,
                    'total_output_tokens': total_output_tokens,
                    'total_tokens_count': total_tokens_count,
                    'total_spending': total_spending,
                    'average_cost_per_request': total_spending / sum(stats['requests'] for stats in self.token_stats.values()) if any(stats['requests'] for stats in self.token_stats.values()) else 0.0
                },
                'token_stats': dict(self.token_stats)
            }
    
    def add_token(self, token: str):
        """Add a new token to the rotation."""
        if token and token not in self.tokens:
            with self.rotation_lock:
                self.tokens.append(token)
                self.token_stats[token] = {
                    'requests': 0,
                    'errors': 0,
                    'last_used': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_cost': 0.0
                }
    
    def remove_token(self, token: str):
        """Remove a token from the rotation."""
        if token in self.tokens:
            with self.rotation_lock:
                self.tokens.remove(token)
                if token in self.token_stats:
                    del self.token_stats[token]
                # Reset index if needed
                if self.current_index >= len(self.tokens) and self.tokens:
                    self.current_index = 0
    
    def initialize_tokens(self, tokens: List[str]):
        """Initialize or reinitialize tokens for the rotator."""
        with self.rotation_lock:
            self.tokens = tokens if tokens else []
            self.current_index = 0
            self.token_stats = {
                token: {
                    'requests': 0,
                    'errors': 0,
                    'last_used': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_cost': 0.0
                } for token in self.tokens
            }


def load_tokens_from_file(filename: str = 'token.json') -> List[str]:
    """Load tokens from JSON file."""
    logger = logging.getLogger(__name__)
    
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


def get_token_rotator() -> TokenRotator:
    """Get the singleton TokenRotator instance."""
    return TokenRotator()


def initialize_token_rotator_from_config() -> TokenRotator:
    """Initialize the token rotator with tokens from file and environment."""
    logger = logging.getLogger(__name__)
    
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
    
    # Get or create the singleton instance
    rotator = TokenRotator()
    rotator.initialize_tokens(tokens)
    
    if tokens:
        logger.info(f"Token rotation initialized with {len(tokens)} tokens")
    else:
        logger.error("No tokens available for GLM API. Please add tokens to token.json or set GLM_API_TOKEN environment variable.")
    
    return rotator