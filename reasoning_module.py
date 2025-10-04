import time
from typing import Dict, Any, List, Optional
from xml_tool_parser import XMLToolParser
from glm_api_client import GLMAPIClient


class ReasoningModule:
    """A modular reasoning system that can be used in both streaming and non-streaming contexts."""
    
    def __init__(self, glm_client: GLMAPIClient, tool_instructions_file: str = 'tool_use.md'):
        self.glm_client = glm_client
        self.tool_instructions_file = tool_instructions_file
        self.reasoning_messages = []
        self.reasoning_length = 0
        self.debug = True
    
    def _load_tool_instructions(self) -> str:
        """Load tool use instructions from markdown file."""
        try:
            with open(self.tool_instructions_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "No tool instructions found."
    
    def _log(self, message: str):
        """Log debug messages if debug mode is enabled."""
        if self.debug:
            print(message)
    
    def initialize_reasoning(self, initial_messages: List[Dict[str, Any]]) -> None:
        """Initialize the reasoning process with initial messages."""
        self.reasoning_messages = initial_messages.copy()
        tool_instructions = self._load_tool_instructions()
        
        # Add tool use instruction to guide AI reasoning
        self.reasoning_messages.append({
            "role": "user",
            "content": tool_instructions
        })
        
        self.reasoning_length = 0
        self._log("\n=== Reasoning Module Initialized ===")
    
    def perform_reasoning_step(self, model: str) -> Dict[str, Any]:
        """Perform a single reasoning step and return the result."""
        self.reasoning_length += 1
        reason_token = ""
        
        self._log("\nPerforming reasoning step...")
        
        # Get response from GLM client
        for reasoning in self.glm_client.send_messages(self.reasoning_messages, model=model, reasoning=False):
            content = reasoning.get("content", "")
            phase = reasoning.get("phase", "")
            
            if phase == "answer" and content:
                self._log(f"Content: {content[:50]}...")
                reason_token += content
        
        # Append assistant message
        self._log("\nAppending assistant message...")
        self.reasoning_messages.append({
            "role": "assistant",
            "content": reason_token
        })
        
        # Parse token for tools
        self._log("\nParsing token for tools...")
        parser = XMLToolParser()
        xml_blocks = parser.extract_xml_blocks(reason_token)
        commands = []
        for block in xml_blocks:
            cmd = parser.parse_xml_block(block)
            if cmd:
                commands.append({
                    'tool_name': cmd.name,
                    'parameters': cmd.parameters,
                    'raw_xml': cmd.raw_xml
                })
        
        # Process parsed commands
        self._log("\nProcessing parsed commands...")
        
        result = {
            "reason_token": reason_token,
            "commands": commands,
            "should_continue": True,
            "final_message": None
        }
        
        # Handle case when no tool use is detected
        if ["thoughts", "attempt_completion"] not in [cmd['tool_name'] for cmd in commands]:
            self._log("\nNo tool commands found - adding nudge prompt...")
            # Add a nudge message to encourage proper tool usage
            nudge_message = (
                "IMPORTANT: You must use the available tools to structure your response properly. "
                "Please use <thoughts>your reasoning here</thoughts> to continue thinking, or "
                "<attempt_completion>your final answer</attempt_completion> when you're ready to provide the final response. "
                "Do not provide responses without using these tools."
            )
            self.reasoning_messages.append({
                "role": "user",
                "content": nudge_message
            })
            result["should_continue"] = True  # Continue reasoning with the nudge
            return result
        
        for i, cmd in enumerate(commands, 1):
            self._log(f"Command {i}: {cmd['tool_name']}")
            
            if cmd['tool_name'] == "attempt_completion":
                self._log("\nFound attempt_completion - finalizing...")
                result["should_continue"] = False
                reason_token += "\n" + str(cmd.get('parameters', {}))
                result["final_message"] = {
                    "role": "system",
                    "content": f"<thinking>\n{reason_token}\n</thinking>\n\nBased on the reasoning above, here is my response:"
                }
                break
            elif cmd['tool_name'] == "thoughts":
                self._log("\nFound thoughts - continuing reasoning...")
                self.reasoning_messages.append({
                    "role": "user",
                    "content": str(cmd.get('parameters', { reason_token }))
                })
                break
        
        return result
    
    def run_complete_reasoning(self, initial_messages: List[Dict[str, Any]], model: str, max_iterations: int = 10) -> Dict[str, Any]:
        """Run the complete reasoning process until completion or max iterations."""
        self.initialize_reasoning(initial_messages)
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            self._log(f"\n--- Reasoning Iteration {iteration} ---")
            
            result = self.perform_reasoning_step(model)
            
            if not result["should_continue"]:
                self._log("\n=== Reasoning Phase Complete ===")
                return {
                    "success": True,
                    "iterations": iteration,
                    "final_message": result["final_message"],
                    "reasoning_messages": self.reasoning_messages,
                    "last_result": result
                }
            
            # Optional delay for testing
            # time.sleep(0.1)
        
        self._log(f"\n=== Reasoning stopped after {max_iterations} iterations ===")
        return {
            "success": False,
            "iterations": iteration,
            "final_message": None,
            "reasoning_messages": self.reasoning_messages,
            "last_result": None,
            "error": "Max iterations reached"
        }
    
    def get_reasoning_messages(self) -> List[Dict[str, Any]]:
        """Get the current reasoning messages."""
        return self.reasoning_messages.copy()
    
    def set_debug(self, debug: bool):
        """Enable or disable debug logging."""
        self.debug = debug