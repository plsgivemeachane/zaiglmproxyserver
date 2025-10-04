# Reasoning Module Documentation

The `ReasoningModule` is a modular reasoning system that can be used in both streaming and non-streaming contexts. It extracts the reasoning logic from `GLMHyperthinkWrapper` into a reusable component.

## Features

- **Modular Design**: Can be used independently or integrated into existing systems
- **Streaming & Non-Streaming Support**: Works in both contexts
- **Configurable**: Supports custom tool instructions and debug settings
- **Step-by-Step Control**: Allows fine-grained control over reasoning process
- **Automatic Tool Parsing**: Handles XML tool parsing and command execution

## Basic Usage

### Standalone Reasoning

```python
from glm_api_client import GLMAPIClient
from reasoning_module import ReasoningModule

# Initialize
glm_client = GLMAPIClient()
reasoning_module = ReasoningModule(glm_client)

# Prepare messages
messages = [
    {"role": "user", "content": "Help me solve this problem..."}
]

# Run complete reasoning
result = reasoning_module.run_complete_reasoning(messages, model="glm-4-plus")

if result["success"]:
    print(f"Reasoning completed in {result['iterations']} iterations")
    final_message = result["final_message"]
else:
    print(f"Reasoning failed: {result.get('error')}")
```

### Step-by-Step Control

```python
# Initialize reasoning
reasoning_module.initialize_reasoning(messages)

# Perform reasoning steps manually
while True:
    result = reasoning_module.perform_reasoning_step("glm-4-plus")
    
    if not result["should_continue"]:
        print("Reasoning completed!")
        final_message = result["final_message"]
        break
    
    # Process intermediate results if needed
    print(f"Commands found: {len(result['commands'])}")
```

### Integration with Existing Systems

```python
class MyWrapper:
    def __init__(self):
        self.glm_client = GLMAPIClient()
        self.reasoning_module = ReasoningModule(self.glm_client)
    
    def process_with_reasoning(self, messages, model):
        # Use reasoning module
        reasoning_result = self.reasoning_module.run_complete_reasoning(messages, model)
        
        if reasoning_result["success"] and reasoning_result["final_message"]:
            messages.append(reasoning_result["final_message"])
        
        # Continue with your processing...
        return self.glm_client.send_messages(messages, model=model)
```

## Configuration Options

### Custom Tool Instructions

```python
# Use custom tool instructions file
reasoning_module = ReasoningModule(glm_client, tool_instructions_file='my_tools.md')
```

### Debug Settings

```python
# Enable/disable debug output
reasoning_module.set_debug(True)   # Enable debug logging
reasoning_module.set_debug(False)  # Disable debug logging
```

### Max Iterations

```python
# Control maximum reasoning iterations
result = reasoning_module.run_complete_reasoning(
    messages, 
    model="glm-4-plus", 
    max_iterations=5  # Limit to 5 iterations
)
```

## API Reference

### ReasoningModule Class

#### Constructor
```python
ReasoningModule(glm_client: GLMAPIClient, tool_instructions_file: str = 'tool_use.md')
```

#### Methods

- `initialize_reasoning(initial_messages: List[Dict[str, Any]]) -> None`
  - Initialize the reasoning process with initial messages

- `perform_reasoning_step(model: str) -> Dict[str, Any]`
  - Perform a single reasoning step
  - Returns: Dictionary with `reason_token`, `commands`, `should_continue`, `final_message`

- `run_complete_reasoning(initial_messages: List[Dict[str, Any]], model: str, max_iterations: int = 10) -> Dict[str, Any]`
  - Run the complete reasoning process
  - Returns: Dictionary with `success`, `iterations`, `final_message`, `reasoning_messages`, `last_result`

- `get_reasoning_messages() -> List[Dict[str, Any]]`
  - Get the current reasoning messages

- `set_debug(debug: bool)`
  - Enable or disable debug logging

## Return Values

### perform_reasoning_step() Returns
```python
{
    "reason_token": str,           # The reasoning text generated
    "commands": List[Dict],        # Parsed XML commands
    "should_continue": bool,       # Whether to continue reasoning
    "final_message": Dict | None   # Final message if reasoning complete
}
```

### run_complete_reasoning() Returns
```python
{
    "success": bool,                    # Whether reasoning completed successfully
    "iterations": int,                 # Number of iterations performed
    "final_message": Dict | None,      # Final message for next processing
    "reasoning_messages": List[Dict],  # All reasoning messages
    "last_result": Dict | None,       # Last step result
    "error": str | None               # Error message if failed
}
```

## Integration Examples

See `reasoning_example.py` for complete working examples of:
- Standalone reasoning usage
- Step-by-step control
- Custom configuration
- Integration patterns

## Tool Support

The reasoning module automatically handles:
- `thoughts` tool: Continues reasoning process
- `attempt_completion` tool: Finalizes reasoning and returns result

Tool instructions are loaded from the specified markdown file (default: `tool_use.md`).