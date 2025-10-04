#!/usr/bin/env python3
"""
Example script demonstrating how to use the ReasoningModule independently
for both streaming and non-streaming contexts.
"""

from glm_api_client import GLMAPIClient
from reasoning_module import ReasoningModule
import json


def example_standalone_reasoning():
    """Example of using ReasoningModule independently."""
    
    # Initialize GLM client and reasoning module
    glm_client = GLMAPIClient()
    reasoning_module = ReasoningModule(glm_client)
    
    # Example messages
    messages = [
        {"role": "user", "content": "Help me write a Python function to calculate fibonacci numbers"}
    ]
    
    print("=== Standalone Reasoning Example ===")
    
    # Run complete reasoning process
    result = reasoning_module.run_complete_reasoning(messages, model="glm-4-plus")
    
    if result["success"]:
        print(f"Reasoning completed in {result['iterations']} iterations")
        print("Final message:")
        print(json.dumps(result["final_message"], indent=2))
    else:
        print(f"Reasoning failed: {result.get('error', 'Unknown error')}")
        print(f"Completed {result['iterations']} iterations")
    
    return result


def example_step_by_step_reasoning():
    """Example of using ReasoningModule step by step for more control."""
    
    # Initialize GLM client and reasoning module
    glm_client = GLMAPIClient()
    reasoning_module = ReasoningModule(glm_client)
    
    # Example messages
    messages = [
        {"role": "user", "content": "Explain the concept of recursion in programming"}
    ]
    
    print("\n=== Step-by-Step Reasoning Example ===")
    
    # Initialize reasoning
    reasoning_module.initialize_reasoning(messages)
    
    max_steps = 5
    step = 0
    
    while step < max_steps:
        step += 1
        print(f"\n--- Step {step} ---")
        
        # Perform one reasoning step
        result = reasoning_module.perform_reasoning_step("glm-4-plus")
        
        print(f"Reason token length: {len(result['reason_token'])}")
        print(f"Commands found: {len(result['commands'])}")
        
        if not result["should_continue"]:
            print("Reasoning completed!")
            print("Final message:")
            print(json.dumps(result["final_message"], indent=2))
            break
        
        print("Continuing to next step...")
    
    if step >= max_steps:
        print(f"\nReached maximum steps ({max_steps})")
    
    # Get all reasoning messages
    all_messages = reasoning_module.get_reasoning_messages()
    print(f"\nTotal reasoning messages: {len(all_messages)}")
    
    return reasoning_module


def example_with_custom_settings():
    """Example showing how to customize the reasoning module."""
    
    print("\n=== Custom Settings Example ===")
    
    # Initialize with custom tool instructions file
    glm_client = GLMAPIClient()
    reasoning_module = ReasoningModule(glm_client, tool_instructions_file='tool_use.md')
    
    # Disable debug output
    reasoning_module.set_debug(False)
    
    messages = [
        {"role": "user", "content": "What are the best practices for error handling in Python?"}
    ]
    
    # Run reasoning with custom max iterations
    result = reasoning_module.run_complete_reasoning(messages, model="glm-4-plus", max_iterations=3)
    
    print(f"Reasoning result: {'Success' if result['success'] else 'Failed'}")
    print(f"Iterations: {result['iterations']}")
    
    return result


if __name__ == "__main__":
    print("ReasoningModule Examples")
    print("=" * 50)
    
    try:
        # Example 1: Complete reasoning process
        example_standalone_reasoning()
        
        # Example 2: Step-by-step control
        example_step_by_step_reasoning()
        
        # Example 3: Custom settings
        example_with_custom_settings()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your GLM API credentials properly.")