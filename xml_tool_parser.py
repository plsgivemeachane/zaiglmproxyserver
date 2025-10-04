import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ToolCommand:
    """Represents a parsed tool command with its name and parameters."""
    name: str
    parameters: Dict[str, str]
    raw_xml: str


class XMLToolParser:
    """Parser for extracting XML-formatted tool commands from agent outputs."""
    
    def __init__(self):
        # Define supported tools based on the documentation
        self.supported_tools = {
            'thoughts',
            'attempt_completion'
        }
    
    def extract_xml_blocks(self, text: str) -> List[str]:
        """Extract all XML blocks from the given text."""
        # Pattern to match XML tags with content
        pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)>.*?</\1>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        # Extract full XML blocks
        xml_blocks = []
        for match in matches:
            tag_pattern = f'<{match}>.*?</{match}>'
            full_match = re.search(tag_pattern, text, re.DOTALL)
            if full_match:
                xml_blocks.append(full_match.group(0))
        
        return xml_blocks
    
    def parse_xml_block(self, xml_block: str) -> Optional[ToolCommand]:
        """Parse a single XML block into a ToolCommand."""
        try:
            # Clean up the XML block
            xml_block = xml_block.strip()
            
            # Parse with ElementTree
            root = ET.fromstring(xml_block)
            tool_name = root.tag
            
            # Only process supported tools
            if tool_name not in self.supported_tools:
                return None
            
            # Extract parameters
            parameters = {}
            for child in root:
                parameters[child.tag] = child.text or ''
            
            return ToolCommand(
                name=tool_name,
                parameters=parameters,
                raw_xml=xml_block
            )
            
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error parsing XML: {e}")
            return None
    
    def parse_agent_output(self, agent_output: str) -> List[ToolCommand]:
        """Parse agent output and extract all valid tool commands."""
        xml_blocks = self.extract_xml_blocks(agent_output)
        commands = []
        
        for block in xml_blocks:
            command = self.parse_xml_block(block)
            if command:
                commands.append(command)
        
        return commands


class ToolExecutor:
    """Central executor for running parsed tool commands."""
    
    def __init__(self):
        self.parser = XMLToolParser()
        self.execution_history = []
    
    def execute_thoughts(self, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Execute thoughts command - typically just logs the thought process."""
        thought_content = parameters.get('content', '')
        
        result = {
            'command': 'thoughts',
            'status': 'executed',
            'content': thought_content,
            'timestamp': self._get_timestamp()
        }
        
        print(f"[THOUGHTS] {thought_content}")
        return result
    
    def execute_attempt_completion(self, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Execute attempt_completion command."""
        result_content = parameters.get('result', '')
        
        result = {
            'command': 'attempt_completion',
            'status': 'executed',
            'result': result_content,
            'timestamp': self._get_timestamp()
        }
        
        print(f"[COMPLETION] {result_content}")
        return result
    
    def execute_command(self, command: ToolCommand) -> Dict[str, Any]:
        """Execute a single tool command."""
        if command.name == 'thoughts':
            return self.execute_thoughts(command.parameters)
        elif command.name == 'attempt_completion':
            return self.execute_attempt_completion(command.parameters)
        else:
            return {
                'command': command.name,
                'status': 'unsupported',
                'error': f'Unsupported command: {command.name}'
            }
    
    def process_agent_output(self, agent_output: str) -> List[Dict[str, Any]]:
        """Process complete agent output and execute all found commands."""
        commands = self.parser.parse_agent_output(agent_output)
        results = []
        
        for command in commands:
            result = self.execute_command(command)
            results.append(result)
            self.execution_history.append(result)
        
        return results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the history of all executed commands."""
        return self.execution_history.copy()


# Example usage and testing
if __name__ == "__main__":
    # Test the parser with sample agent output
    sample_output = """
    Here's my analysis of the problem:
    
    <thoughts>
    <content>I think we need to consider multiple approaches here. Perhaps we should start with the simplest solution first.</content>
    </thoughts>
    
    After reviewing the code, I believe the task is complete.
    
    <attempt_completion>
    <result>I've successfully implemented the XML parser class that can extract and execute tool commands from agent outputs.</result>
    </attempt_completion>
    """
    
    # Create executor and process the output
    executor = ToolExecutor()
    results = executor.process_agent_output(sample_output)
    
    print("\nExecution Results:")
    for result in results:
        print(f"- {result['command']}: {result['status']}")
    
    print("\nExecution History:")
    for entry in executor.get_execution_history():
        print(f"- {entry['timestamp']}: {entry['command']} - {entry['status']}")