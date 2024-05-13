import json
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field
from datetime import datetime
from src.clients import CLIENTS
from src import tools
from src.tools import *
from src.rag_tools import *
from src.utils import inference_logger
from src.utils import validate_and_extract_tool_calls
from langchain.tools import StructuredTool, BaseTool

from langchain_core.messages import ToolMessage

## TODO: add default tools such as "get_user_feedback", "get_additional_context", "code_interpreter" etc.

class Agent(BaseModel):
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types
        exclude = {"client", "tool_objects"}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    persona: Optional[str] = None
    goal: str
    tools: List[str] = []
    dependencies: Optional[List[str]] = None
    verbose: bool = False
    model: str = Field(default_factory=lambda: os.getenv('AGENT_MODEL'))  # agent model from environment variable
    max_iter: int = 2
    client: str = Field(default_factory=lambda: os.getenv('AGENT_CLIENT')) 
    tool_objects: Dict[str, Any] = {}
    input_messages: List[Dict] = []
    interactions: List[Dict] = []

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.client:
            raise ValueError("Invalid client specified.")
        self.tool_objects = self.create_tool_objects()

    def create_tool_objects(self) -> Dict[str, Any]:
        tool_objects = {}
        for tool_name in self.tools:
            if tool_name in globals():
                tool_objects[tool_name] = globals()[tool_name]
            else:
                raise ValueError(f"Tool '{tool_name}' not found.")
        return tool_objects

    def execute(self) -> str:
        messages = []
        if self.persona and self.verbose:
            messages.append({"role": "system", "content": f"You are a {self.role} with the persona: {self.persona}."})

          # Check if the agent has available tools
        if self.tool_objects:
            #print(self.tool_objects)
            # Serialize the tool objects to JSON schema
            tool_schemas = []
            for tool_name, tool_object in self.tool_objects.items():
                if isinstance(tool_object, StructuredTool):
                    tool_schema = {
                        "name": tool_object.name,
                        "description": tool_object.description,
                        "parameters": tool_object.args_schema.schema()
                    }
                    tool_schemas.append(tool_schema)

            # Append the tool schemas to the system prompt within <tools></tools> tags
            system_prompt = "You are a function calling AI model."
            system_prompt += f"\nHere are the available tools:\n<tools>\n{json.dumps(tool_schemas, indent=2)}\n</tools>\n"
            system_prompt += """
Please use <scratchpad> XML tags to record your reasoning and planning before you call the functions as follows:
<scratchpad>
{step-by-step reasoning}
</scratchpad>
For each function call return a json object with function name and arguments within <tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>
"""
            messages.append({"role": "system-tool", "content": system_prompt})

        if self.input_messages:
            for input_message in self.input_messages:
                role = input_message["role"]
                content = input_message["content"]
                inference_logger.info(f"Appending input messages from previous agent: {role}")
                messages.append({"role": "system-agent", "content": f"<{role}>\n{content}\n</{role}>"})

        messages.append({"role": "user", "content": f"Your task is to {self.goal}."})

        depth = 0
        while depth < self.max_iter:
            inference_logger.info(f"Running inference with {self.client}")
            result = CLIENTS.chat_completion(
                client=self.client,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": result})
            print(result)

            # Process the agent's response and extract tool calls
            if self.tool_objects:
                validation, tool_calls, error_message = validate_and_extract_tool_calls(result)

                if validation and tool_calls:
                    inference_logger.info(f"Parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")

                    # Execute the tool calls
                    tool_message = f"Sub-agent iteration {depth} to assist with user query: {self.goal}\n"
                    if tool_calls:
                        for tool_call in tool_calls:
                            tool_name = tool_call.get("name")
                            tool_object = self.tool_objects.get(tool_name)
                            if tool_object:
                                try:
                                    tool_args = tool_call.get("arguments", {})
                                    tool_result = tool_object._run(**tool_args) if isinstance(tool_object, BaseTool) else self.execute_function_call(tool_call)
                                    tool_message += f"<tool_response>\n{tool_result}\n</tool_response>\n"
                                    inference_logger.info(f"Response from tool '{tool_name}':\n{tool_result}")
                                except Exception as e:
                                    error_message = f"Error executing tool '{tool_name}': {str(e)}"
                                    tool_message += f"<tool_error>\n{error_message}\n</tool_error>\n"
                            else:
                                error_message = f"Tool '{tool_name}' not found."
                                tool_message += f"<tool_error>\n{error_message}\n</tool_error>\n"
                        #messages.append({"role": "user", "content": tool_message})
                        messages.append({"role": "tool", "content": tool_message})
                        #messages.append(ToolMessage(tool_message, tool_call_id=0))
                    else:
                        inference_logger.info(f"No tool calls found in the agent's response.")
                        break
                elif error_message:
                    inference_logger.info(f"Error parsing tool calls: {error_message}")
                    tool_message = f"<tool_error>\n{error_message}\n</tool_error>\n"
                    #messages.append({"role": "user", "content": tool_message})
                    messages.append({"role": "tool", "content": tool_message})
                    #messages.append(ToolMessage(tool_message, tool_call_id=0))

                depth += 1
            else:
                break

        # Log the final interaction
        self.log_interaction(messages, result)
        return result

    def execute_function_call(self, tool_call):
        function_name = tool_call.get("name")
        function_to_call = getattr(tools, function_name, None)
        function_args = tool_call.get("arguments", {})

        if function_to_call:
            inference_logger.info(f"Invoking function call {function_name} ...")
            function_response = function_to_call(**function_args)
            results_dict = f'{{"name": "{function_name}", "content": {json.dumps(function_response)}}}'
            return results_dict
        else:
            raise ValueError(f"Function '{function_name}' not found.")

    def log_interaction(self, prompt, response):
        self.interactions.append({
            "role": self.role,
            "messages": prompt,
            "response": response,
            "agent_messages": self.input_messages,
            "tools": self.tools,
            "timestamp": datetime.now().isoformat()
        })