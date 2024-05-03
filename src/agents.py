import json
from typing import Any, Dict, List, Optional, Callable
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
# Import other tool classes and functions here

class Agent(BaseModel):
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types
        exclude = {"client", "tool_objects"}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    goal: str
    tools: List[str] = []
    dependencies: Optional[List[str]] = None
    verbose: bool = False 
    model: str = "adrienbrault/nous-hermes2pro:Q4_0"  # default agent model
    max_iter: int = 25
    max_rpm: Optional[int] = None
    max_execution_time: Optional[int] = None
    cache: bool = True
    step_callback: Optional[Callable] = None
    persona: Optional[str] = None
    allow_delegation: bool = False
    input_tasks: List = []
    output_tasks: List = []
    interactions: List[Dict] = []
    client: str = "ollama"
    tool_objects: Dict[str, Any] = {}

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.client:
            raise ValueError("Client must be specified.")
        self.client = CLIENTS.ollama
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

    def execute(self, context: Optional[str] = None) -> str:
        messages = []
        if self.persona and self.verbose:
            messages.append({"role": "system", "content": f"Background: {self.persona}"})
        messages.append({"role": "system", "content": f"You are a {self.role} with the goal: {self.goal}."})
        # Check if the agent has available tools
        # Check if the agent has available tools
        if self.tool_objects:
            #print(self.tool_objects)
            # Serialize the tool objects to JSON schema
            tool_schemas = []
            for tool_name, tool_object in self.tool_objects.items():
                if isinstance(tool_object, BaseTool):
                    tool_schema = {
                        "name": tool_object.name,
                        "description": tool_object.description,
                        "parameters": tool_object.args_schema.schema()
                    }
                    tool_schemas.append(tool_schema)
                elif isinstance(tool_object, StructuredTool):
                    tool_schema = {
                        "name": tool_object.name,
                        "description": tool_object.description,
                        "parameters": tool_object.args_schema.schema()
                    }
                    tool_schemas.append(tool_schema)

            # Append the tool schemas to the system prompt within <tools></tools> tags
            system_prompt = "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions"
            system_prompt += f"Here are the available tools:\n<tools>\n{json.dumps(tool_schemas, indent=2)}\n</tools>"
            system_prompt += """
Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"}
Please use <scratchpad></scratchpad> XML tags to record your reasoning and planning before you call the functions as follows:
<scratchpad>
{step-by-step reasoning}
</scatchpad>
For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>
"""
            messages.append({"role": "system", "content": system_prompt})

        

        try:
            depth = 0
            messages.append({"role": "user", "content": f"Your task is to {self.goal}."})
            if context:
                messages.append({"role": "assistant", "content": f"Context:\n{context}"})
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            def recursive_loop(prompt, completion, depth):
                result = completion.choices[0].message.content
                print(result)

                # Process the agent's response and extract tool calls
                if self.tool_objects:
                    validation, tool_calls, error_message = validate_and_extract_tool_calls(result)

                    if validation:
                        inference_logger.info(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")

                        # Execute the tool calls
                        tool_message = f"Agent iteration {depth} to assist with user query: {self.goal}\n"
                        if tool_calls:
                            for tool_call in tool_calls:
                                tool_name = tool_call.get("name")
                                tool_object = self.tool_objects.get(tool_name)
                                if tool_object:
                                    try:
                                        if isinstance(tool_object, BaseTool):
                                            # Extract the arguments from the tool call
                                            tool_args = tool_call.get("arguments", {})
                                            # Call the tool's _run method with the provided arguments
                                            tool_result = tool_object._run(**tool_args)
                                        else:
                                            # Call the function-based tool using execute_function_call
                                            tool_result = self.execute_function_call(tool_call)
                                        
                                        # Append the tool result to the tool message
                                        tool_message += f"<tool_response>\n{tool_result}\n</tool_response>\n"
                                        inference_logger.info(f"Here's the response from the tool call: {tool_name}\n{tool_result}")
                                    except Exception as e:
                                        error_message = f"Error executing tool '{tool_name}': {str(e)}"
                                        tool_message += f"<tool_response>\nThere was an error when executing the tool: {tool_name}\nHere's the error traceback: {error_message}\nPlease call this tool again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                                else:
                                    error_message = f"Tool '{tool_name}' not found."
                                    tool_message += f"<tool_response>\nThere was an error finding the tool: {tool_name}\nHere's the error message: {error_message}\nPlease call a valid tool within XML tags <tool_call></tool_call>\n</tool_response>\n"
                            prompt.append({"role": "user", "content": tool_message})

                            depth += 1
                            if depth >= self.max_iter:
                                print(f"Maximum recursion depth reached ({self.max_iter}). Stopping recursion.")
                                return

                            completion = self.client.chat.completions.create(
                                model=self.model,
                                messages=prompt,
                            )
                            recursive_loop(prompt, completion, depth)
                        elif error_message:
                            inference_logger.info(f"Assistant Message:\n{result}")
                            tool_message += f"<tool_response>\nThere was an error parsing tool calls\n Here's the error stack trace: {error_message}\nPlease call the tool again with correct syntax<tool_response>"
                            prompt.append({"role": "user", "content": tool_message})

                            depth += 1
                            if depth >= self.max_iter:
                                print(f"Maximum recursion depth reached ({self.max_iter}). Stopping recursion.")
                                return

                            completion = self.client.chat.completions.create(
                                model=self.model,
                                messages=prompt,
                            )
                            recursive_loop(prompt, completion, depth)
                        else:
                            inference_logger.info(f"Assistant Message:\n{result}")
                    else:
                        inference_logger.info(error_message)
                    
                           # Log the interaction
                    self.log_interaction(messages, result)
                    return completion
            result = recursive_loop(messages, completion, depth)
            result = completion.choices[0].message.content

        except Exception as e:
            inference_logger.error(f"Exception occurred: {e}")
            raise e

        return result
    
    def execute_function_call(self, tool_call):
        function_name = tool_call.get("name")
        print(function_name)
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
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })