from typing import Any, Dict, List, Optional, Callable
import uuid
from pydantic import BaseModel, Field
from datetime import datetime
from src.clients import CLIENTS
from src.tools import *
from src.rag_tools import *
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
        messages.append({"role": "user", "content": f"Your task is to {self.goal}."})
        if context:
            messages.append({"role": "assistant", "content": f"Context:\n{context}"})

        for tool_name, tool_object in self.tool_objects.items():
            if callable(tool_object):
                # Call the tool and append the result to messages
                result = tool_object()
                messages.append({"role": "system", "content": f"Tool '{tool_name}' result: {result}"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        result = response.choices[0].message.content
        self.log_interaction(messages, result)

        if self.step_callback:
            self.step_callback(self, result)

        return result

    def log_interaction(self, prompt, response):
        self.interactions.append({
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })