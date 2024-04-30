import os
from typing import Any, Callable, List, Optional
from src.agents import Agent
import uuid
import json
from datetime import datetime
from pydantic import BaseModel

import os
import json

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/agents.json")
with open(file_path, "r") as file:
    agents = json.load(file)

agents = [Agent(**agent) for agent in agents]

class Task(BaseModel):
    id: str = None
    instructions: str
    expected_output: str
    agent: Optional[str] = None
    async_execution: bool = False
    context: Optional[List[str]] = None
    output_file: Optional[str] = None
    callback: Optional[Callable] = None
    human_input: bool = False
    tool_name: Optional[str] = None
    input_tasks: Optional[List["Task"]] = None
    output_tasks: Optional[List["Task"]] = None
    context_agent_role: str = None
    prompt_data: Optional[List] = None
    output: Optional[str] = None
    

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.id = str(uuid.uuid4())
        self.agent = self.load_agent(self.agent)
        self.context = self.context or []
        self.prompt_data = []  # List to hold prompt data for logging
        self.output = None

    def execute(self, context: Optional[str] = None) -> str:
        if not self.agent:
            raise Exception("No agent assigned to the task.")

        context_tasks = [task for task in self.context if task.output]
        if context_tasks:
            self.context_agent_role = context_tasks[0].agent.role
            original_context = "\n".join([f"{task.agent.role}: {task.output}" for task in context_tasks])

            if self.tool_name == 'semantic_search':
                query = "\n".join([task.output for task in context_tasks])
                context = query
            else:
                context = original_context

        # Prepare the prompt for logging before execution
        prompt_details = self.prepare_prompt(context)
        self.prompt_data.append(prompt_details)

        # Execute the task with the agent
        result = self.agent.execute_task(self, context)
        self.output = result

        if self.output_file:
            with open(self.output_file, "w") as file:
                file.write(result)

        if self.callback:
            self.callback(self)

        return result

    def prepare_prompt(self, context):
        """ Prepare and return the prompt details for logging """
        prompt = {
            "timestamp": datetime.now().isoformat(),
            "task_id": self.id,
            "instructions": self.instructions,
            "context": context,
            "expected_output": self.expected_output
        }
        return prompt

    def load_agent(self, agent_role: str) -> Optional[Agent]:
        """ Load the agent based on the given role """
        return next(agent for agent in agents if agent.role == agent_role)

