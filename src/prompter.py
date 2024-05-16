import datetime
from pydantic import BaseModel
from typing import Dict
from src.schema import Agent
from src.utils import (
    get_fewshot_examples
)
import yaml
import json
import os

class PromptSchema(BaseModel):
    Role: str
    Objective: str
    Agents: str
    Tools: str
    #Resources: str
    #Examples: str
    Schema: str
    Instructions: str 

class PromptManager:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
    def format_yaml_prompt(self, prompt_schema: PromptSchema, variables: Dict) -> str:
        formatted_prompt = ""
        for field, value in prompt_schema.dict().items():
            if field == "Examples" and variables.get("examples") is None:
                continue
            formatted_value = value.format(**variables)
            if field == "Instructions":
                formatted_prompt += f"{formatted_value}"
            else:
                formatted_value = formatted_value.replace("\n", " ")
                formatted_prompt += f"{formatted_value}"
        return formatted_prompt

    def read_yaml_file(self, file_path: str) -> PromptSchema:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        
        prompt_schema = PromptSchema(
            Role=yaml_content.get('Role', ''),
            Objective=yaml_content.get('Objective', ''),
            Agents=yaml_content.get('Agents', ''),
            Tools=yaml_content.get('Tools', ''),
            #Resources=yaml_content.get('Resources', ''),
            #Examples=yaml_content.get('Examples', ''),
            Schema=yaml_content.get('Schema', ''),
            Instructions=yaml_content.get('Instructions', ''),
        )
        return prompt_schema
    
    def generate_prompt(self, tools, agents, resources, one_shot=False):
        prompt_path = os.path.join(self.script_dir, '../configs', 'sys_prompt.yaml')
        prompt_schema = self.read_yaml_file(prompt_path)

        schema_json = json.loads(Agent.schema_json())
        #schema = schema_json.get("properties", {})

        variables = {
            "date": datetime.date.today(),
            "agents": agents,
            "tools": tools,
            "resources": None,
            #"examples": examples,
            "schema": schema_json
        }
        sys_prompt = self.format_yaml_prompt(prompt_schema, variables)
        #print(sys_prompt)

        prompt = [
                {'role': 'system', 'content': sys_prompt}
            ]

        if one_shot:
            #examples = get_fewshot_examples(num_fewshot)
            with open(os.path.join(self.script_dir, '../configs', 'example.txt'), 'r') as file:
                examples = file.read()

            prompt.extend([
                {"role": "user", "content": "Perform fundamental analysis of NVDA stock and provide portfolio recommendations"},
                {"role": "assistant", "content": examples}
            ])
        print(prompt)
        return sys_prompt