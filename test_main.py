import json
import uuid
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from tiktoken import get_encoding
import pickle
import requests
from bs4 import BeautifulSoup
import PyPDF2
import os
import hashlib
from src.resources import Resources
import argparse
from datetime import datetime


from src.tools import TextReaderTool, WebScraperTool, SemanticAnalysisTool, NERExtractionTool, SemanticFileSearchTool, WikipediaSearchTool
from src import agents
from src.agents import Agent
from src import tasks
from src.tasks import Task

class Resources:
    def __init__(self, resource_type, path, template):
        self.resource_type = resource_type
        self.path = path
        self.template = template

class Agent:
    def __init__(self, role, tools):
        self.role = role
        self.tools = tools
        self.interactions = []  # Simulated interactions log

class Task:
    def __init__(self, instructions, agent, tool_name=None):
        self.instructions = instructions
        self.agent = agent
        self.tool_name = tool_name
        self.id = str(uuid.uuid4())
        self.output = None

    def execute(self, context):
        # Placeholder for task execution logic
        return f"Executed {self.instructions} using {self.agent.role}"

class Squad:
    def __init__(self, agents: List[Agent], tasks: List[Task], resources: List[Resources], verbose: bool = False, log_file: str = "squad_log.json"):
        self.id = str(uuid.uuid4())
        self.agents = agents
        self.tasks = tasks
        self.resources = resources
        self.verbose = verbose
        self.log_file = log_file
        self.log_data = []
        self.llama_logs = []

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> str:
        context = ""
        for task in self.tasks:
            if self.verbose:
                print(f"Starting Task:\n{task.instructions}")

            self.log_data.append({
                "timestamp": datetime.now().isoformat(),
                "type": "input",
                "agent_role": task.agent.role,
                "task_name": task.instructions,
                "task_id": task.id,
                "content": task.instructions
            })

            output = task.execute(context=context)
            task.output = output

            if self.verbose:
                print(f"Task output:\n{output}\n")

            self.log_data.append({
                "timestamp": datetime.now().isoformat(),
                "type": "output",
                "agent_role": task.agent.role,
                "task_name": task.instructions,
                "task_id": task.id,
                "content": output
            })

            context += f"Task:\n{task.instructions}\nOutput:\n{output}\n\n"

        self.save_logs()
        return context

    def save_logs(self):
        with open(self.log_file, "w") as file:
            json.dump(self.log_data, file, indent=2)

def load_configuration(file_path):  # loading a config file alt is full cli?
    with open(file_path, 'r') as file:
        return json.load(file)

def initialize_resources(config): 
    resources = []
    for res in config["resources"]:
        resources.append(Resources(res['type'], res['path'], res['template']))
    return resources

def initialize_agents_and_tasks(config):
    agents = [Agent(**ag) for ag in config['agents']]
    tasks = [Task(**tk) for tk in config['tasks']]
    return agents, tasks

def parse_args():
    parser = argparse.ArgumentParser(description="Run the squad with dynamic configurations.")
    parser.add_argument('-c', '--config', type=str, help="Path to configuration JSON file", required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    return parser.parse_args()

def mainflow():
    args = parse_args()
    config = load_configuration(args.config)
    
    resources = initialize_resources(config)
    agents, tasks = initialize_agents_and_tasks(config)
    
    squad = Squad(
        agents=agents,
        tasks=tasks,
        resources=resources,
        verbose=args.verbose,
        log_file="squad_goals_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    )

    result = squad.run()
    print(f"Final output:\n{result}")

if __name__ == "__main__":
    mainflow()
