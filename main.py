import os
import re
import json
import uuid
import argparse
import networkx as nx
from typing import Any, Dict, List, Optional
from openai import OpenAI
from datetime import datetime

from src.rag_tools import *
from src.prompter import PromptManager
from src.resources import Resource
from src.agents import Agent
from src.clients import CLIENTS
from src.utils import get_tool_names
from src.tools import get_openai_tools

import logfire

class AgentOrchestrator:
    def __init__(self, agents: List['Agent'], resources: List['Resource'], verbose: bool = False, log_file: str = "orchestrator_log.json"):
        self.id = str(uuid.uuid4())
        self.agents = agents
        self.resources = resources
        self.verbose = verbose
        self.log_file = log_file
        self.log_data = []
        self.llama_logs = []

    def run(self, query: str) -> str:
        tools = get_openai_tools()
        tools_dict = {tool["function"]["name"]: tool for tool in tools}

        mermaid_graph, agents_metadata = self.load_or_generate_graph(query, self.agents, tools, self.resources)
        print(mermaid_graph)

        G = nx.DiGraph()
        for agent_data in agents_metadata:
            agent_role = agent_data["role"]
            G.add_node(agent_role, **agent_data)

        execution_order = list(nx.topological_sort(G))

        context = ""
        for agent_role in execution_order:
            agent_data = G.nodes[agent_role]
            agent = Agent(**agent_data)

            if agent.verbose:
                print(f"Starting Agent: {agent.role}")

            output = agent.execute(context=context)

            if agent.verbose:
                print(f"Agent output:\n{output}\n")

            context += f"Agent: {agent.role}\nGoal: {agent.goal}\nOutput:\n{output}\n\n"

            self.llama_logs.extend(agent.interactions)

        self.save_logs()
        self.save_llama_logs()

        return context

    def load_or_generate_graph(self, query, agents, tools, resources):
        mermaid_graph_file = "mermaid_graph.txt"
        agent_metadata_file = "agent_metadata.json"

        if os.path.exists(mermaid_graph_file) and os.path.exists(agent_metadata_file):
            with open(mermaid_graph_file, "r") as file:
                mermaid_graph = file.read()
            with open(agent_metadata_file, "r") as file:
                agents_metadata = json.load(file)
        else:
            mermaid_graph = self.agent_dispatcher(query, agents, tools, resources)
            agents_metadata = self.extract_agents_from_mermaid(mermaid_graph)

            with open(mermaid_graph_file, "w") as file:
                file.write(mermaid_graph)
            with open(agent_metadata_file, "w") as file:
                json.dump(agents_metadata, file, indent=2)

        return mermaid_graph, agents_metadata

    def agent_dispatcher(self, query, agents, tools, resources):
        chat = [{"role": "user", "content": query}]
        prompter = PromptManager()
        sys_prompt = prompter.generate_prompt(tools, agents, resources)

        client = CLIENTS.anthropic
        response = client.messages.create(
            model="claude-3-opus-20240229",
            system=sys_prompt,
            max_tokens=1000,
            temperature=0.5,
            messages=chat,
        )
        print(response)
        completion = response.content[0].text
        return completion

    def extract_agents_from_mermaid(self, mermaid_graph):
        graph_content = re.search(r'<graph>(.*?)</graph>', mermaid_graph, re.DOTALL).group(1)
        metadata_content = re.search(r'<agents>(.*?)</agents>', mermaid_graph, re.DOTALL).group(1)

        dependency_pattern = r'(\w+) --> (\w+)'

        agents_metadata = json.loads(metadata_content)
        dependencies = []

        for match in re.finditer(dependency_pattern, graph_content):
            source = match.group(1)
            target = match.group(2)
            dependencies.append((source, target))

        #for agent in agents_metadata:
        #    agent["dependencies"] = [target for source, target in dependencies if source == agent["role"]]

        return agents_metadata

    def save_llama_logs(self):
        with open(("qa_interactions" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"), "w") as file:
            json.dump(self.llama_logs, file, indent=2)

    def save_logs(self):
        with open(self.log_file, "w") as file:
            json.dump(self.log_data, file, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the agent orchestrator with dynamic configurations.")
    parser.add_argument('-q', '--query', type=str, help="user query for agents to assist with", required=True)
    return parser.parse_args()

def mainflow():
    args = parse_args()

    file_path = os.path.join(os.getcwd())
    with open(os.path.join(file_path, "configs/agents.json"), "r") as file:
        agents_data = json.load(file)
    agents = [Agent(**agent_data) for agent_data in agents_data]

    with open(os.path.join(file_path, "configs/resources.json"), "r") as file:
        resources_data = json.load(file)
    resources = [Resource(**resource_data) for resource_data in resources_data]

    orchestrator = AgentOrchestrator(
        agents=agents,
        resources=resources,
        verbose=True,
        log_file="orchestrator_log" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    )

    result = orchestrator.run(args.query)
    print(f"Final output:\n{result}")

if __name__ == "__main__":
    mainflow()