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
#from src.tasks import Task
from src.clients import CLIENTS
from src.utils import get_tool_names
from src.tools import get_openai_tools

import logfire

#logfire.install_auto_tracing(modules=['Agent'])
#logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))


#class Squad:
#    def __init__(self, agents: List['Agent'], tasks: List['Task'], resources: List['Resource'], verbose: bool = False, log_file: str = "squad_log.json"):
#        self.id = str(uuid.uuid4())
#        self.agents = agents
#        self.tasks = tasks
#        self.resources = resources
#        self.verbose = verbose
#        self.log_file = log_file
#        self.log_data = []
#        self.llama_logs = []  # Attribute to store LLM interactions

#    def run(self, inputs: Optional[Dict[str, Any]] = None) -> str:
#        context = ""
#        for task in self.tasks:
#            if self.verbose:
#                print(f"Starting Task:\n{task.instructions}")

#            # Log the input to the task
#            self.log_data.append({
#                "timestamp": datetime.now().isoformat(),
#                "type": "input",
#                "agent_role": task.agent.role,
#                "task_name": task.instructions,
#                "task_id": task.id,
#                "content": task.instructions
#            })

#            # Execute the task and retrieve the output
#            output = task.execute(context=context)
#            task.output = output

#            if self.verbose:
#                print(f"Task output:\n{output}\n")

#            # Log the output from the task
#            self.log_data.append({
#                "timestamp": datetime.now().isoformat(),
#                "type": "output",
#                "agent_role": task.agent.role,
#                "task_name": task.instructions,
#                "task_id": task.id,
#                "content": output
#            })

#            # Collect LLM interactions from the agent after task execution
#            self.llama_logs.extend(task.agent.interactions)  # Assuming interactions are collected in each agent

#            context += f"Task:\n{task.instructions}\nOutput:\n{output}\n\n"

#            # Additional logic for handling specific tools used by tasks
#            self.handle_tool_logic(task, context)

#        # Writing all logged data and LLM interactions to JSON files
#        self.save_logs()
#        self.save_llama_logs()

#        return context

#    def handle_tool_logic(self, task, context):
#        if task.tool_name in task.agent.tools:
#            tool = task.agent.tools[task.tool_name]
#            if isinstance(tool, TextReaderTool) or isinstance(tool, WebScraperTool) or isinstance(tool, SemanticFileSearchTool):
#                text_chunks = self.handle_specific_tool(task, tool)
#                for i, chunk in enumerate(text_chunks, start=1):
#                    self.log_data.append({
#                        "timestamp": datetime.now().isoformat(),
#                        "type": "text_chunk",
#                        "task_id": task.id,
#                        "chunk_id": i,
#                        "text": chunk['text'],
#                        "start": chunk.get('start', 0),
#                        "end": chunk.get('end', len(chunk['text'])),
#                        "file": chunk.get('file', '')
#                    })

#            if isinstance(tool, SemanticAnalysisTool):
#                sentiment_result = tool.analyze_sentiment(task.output)
#                self.log_data.append({
#                    "timestamp": datetime.now().isoformat(),
#                    "type": "sentiment_analysis",
#                    "task_id": task.id,
#                    "content": sentiment_result
#                })
#                context += f"Sentiment Analysis Result: {sentiment_result}\n\n"

#            if isinstance(tool, NERExtractionTool):
#                entities = tool.extract_entities(task.output)
#                self.log_data.append({
#                    "timestamp": datetime.now().isoformat(),
#                    "type": "ner_extraction",
#                    "task_id": task.id,
#                    "content": [ent['text'] for ent in entities]
#                })
#                context += f"Extracted Entities: {[ent['text'] for ent in entities]}\n\n"

#    def handle_specific_tool(self, task, tool):
#        if isinstance(tool, SemanticFileSearchTool):
#            # Construct the query by joining the outputs from the context tasks
#            query = "\n".join([c.output for c in task.context if c.output])
#            return tool.search(query)
#        else:
#            return tool.read_text() if isinstance(tool, TextReaderTool) else tool.scrape_text()

#    def save_llama_logs(self):
#        with open(("qa_interactions" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"), "w") as file:
#            json.dump(self.llama_logs, file, indent=2)

#    def save_logs(self):
#        with open(self.log_file, "w") as file:
#            json.dump(self.log_data, file, indent=2)

def agent_dispatcher(query, agents,tools, resources):
    chat = [{"role": "user", "content": query}]
    prompter = PromptManager()
    sys_prompt = prompter.generate_prompt(tools, agents, resources)

    #client = CLIENTS['groq']
    client = CLIENTS.anthropic
    #response = client.chat.completions.create(
    response = client.messages.create(
        #model="adrienbrault/nous-hermes2pro:Q4_0",
        #model='llama3-70b-8192',
        model = "claude-3-opus-20240229",
        system = sys_prompt,
        max_tokens = 1000,
        temperature=0.5,
        messages=chat,
    )
    print(response)
    #completion = response.choices[0].message.content
    completion = response.content[0].text
        # Convert the tools list to a dictionary
    tools_dict = {tool: None for tool in tools}

    return completion, tools_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Run the squad with dynamic configurations.")
    parser.add_argument('-q', '--query', type=str, help="user query for agents to assist with", required=True)
    return parser.parse_args(

    )

def extract_agents_from_mermaid(mermaid_graph):
    # Extract the graph content from within the <graph> tags
    graph_content = re.search(r'<graph>(.*?)</graph>', mermaid_graph, re.DOTALL).group(1)

    # Extract the metadata content from within the <agents> tags
    metadata_content = re.search(r'<agents>(.*?)</agents>', mermaid_graph, re.DOTALL).group(1)

    dependency_pattern = r'(\w+) --> (\w+)'

    agents_metadata = json.loads(metadata_content)
    dependencies = []

    for match in re.finditer(dependency_pattern, graph_content):
        source = match.group(1)
        target = match.group(2)
        dependencies.append((source, target))

    # Add dependencies to agent metadata
    for agent in agents_metadata:
        agent["dependencies"] = [target for source, target in dependencies if source == agent["role"]]
        agent["tools"] = {tool: None for tool in agent["tools"]}
    
    return agents_metadata

def mainflow():
    args = parse_args()

    file_path = os.path.join(os.getcwd())
    with open(os.path.join(file_path, "configs/agents.json"), "r") as file:
        agents_data = json.load(file)
    agents = [Agent(**agent_data) for agent_data in agents_data]

    #with open(os.path.join(file_path, "configs/tasks.json"), "r") as file:
    #    tasks_data = json.load(file)
    #tasks = [Task(**task_data) for task_data in tasks_data]
#
    with open(os.path.join(file_path, "configs/resources.json"), "r") as file:
        resources_data = json.load(file)
    resources = [Resource(**resource_data) for resource_data in resources_data]

    tools = get_openai_tools()
    tools_dict = {tool["function"]["name"]: tool for tool in tools}

    # Check if the mermaid graph and agent metadata files exist
    mermaid_graph_file = "mermaid_graph.txt"
    agent_metadata_file = "agent_metadata.json"

    if os.path.exists(mermaid_graph_file) and os.path.exists(agent_metadata_file):
        # Read the mermaid graph and agent metadata from files
        with open(mermaid_graph_file, "r") as file:
            mermaid_graph = file.read()
        with open(agent_metadata_file, "r") as file:
            agents_metadata = json.load(file)
    else:
        # Generate the mermaid graph using the LLM
        mermaid_graph, tools_dict = agent_dispatcher(args.query, agents, tools, tools_dict)

        print(mermaid_graph)

        # Extract the agents metadata from the mermaid graph
        agents_metadata = extract_agents_from_mermaid(mermaid_graph)

        # Save the mermaid graph and agent metadata to files
        with open(mermaid_graph_file, "w") as file:
            file.write(mermaid_graph)
        with open(agent_metadata_file, "w") as file:
            json.dump(agents_metadata, file, indent=2)

    # Create a directed graph using NetworkX
    G = nx.DiGraph()

    # Add nodes to the graph using agent metadata
    for agent_data in agents_metadata:
        agent_role = agent_data["role"]
        G.add_node(agent_role, **agent_data)

    # Perform a topological sort to determine the execution order
    execution_order = list(nx.topological_sort(G))

    # Execute the agents in the determined order
    context = ""
    for agent_role in execution_order:
        agent_data = G.nodes[agent_role]
        agent = Agent(**agent_data)

        if agent.verbose:
            print(f"Starting Agent: {agent.role}")

        # Execute the agent and retrieve the output
        output = agent.execute(context=context)

        if agent.verbose:
            print(f"Agent output:\n{output}\n")

        context += f"Agent: {agent.role}\nGoal: {agent.goal}\nOutput:\n{output}\n\n"

    print(f"Final output:\n{context}")

    #system_docs_resource = Resources('text', "inputs/system_documentation.txt", "The following is a snippet from the system documentation '{file}' (start: {start}, end: {end}):\n{chunk}")
    #define toolsettings for flow sesh

    #text_reader_tool = TextReaderTool("cyberanimism_clean.txt", chunk_size=1000, num_chunks=5)
    #web_scraper_tool = WebScraperTool("http://matplotlib.org/stable/gallery/mplot3d/2dcollections3d.html#sphx-glr-gallery-mplot3d-2dcollections3d-py", chunk_size=512, num_chunks=2)
    #sentiment_analysis_tool = SemanticAnalysisTool("")
    #ner_extraction_tool = NERExtractionTool("")
    #system_docs_tool = TextReaderTool("system_documentation.txt", chunk_size=1000, num_chunks=10)  
    #semantic_search_tool = SemanticFileSearchTool(file_paths=['book1.pdf', 'cyberanimism_clean.txt'], embed_model='nomic-embed-text-v1.5.f16.gguf', chunk_size=500, top_k=5)
    #semantic_search_tool.save_embeddings('file_embeddings.pickle')
    #wikipedia_search_tool = WikipediaSearchTool(chunk_size=500, num_chunks=5)

    # the squad is where you define the flow of the pipeline, sequence of events.
#
    #squad = Squad(
    #    agents=agents,
    #    tasks=tasks,
    #    resources=resources,
    #    verbose=True,
    #    log_file="squad_goals" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    #)
#
    #result = squad.run()
    #print(f"Final output:\n{result}")

if __name__ == "__main__":
    mainflow()