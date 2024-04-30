import argparse
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
import sys
import inspect

from src.prompter import PromptManager
from src.tools import *
from src.resources import Resource
from src.agents import Agent
from src.tasks import Task

class Squad:
    def __init__(self, agents: List['Agent'], tasks: List['Task'], resources: List['Resource'], verbose: bool = False, log_file: str = "squad_log.json"):
        self.id = str(uuid.uuid4())
        self.agents = agents
        self.tasks = tasks
        self.resources = resources
        self.verbose = verbose
        self.log_file = log_file
        self.log_data = []
        self.llama_logs = []  # Attribute to store LLM interactions

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> str:
        context = ""
        for task in self.tasks:
            if self.verbose:
                print(f"Starting Task:\n{task.instructions}")

            # Log the input to the task
            self.log_data.append({
                "timestamp": datetime.now().isoformat(),
                "type": "input",
                "agent_role": task.agent.role,
                "task_name": task.instructions,
                "task_id": task.id,
                "content": task.instructions
            })

            # Execute the task and retrieve the output
            output = task.execute(context=context)
            task.output = output

            if self.verbose:
                print(f"Task output:\n{output}\n")

            # Log the output from the task
            self.log_data.append({
                "timestamp": datetime.now().isoformat(),
                "type": "output",
                "agent_role": task.agent.role,
                "task_name": task.instructions,
                "task_id": task.id,
                "content": output
            })

            # Collect LLM interactions from the agent after task execution
            self.llama_logs.extend(task.agent.interactions)  # Assuming interactions are collected in each agent

            context += f"Task:\n{task.instructions}\nOutput:\n{output}\n\n"

            # Additional logic for handling specific tools used by tasks
            self.handle_tool_logic(task, context)

        # Writing all logged data and LLM interactions to JSON files
        self.save_logs()
        self.save_llama_logs()

        return context

    def handle_tool_logic(self, task, context):
        if task.tool_name in task.agent.tools:
            tool = task.agent.tools[task.tool_name]
            if isinstance(tool, TextReaderTool) or isinstance(tool, WebScraperTool) or isinstance(tool, SemanticFileSearchTool):
                text_chunks = self.handle_specific_tool(task, tool)
                for i, chunk in enumerate(text_chunks, start=1):
                    self.log_data.append({
                        "timestamp": datetime.now().isoformat(),
                        "type": "text_chunk",
                        "task_id": task.id,
                        "chunk_id": i,
                        "text": chunk['text'],
                        "start": chunk.get('start', 0),
                        "end": chunk.get('end', len(chunk['text'])),
                        "file": chunk.get('file', '')
                    })

            if isinstance(tool, SemanticAnalysisTool):
                sentiment_result = tool.analyze_sentiment(task.output)
                self.log_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "sentiment_analysis",
                    "task_id": task.id,
                    "content": sentiment_result
                })
                context += f"Sentiment Analysis Result: {sentiment_result}\n\n"

            if isinstance(tool, NERExtractionTool):
                entities = tool.extract_entities(task.output)
                self.log_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "ner_extraction",
                    "task_id": task.id,
                    "content": [ent['text'] for ent in entities]
                })
                context += f"Extracted Entities: {[ent['text'] for ent in entities]}\n\n"

    def handle_specific_tool(self, task, tool):
        if isinstance(tool, SemanticFileSearchTool):
            # Construct the query by joining the outputs from the context tasks
            query = "\n".join([c.output for c in task.context if c.output])
            return tool.search(query)
        else:
            return tool.read_text() if isinstance(tool, TextReaderTool) else tool.scrape_text()

    def save_llama_logs(self):
        with open(("qa_interactions" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"), "w") as file:
            json.dump(self.llama_logs, file, indent=2)

    def save_logs(self):
        with open(self.log_file, "w") as file:
            json.dump(self.log_data, file, indent=2)

def get_tool_names():
    # Get all the classes defined in the script
    classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    # Extract the class names excluding the imported ones
    class_names = [cls[0] for cls in classes if cls[1].__module__ == 'src.tools']

    return class_names

def agent_dispatcher(query, agents,tools, resources):
    chat = [{"role": "user", "content": query}]
    prompter = PromptManager()
    messages = prompter.generate_prompt(chat, tools, agents, resources)

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama'
    )
    response = client.chat.completions.create(
        model="adrienbrault/nous-hermes2pro:Q4_0",
        messages=messages,
    )
    print(response)
    completion = response.choices[0].message.content

    return completion

def parse_args():
    parser = argparse.ArgumentParser(description="Run the squad with dynamic configurations.")
    parser.add_argument('-q', '--query', type=str, help="user query for agents to assist with", required=True)
    return parser.parse_args()

def mainflow():
    args = parse_args()

    file_path = os.path.join(os.getcwd())
    with open(os.path.join(file_path, "configs/agents.json"), "r") as file:
        agents = json.load(file)
    agents = [Agent(**agent) for agent in agents]

    with open(os.path.join(file_path, "configs/tasks.json"), "r") as file:
        tasks = json.load(file)
    tasks = [Task(**task) for task in tasks]

    with open(os.path.join(file_path, "configs/resources.json"), "r") as file:
        resources = json.load(file)
    resources = [Resource(**resource) for resource in resources]

    tools = get_tool_names()

    # TODO we need to restrcuture agents such that tasks and tools are filled by the LLM as part of agent metadata
    graph = agent_dispatcher(args.query, agents, tools, resources)

    print(graph)

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