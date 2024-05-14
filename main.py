import os
import re
import json
import uuid
import argparse
import networkx as nx
from datetime import datetime

import streamlit as st

from src.rag_tools import *
from src.prompter import PromptManager
from src.resources import Resource
from src.agents import Agent
from src.clients import CLIENTS
from src.tools import get_openai_tools

from matplotlib import pyplot as plt

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

        mermaid_graph, agents_metadata = self.load_or_generate_graph(query, self.agents, tools, self.resources)
        st.write(mermaid_graph)

        G = nx.DiGraph()
        for agent_data in agents_metadata:
            agent_role = agent_data["role"]
            G.add_node(agent_role, **agent_data)

        # Add edges to the graph based on the dependencies
        for agent_data in agents_metadata:
            agent_role = agent_data["role"]
            dependencies = agent_data.get("dependencies", [])
            for dependency in dependencies:
                G.add_edge(dependency, agent_role)

        # Visualize the graph using python-mermaid
        self.visualize_graph(G)

        # Create a dictionary to store the output of each agent
        agent_outputs = {}

        # Execute agents in topological order (respecting dependencies)
        for agent_role in nx.topological_sort(G):
            agent_data = G.nodes[agent_role]
            agent = Agent(**agent_data)

            if agent.verbose:
                st.write(f"<font color='white'>Starting Agent: {agent.role}</font>", unsafe_allow_html=True)
                st.write(f"<font color='purple'>Agent Persona: {agent.persona}</font>", unsafe_allow_html=True)
                st.write(f"<font color='orange'>Agent Goal: {agent.goal}</font>", unsafe_allow_html=True)

            # Prepare the input messages for the agent
            input_messages = []
            for predecessor in G.predecessors(agent_role):
                if predecessor in agent_outputs:
                    input_messages.append({"role": predecessor, "content": agent_outputs[predecessor]})

            agent.input_messages = input_messages

            # Execute the agent
            output = agent.execute()

            if agent.verbose:
                st.write(f"<font color='green'>Agent Output:\n{output}\n</font>", unsafe_allow_html=True)

            agent_outputs[agent_role] = output
            self.llama_logs.extend(agent.interactions)

        # Collect the final output from all the agents
        final_output = "\n".join([f"Agent: {role}\nGoal: {G.nodes[role]['goal']}\nOutput:\n{output}\n" for role, output in agent_outputs.items()])

        self.save_logs()
        self.save_llama_logs()

        return final_output
    
    def visualize_graph(self, G):
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, node_size=1000, node_color='lightblue', font_size=12, font_weight='bold', arrows=True)
        labels = nx.get_node_attributes(G, 'role')
        nx.draw_networkx_labels(G, pos, labels, font_size=12)
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
    
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

        response = CLIENTS.chat_completion(
            client="anthropic",
            messages=[
                {"role": "system", "content": sys_prompt},
                *chat
            ]
        )
        st.write(response)
        return response

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
    st.title("Stock Analysis with MeeseeksAI Agents")
    multiline_text = """
    Try to ask it "What is the current price of Meta stock?" or "Show me the historical prices of Apple vs Microsoft stock over the past 6 months.".
    """

    st.markdown(multiline_text, unsafe_allow_html=True)

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    additional_context = st.sidebar.text_input('Enter additional summarization context for the LLM here (i.e. write it in spanish):')

    # Get the user's question
    user_question = st.text_input("Ask a question about a stock or multiple stocks:")

    if user_question:
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

        orchestrator.run(user_question)
        
        ## Wrap the final output in a scrollable container
        #output_container = st.container()
        #with output_container:
        #    st.write(f"Final output:\n{result}")
        #
        ## Make the output container scrollable
        #output_container_height = min(len(result.split('\n')) * 30, 500)  # Adjust the height based on the number of lines
        #output_container.markdown(
        #    f"""
        #    <style>
        #        .stContainer {{
        #            max-height: {output_container_height}px;
        #            overflow-y: auto;
        #        }}
        #    </style>
        #    """,
        #    unsafe_allow_html=True
        #)

if __name__ == "__main__":
    mainflow()