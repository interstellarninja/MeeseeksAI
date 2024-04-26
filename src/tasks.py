from typing import Callable, List, Optional
from src import agents
from src.agents import Agent
import uuid
from datetime import datetime
# Task class for defining the tasks, expected output, agents, async execution, context, output file, callback, human input, tool name, input tasks, and output tasks
class Task:
    def __init__(
        self,
        instructions: str,
        expected_output: str,
        agent: Optional[Agent] = None,
        async_execution: bool = False,
        context: Optional[List["Task"]] = None,
        output_file: Optional[str] = None,
        callback: Optional[Callable] = None,
        human_input: bool = False,
        tool_name: Optional[str] = None,
        input_tasks: Optional[List["Task"]] = None,
        output_tasks: Optional[List["Task"]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.instructions = instructions
        self.expected_output = expected_output
        self.agent = agent
        self.async_execution = async_execution
        self.context = context or []
        self.output_file = output_file
        self.callback = callback
        self.human_input = human_input
        self.output = None
        self.context_agent_role = None
        self.tool_name = tool_name
        self.input_tasks = input_tasks or []
        self.output_tasks = output_tasks or []
        self.prompt_data = []  # List to hold prompt data for logging

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
    

txt_task = Task(
    instructions="Analyze the provided text and identify key insights and patterns.",
    expected_output="A list of key insights and patterns found in the text.",
    agent=agents.researcher,
    output_file='txt_analyzed.txt',
    tool_name="text_reader",
)

web_task = Task(
    instructions="Scrape the content from the provided URL and provide a summary.",
    expected_output="A summary of the scraped web content.",
    agent=agents.web_analyzer,
    tool_name="web_scraper",
    output_file='web_task_output.txt',
)
    
system_plan = Task(
    instructions="Analyze the provided system documentation and develop a comprehensive plan for enhancing system performance, reliability, and efficiency.",
    expected_output="A detailed plan outlining strategies and steps for optimizing the systems.",
    agent=agents.planner,
    tool_name="system_docs",
    output_file="system_plan.txt",
)

search_task = Task(
    instructions='Search the provided files for information relevant to the given query.',
    expected_output='A list of relevant files with their similarity scores.',
    agent=agents.semantic_searcher,
    tool_name='semantic_search',
    context=[system_plan, txt_task],
)

summary = Task(
    instructions="Using the insights from the researcher and web analyzer, compile a summary report.",
    expected_output="A well-structured summary report based on the extracted information.",
    agent=agents.summarizer,
    context=[system_plan, txt_task, web_task],
    output_file='task2_output.txt',
)

vibes = Task(
    instructions="Analyze the sentiment of the extracted information.",
    expected_output="A sentiment analysis report based on the extracted information.",
    agent=agents.sentimentalizer,
    context=[summary, txt_task],
    output_file='sentimentalizer_output.txt',
    tool_name="sentiment_analysis",
)

ner_task = Task(
    instructions="Extract named entities from the summary report.",
    expected_output="A list of extracted named entities.",
    agent=agents.entity_extractor,
    context=[summary, search_task],
    output_file='ner_output.txt',
    tool_name="ner_extraction",
)

mermaidGRAPH = Task(
    instructions="Generate a mermaid diagram based on the summary report.",
    expected_output="A mermaid graph illustrating the relationships and connections in the summary report.\n```mermaid\ngraph TD\n",
    agent=agents.mermaid,
    context=[summary, txt_task, search_task],
    output_file='mermaid_output.txt',
)   