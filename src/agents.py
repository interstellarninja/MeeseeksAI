from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Callable
from src.tools import TextReaderTool, WebScraperTool, SemanticAnalysisTool, NERExtractionTool, SemanticFileSearchTool, WikipediaSearchTool
from openai import OpenAI
import uuid
from datetime import datetime

class Agent(BaseModel):
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    goal: str
    tools: Dict[str, Any] = {}
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
    client: OpenAI = Field(default_factory=lambda: OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    ))

    def execute_task(self, task, context: Optional[str] = None) -> str:
        messages = []
        if self.persona and self.verbose:
            messages.append({"role": "system", "content": f"Background: {self.persona}"})
        messages.append({"role": "system", "content": f"You are a {self.role} with the goal: {self.goal}. The expected output is: {task.expected_output}"})
        messages.append({"role": "user", "content": f"Your task is to {task.instructions}."})
        if context:
            messages.append({"role": "assistant", "content": f"Context from {task.context_agent_role}:\n{context}"})

        if task.tool_name in self.tools:
            tool = self.tools[task.tool_name]
            if isinstance(tool, TextReaderTool) or isinstance(tool, WebScraperTool):
                text_chunks = tool.read_text() if isinstance(tool, TextReaderTool) else tool.scrape_text()
                for i, chunk in enumerate(text_chunks, start=1):
                    messages.append({"role": "system", "content": f"<text_chunk{i}>{chunk['text']}</text_chunk{i}>"})
            elif isinstance(tool, SemanticAnalysisTool):
                sentiment_result = tool.analyze_sentiment()
                messages.append({"role": "system", "content": f"Sentiment Analysis Result: {sentiment_result}"})
            elif isinstance(tool, NERExtractionTool):
                entities = tool.extract_entities(context)
                messages.append({"role": "system", "content": f"Extracted Entities: {entities}"})
            elif isinstance(tool, SemanticFileSearchTool):
                query = "\n".join([c.output for c in task.context if c.output])
                relevant_chunks = tool.search(query)
                for chunk in relevant_chunks:
                    chunk_text = f"File: {chunk['file']}\nText: {chunk['text']}\nScore: {chunk['score']:.3f}"
                    messages.append({"role": "system", "content": chunk_text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        result = response.choices[0].message.content
        self.log_interaction(messages, result)

        if self.step_callback:
            self.step_callback(task, result)

        return result

    def log_interaction(self, prompt, response):
        self.interactions.append({
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

# Define the agents
researcher = Agent(
    role='Researcher',
    goal='Analyze the provided text and extract relevant information.',
    persona="""You are a renowned Content Strategist, known for your insightful and engaging articles. You transform complex concepts into compelling narratives.""",
    tools={"text_reader": TextReaderTool},
    verbose=True
)

wikipedia_expert = Agent(
    role='Wikipedia Expert',
    goal='Provide contextual information from Wikipedia articles relevant to a given topic.',
    persona='You are an expert in searching and summarizing information from Wikipedia on various topics.',
    tools={'wikipedia_search': WikipediaSearchTool},
    verbose=True
)

web_analyzer = Agent(
    role='Web Analyzer',
    goal='Analyze the scraped web content and provide a summary.',
    tools={"web_scraper": WebScraperTool},
    verbose=True
)

sentimentalizer = Agent(
    role='Sentimentalizer',
    goal='Analyze the sentiment of the extracted information.',
    tools={"sentiment_analysis": SemanticAnalysisTool},
    verbose=True
)

planner = Agent(
    role="Planner",
    goal="Develop comprehensive plans and strategies for efficient systems management.",
    persona="You are an experienced Systems Manager with a proven track record in developing and implementing effective plans and strategies to optimize system performance, ensure reliability, and improve operational efficiency.",
    tools={"system_docs": TextReaderTool},
    verbose=True
)

semantic_searcher = Agent(
    role='Semantic Searcher',
    goal='Perform semantic searches on a corpus of files to find relevant information.',
    persona='You are an expert in semantic search and information retrieval.',
    tools={'semantic_search': SemanticFileSearchTool},
    verbose=True
)

summarizer = Agent(
    role='Summarizer',
    persona="""You are a skilled Data Analyst with a knack for distilling complex information into concise summaries.""",
    goal='Compile a summary report based on the extracted information.',
    verbose=True
)

entity_extractor = Agent(
    role='Entity Extractor',
    goal='Extract named entities from the given text.',
    tools={"ner_extraction": NERExtractionTool},
    model="llama3",
    verbose=True
)

mermaid = Agent(
    role='Mermaid',
    goal='Generate an accurate representation of the information as a graph.',
    model="adrienbrault/nous-hermes2pro:Q4_0",
    verbose=True
)