from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Callable
from src.tools import TextReaderTool, WebScraperTool, SemanticAnalysisTool, NERExtractionTool, SemanticFileSearchTool, WikipediaSearchTool
from openai import OpenAI
import uuid
from datetime import datetime

CLIENTS = {
    "ollama": OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    ),
    "openai": OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='openai',
    ),
    # Add more client definitions as needed
}

class Agent(BaseModel):
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types
        exclude = {"client", "tools"}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    goal: str
    tools: List[str] = []
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

    def __init__(self, **data: Any):
            super().__init__(**data)
            if not self.client:
                raise ValueError("Client must be specified.")
            self.client = CLIENTS.get(self.client)
            if not self.client:
                raise ValueError("Invalid client specified.")

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

