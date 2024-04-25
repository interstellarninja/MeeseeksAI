from tools import TextReaderTool, WebScraperTool, SemanticAnalysisTool, NERExtractionTool, SemanticFileSearchTool, WikipediaSearchTool
from openai import OpenAI
import uuid
from datetime import datetime

class Agent:
    def __init__(
        self,
        role: str,
        goal: str,
        tools: dict = {},
        verbose: bool = False,
        model: str = "adrienbrault/nous-hermes2pro:Q4_0", #default agent model
        max_iter: int = 25,
        max_rpm: int = None,
        max_execution_time: int = None,
        cache: bool = True,
        step_callback: callable = None,
        persona: str = None,
        allow_delegation: bool = False,
        input_tasks: list = None,
        output_tasks: list = None,
    ):
        self.id = str(uuid.uuid4())
        self.role = role
        self.goal = goal
        self.tools = tools
        self.verbose = verbose
        self.model = model
        self.max_iter = max_iter
        self.max_rpm = max_rpm
        self.max_execution_time = max_execution_time
        self.cache = cache
        self.step_callback = step_callback
        self.persona = persona
        self.allow_delegation = allow_delegation
        self.input_tasks = input_tasks or []
        self.output_tasks = output_tasks or []
        self.interactions = []  # To log prompts and responses
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',
        )

    def execute_task(self, task, context: str = None) -> str:
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
                # Assume 'context' contains the necessary text to analyze
                entities = tool.extract_entities(context)
                messages.append({"role": "system", "content": f"Extracted Entities: {entities}"})
            elif isinstance(tool, SemanticFileSearchTool):
                # Construct the query by joining the outputs from the context tasks
                query = "\n".join([c.output for c in task.context if c.output])
                
                # Perform the semantic search and get the relevant file chunks
                relevant_chunks = tool.search(query)
                
                # Add the relevant file chunks to the messages
                for chunk in relevant_chunks:
                    chunk_text = f"File: {chunk['file']}\nText: {chunk['text']}\nScore: {chunk['score']:.3f}"
                    messages.append({"role": "system", "content": chunk_text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        result = response.choices[0].message.content

        # Log the interaction
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
    model="madrienbrault/nous-hermes2pro:Q4_0",
    verbose=True
)