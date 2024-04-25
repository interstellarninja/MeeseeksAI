import json
import uuid
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import spacy
from textblob import TextBlob
from tiktoken import get_encoding
import pickle
from gpt4all import Embed4All
import requests
from bs4 import BeautifulSoup
import PyPDF2
import os
import hashlib

# Utilities for chunking text

class TextChunker:
    def __init__(self, text: str = None, chunk_size: int = 1000, overlap: int = 0):
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = get_encoding("cl100k_base")

    def chunk_text(self, text: str = None, chunk_size: int = None, start_pos: int = 0) -> List[Dict[str, Any]]:
        if text is not None:
            self.text = text
        if chunk_size is not None:
            self.chunk_size = chunk_size

        tokens = self.encoding.encode(self.text)
        num_tokens = len(tokens)

        chunks = []
        current_pos = start_pos

        while current_pos < num_tokens:
            chunk_start = max(0, current_pos - self.overlap)
            chunk_end = min(current_pos + self.chunk_size, num_tokens)

            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append({
                "text": chunk_text,
                "start": chunk_start,
                "end": chunk_end
            })

            current_pos += self.chunk_size - self.overlap

        return chunks

# Tools for reading text, scraping web content, extracting named entities, and performing sentiment analysis

class WikipediaSearchTool:
    def __init__(self, chunk_size: int = 1000, num_chunks: int = 10):
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.chunker = TextChunker()

    def search_wikipedia(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        url = f"https://en.wikipedia.org/w/index.php?search={query}&title=Special:Search&fulltext=1"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        search_results = []
        for result in soup.find_all('li', class_='mw-search-result'):
            title = result.find('a').get_text()
            url = 'https://en.wikipedia.org' + result.find('a')['href']
            page_response = requests.get(url)
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            content = page_soup.find('div', class_='mw-parser-output').get_text()
            chunks = self.chunker.chunk_text(text=content, chunk_size=self.chunk_size, num_chunks=self.num_chunks)
            search_results.append({'title': title, 'url': url, 'chunks': chunks})
            if len(search_results) >= top_k:
                break

        return search_results

# Takes a list of file paths, embeds the text in the files, and allows semantic search based on a query

class SemanticFileSearchTool:
    def __init__(self, file_paths: List[str], embed_model: str, embed_dim: int = 768, chunk_size: int = 1000, top_k: int = 3):
        self.embedder = Embed4All(embed_model)
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.chunker = TextChunker(text=None, chunk_size=chunk_size)
        self.file_embeddings = self.load_or_generate_file_embeddings(file_paths)

    def load_or_generate_file_embeddings(self, file_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        file_hash = self.get_file_hash(file_paths)
        pickle_file = f"file_embeddings_{file_hash}.pickle"
        if os.path.exists(pickle_file):
            self.load_embeddings(pickle_file)
        else:
            self.file_embeddings = self.generate_file_embeddings(file_paths)
            self.save_embeddings(pickle_file)
        return self.file_embeddings

    def get_file_hash(self, file_paths: List[str]) -> str:
        file_contents = "".join(sorted([os.path.basename(path) for path in file_paths]))
        return hashlib.sha256(file_contents.encode()).hexdigest()

    def generate_file_embeddings(self, file_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        file_embeddings = {}
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r') as file:
                    text = file.read()
            chunks = self.chunker.chunk_text(text=text, chunk_size=self.chunk_size)
            chunk_embeddings = [self.embedder.embed(chunk['text'], prefix='search_document') for chunk in chunks]
            file_embeddings[file_path] = [(chunk['text'], embedding) for chunk, embedding in zip(chunks, chunk_embeddings)]
        return file_embeddings

    def extract_text_from_pdf(self, file_path: str) -> str:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def save_embeddings(self, pickle_file: str):
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.file_embeddings, f)

    def load_embeddings(self, pickle_file: str):
        with open(pickle_file, 'rb') as f:
            self.file_embeddings = pickle.load(f)

    def search(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed(query, prefix='search_query')
        scores = []
        for file_path, chunk_data in self.file_embeddings.items():
            for chunk_text, embedding in chunk_data:
                chunk_score = self.cosine_similarity(query_embedding, embedding)
                scores.append(((file_path, chunk_text), chunk_score))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_scores = sorted_scores[:self.top_k]
        result = []
        for (file_path, chunk_text), score in top_scores:
            result.append({
                'file': file_path,
                'text': chunk_text,
                'score': score
            })
        return result

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Simple tool to read text from a file and chunk it into smaller pieces

class TextReaderTool:
    def __init__(self, text_file: str, chunk_size: int, num_chunks: int):
        self.text_file = text_file
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

    def read_text(self) -> List[Dict[str, Any]]:
        with open(self.text_file, "r") as file:
            text = file.read()
            chunker = TextChunker(text, self.chunk_size, overlap=0)
            chunks = chunker.chunk_text()
            return chunks[:self.num_chunks]

# Tool to scrape text from a web page and chunk it into smaller pieces

class WebScraperTool:
    def __init__(self, url: str, chunk_size: int, num_chunks: int):
        self.url = url
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

    def scrape_text(self) -> List[Dict[str, Any]]:
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        text = soup.get_text(separator='\n')
        cleaned_text = ' '.join(text.split())
        
        chunker = TextChunker(cleaned_text, self.chunk_size, overlap=0)
        chunks = chunker.chunk_text()
        return chunks[:self.num_chunks]

# Tool to extract named entities from text using spaCy

class NERExtractionTool:
    def __init__(self, text: str = None):
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text: Optional[str] = None) -> List[Dict[str, Any]]:
        if text is not None:
            self.text = text
        doc = self.nlp(self.text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })

        return entities

# Tool to perform sentiment analysis using TextBlob

class SemanticAnalysisTool:
    def __init__(self, text: str = None):
        self.text = text

    def analyze_sentiment(self, text: Optional[str] = None) -> Dict[str, Any]:
        if text is not None:
            self.text = text
        blob = TextBlob(self.text)
        sentiment = blob.sentiment
        return {
            "polarity": sentiment.polarity,
            "subjectivity": sentiment.subjectivity
        }

# Agent construction class for defining the agents, roles, goals, tools, verbose, model, max_iter, max_rpm, max_execution_time, cache, step_callback, persona, allow_delegation, input_tasks, and output_tasks

class Agent:
    def __init__(
        self,
        role: str,
        goal: str,
        tools: Optional[List[Any]] = None,
        verbose: bool = False,
        model: str = "mistral:instruct", #default agent model
        max_iter: int = 25,
        max_rpm: Optional[int] = None,
        max_execution_time: Optional[int] = None,
        cache: bool = True,
        step_callback: Optional[Callable] = None,
        persona: Optional[str] = None,
        allow_delegation: bool = False,
        input_tasks: Optional[List["Task"]] = None,
        output_tasks: Optional[List["Task"]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.role = role
        self.goal = goal
        self.tools = tools or {}
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

    def execute_task(self, task: "Task", context: Optional[str] = None) -> str:
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



class Squad:
    def __init__(self, agents: List['Agent'], tasks: List['Task'], verbose: bool = False, log_file: str = "squad_log.json"):
        self.id = str(uuid.uuid4())
        self.agents = agents
        self.tasks = tasks
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


def mainflow():

    #define toolsettings for flow sesh

    text_reader_tool = TextReaderTool("cyberanimism_clean.txt", chunk_size=1000, num_chunks=5)
    web_scraper_tool = WebScraperTool("http://matplotlib.org/stable/gallery/mplot3d/2dcollections3d.html#sphx-glr-gallery-mplot3d-2dcollections3d-py", chunk_size=512, num_chunks=2)
    sentiment_analysis_tool = SemanticAnalysisTool("")
    ner_extraction_tool = NERExtractionTool("")
    system_docs_tool = TextReaderTool("system_documentation.txt", chunk_size=1000, num_chunks=10)  
    semantic_search_tool = SemanticFileSearchTool(file_paths=['book1.pdf', 'cyberanimism_clean.txt'], embed_model='nomic-embed-text-v1.5.f16.gguf', chunk_size=500, top_k=5)
    semantic_search_tool.save_embeddings('file_embeddings.pickle')
    wikipedia_search_tool = WikipediaSearchTool(chunk_size=500, num_chunks=5)


    # the agents, roles, goals, tools, verbose, persona, etc. make up the personas of the agents. these can be re-enforced with grounded txt or semantic data (wip).

    researcher = Agent(
        role='Researcher',
        goal='Analyze the provided text and extract relevant information.',
        persona="""You are a renowned Content Strategist, known for your insightful and engaging articles. You transform complex concepts into compelling narratives.""",
        tools={"text_reader": text_reader_tool},
        verbose=True
    )

    wikipedia_expert = Agent(
        role='Wikipedia Expert',
        goal='Provide contextual information from Wikipedia articles relevant to a given topic.',
        persona='You are an expert in searching and summarizing information from Wikipedia on various topics.',
        tools={'wikipedia_search': wikipedia_search_tool},
        verbose=True
    )

    web_analyzer = Agent(
        role='Web Analyzer',
        goal='Analyze the scraped web content and provide a summary.',
        tools={"web_scraper": web_scraper_tool},
        verbose=True
    )

    sentimentalizer = Agent(
        role='Sentimentalizer',
        goal='Analyze the sentiment of the extracted information.',
        tools={"sentiment_analysis": sentiment_analysis_tool},
        verbose=True
    )

    planner = Agent(
        role="Planner",  # Added the 'role' argument
        goal="Develop comprehensive plans and strategies for efficient systems management.",
        persona="You are an experienced Systems Manager with a proven track record in developing and implementing effective plans and strategies to optimize system performance, ensure reliability, and improve operational efficiency.",
        tools={"system_docs": system_docs_tool},
        verbose=True
    )

    semantic_searcher = Agent(
        role='Semantic Searcher',
        goal='Perform semantic searches on a corpus of files to find relevant information.',
        persona='You are an expert in semantic search and information retrieval.',
        tools={'semantic_search': semantic_search_tool},
        verbose=True
    )

    wikipedia_search_task = Task(
        instructions='Use Wikipedia to find relevant articles and summarize the key information related to the given topic.',
        expected_output='A summary of the top Wikipedia articles related to the given topic, including article titles, URLs, and chunked content.',
        agent=wikipedia_expert,
        tool_name='wikipedia_search'
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
        tools={"ner_extraction": ner_extraction_tool},
        model="llama3",
        verbose=True
    )

    mermaid = Agent(
        role='Mermaid',
        goal='Generate an accurate representation of the information as a graph.',
        model="mermaidGRAPH:latest",
        verbose=True
    )

    # the tasks these are selfexplanatory, agents DO TEH TASKs.

    txt_task = Task(
        instructions="Analyze the provided text and identify key insights and patterns.",
        expected_output="A list of key insights and patterns found in the text.",
        agent=researcher,
        output_file='txt_analyzed.txt',
        tool_name="text_reader",
    )

    web_task = Task(
        instructions="Scrape the content from the provided URL and provide a summary.",
        expected_output="A summary of the scraped web content.",
        agent=web_analyzer,
        tool_name="web_scraper",

        output_file='web_task_output.txt',
    )
    
    system_plan = Task(
        instructions="Analyze the provided system documentation and develop a comprehensive plan for enhancing system performance, reliability, and efficiency.",
        expected_output="A detailed plan outlining strategies and steps for optimizing the systems.",
        agent=planner,
        tool_name="system_docs",
        output_file="system_plan.txt",
    )

    search_task = Task(
        instructions='Search the provided files for information relevant to the given query.',
        expected_output='A list of relevant files with their similarity scores.',
        agent=semantic_searcher,
        tool_name='semantic_search',
        context=[system_plan, txt_task],
    )

    summary = Task(
        instructions="Using the insights from the researcher and web analyzer, compile a summary report.",
        expected_output="A well-structured summary report based on the extracted information.",
        agent=summarizer,
        context=[system_plan, txt_task, web_task],
        output_file='task2_output.txt',
    )

    vibes = Task(
        instructions="Analyze the sentiment of the extracted information.",
        expected_output="A sentiment analysis report based on the extracted information.",
        agent=sentimentalizer,
        context=[summary, txt_task],
        output_file='sentimentalizer_output.txt',
        tool_name="sentiment_analysis",
    )

    ner_task = Task(
        instructions="Extract named entities from the summary report.",
        expected_output="A list of extracted named entities.",
        agent=entity_extractor,
        context=[summary, search_task],
        output_file='ner_output.txt',
        tool_name="ner_extraction",
    )

    mermaidGRAPH = Task(
        instructions="Generate a mermaid diagram based on the summary report.",
        expected_output="A mermaid graph illustrating the relationships and connections in the summary report.\n```mermaid\ngraph TD\n",
        agent=mermaid,
        context=[summary, txt_task, search_task],
        output_file='mermaid_output.txt',
    )   

    # the squad is where you define the flow of the pipeline, sequence of events.

    squad = Squad(
        agents=[researcher, web_analyzer, planner, summarizer, semantic_searcher, sentimentalizer, entity_extractor, mermaid],
        tasks=[txt_task, web_task, system_plan, summary, search_task, vibes, ner_task, mermaidGRAPH],
        verbose=True,
        log_file="squad_goals" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    )

    result = squad.run()
    print(f"Final output:\n{result}")

if __name__ == "__main__":
    mainflow()