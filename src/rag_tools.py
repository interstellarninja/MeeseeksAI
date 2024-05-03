import hashlib
import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
import spacy
from textblob import TextBlob
import pickle
from gpt4all import Embed4All
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from resources import TextChunker

class WikipediaSearchInput(BaseModel):
    query: str = Field(description="The search query for Wikipedia")
    top_k: int = Field(default=3, description="The number of top search results to return")

class WikipediaSearchTool(BaseTool):
    name = "wikipedia_search"
    description = "Searches Wikipedia for relevant information based on a given query"
    args_schema: Type[BaseModel] = WikipediaSearchInput

    def _run(self, query: str, top_k: int = 3, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, str]]:
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
            chunks = TextChunker().chunk_text(text=content, chunk_size=1000, num_chunks=10)
            search_results.append({'title': title, 'url': url, 'chunks': chunks})
            if len(search_results) >= top_k:
                break

        return search_results

    async def _arun(self, query: str, top_k: int = 3, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, str]]:
        raise NotImplementedError("WikipediaSearchTool does not support async")

class SemanticFileSearchInput(BaseModel):
    query: str = Field(description="The search query for semantic file search")

class SemanticFileSearchTool(BaseTool):
    name = "semantic_file_search"
    description = "Performs semantic search on a set of files based on a given query"
    args_schema: Type[BaseModel] = SemanticFileSearchInput

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

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
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

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("SemanticFileSearchTool does not support async")

class TextReaderInput(BaseModel):
    text_file: str = Field(description="The path to the text file to read")

class TextReaderTool(BaseTool):
    name = "text_reader"
    description = "Reads text from a file and chunks it into smaller pieces"
    args_schema: Type[BaseModel] = TextReaderInput

    def __init__(self, chunk_size: int, num_chunks: int):
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

    def _run(self, text_file: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
        with open(text_file, "r") as file:
            text = file.read()
            chunker = TextChunker(text, self.chunk_size, overlap=0)
            chunks = chunker.chunk_text()
            return chunks[:self.num_chunks]

    async def _arun(self, text_file: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("TextReaderTool does not support async")

class WebScraperInput(BaseModel):
    url: str = Field(description="The URL of the web page to scrape")

class WebScraperTool(BaseTool):
    name = "web_scraper"
    description = "Scrapes text from a web page and chunks it into smaller pieces"
    args_schema: Type[BaseModel] = WebScraperInput

    def __init__(self, chunk_size: int, num_chunks: int):
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        text = soup.get_text(separator='\n')
        cleaned_text = ' '.join(text.split())
        
        chunker = TextChunker(cleaned_text, self.chunk_size, overlap=0)
        chunks = chunker.chunk_text()
        return chunks[:self.num_chunks]

    async def _arun(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("WebScraperTool does not support async")

class NERExtractionInput(BaseModel):
    text: str = Field(description="The text to extract named entities from")

class NERExtractionTool(BaseTool):
    name = "ner_extraction"
    description = "Extracts named entities from text using spaCy"
    args_schema: Type[BaseModel] = NERExtractionInput

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })

        return entities

    async def _arun(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("NERExtractionTool does not support async")

class SemanticAnalysisInput(BaseModel):
    text: str = Field(description="The text to perform sentiment analysis on")

class SemanticAnalysisTool(BaseTool):
    name = "semantic_analysis"
    description = "Performs sentiment analysis using TextBlob"
    args_schema: Type[BaseModel] = SemanticAnalysisInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return {
            "polarity": sentiment.polarity,
            "subjectivity": sentiment.subjectivity
        }

    async def _arun(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        raise NotImplementedError("SemanticAnalysisTool does not support async")