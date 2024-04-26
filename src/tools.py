import hashlib
import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
import spacy
from textblob import TextBlob
import pickle
from gpt4all import Embed4All
from typing import Any, Callable, Dict, List, Optional

from src.resources import Resources, TextChunker

# Tools for reading text, scraping web content, extracting named entities, and performing sentiment analysis

class WikipediaSearchTool:
    def __init__(self, resource: Resources, chunk_size: int = 1000, num_chunks: int = 10):
        self.resource = resource
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
    def __init__(self, resource: Resources, file_paths: List[str], embed_model: str, embed_dim: int = 768, chunk_size: int = 1000, top_k: int = 3):
        self.resource = resource
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
    def __init__(self, resource: Resources, text_file: str, chunk_size: int, num_chunks: int):
        self.resource = resource
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
    def __init__(self, resource: Resources, url: str, chunk_size: int, num_chunks: int):
        self.resource = resource
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
    def __init__(self, resource: Resources, text: str = None):
        self.resource = resource
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
    def __init__(self, resource: Resources, text: str = None):
        self.resource = resource
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