import PyPDF2
import requests
from typing import Dict, Any, List, Optional

from tiktoken import get_encoding

from pydantic import BaseModel
import PyPDF2
import requests
from typing import Dict, Any, List

class Resource(BaseModel):
    type: str
    path: str
    context_template: Optional[str] = None
    data: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = []

    def load_resource(self):
        if self.resource_type == 'text':
            return self.load_text()
        elif self.resource_type == 'pdf':
            return self.load_pdf()
        elif self.resource_type == 'web':
            return self.load_web()
        else:
            raise ValueError(f"Unsupported resource type: {self.resource_type}")

    def load_text(self):
        with open(self.resource_path, 'r') as file:
            return file.read()

    def load_pdf(self):
        with open(self.resource_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

    def load_web(self):
        response = requests.get(self.resource_path)
        return response.text

    def chunk_resource(self, chunk_size: int, overlap: int = 0):
        chunker = TextChunker(self.data, chunk_size, overlap)
        self.chunks = chunker.chunk_text()

    def contextualize_chunk(self, chunk: Dict[str, Any]) -> str:
        if self.context_template:
            return self.context_template.format(
                chunk=chunk['text'],
                file=self.resource_path,
                start=chunk['start'],
                end=chunk['end']
            )
        else:
            return chunk['text']

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
