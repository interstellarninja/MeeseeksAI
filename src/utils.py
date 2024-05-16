import ast
import datetime
import importlib
import inspect
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import sys
from typing import List
import xml.etree.ElementTree as ET
import requests

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from unstructured.partition.html import partition_html


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
script_dir = os.path.dirname(os.path.abspath(__file__))
now = datetime.datetime.now()
log_folder = os.path.join(script_dir, "inference_logs")
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(
    log_folder, f"function-calling-inference_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
)
# Use RotatingFileHandler from the logging.handlers module
file_handler = RotatingFileHandler(log_file_path, maxBytes=0, backupCount=0)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d:%H:%M:%S")
file_handler.setFormatter(formatter)

inference_logger = logging.getLogger("function-calling-inference")
inference_logger.addHandler(file_handler)

def get_tool_names():
    # Get all the classes defined in the script
    classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    # Extract the class names excluding the imported ones
    class_names = [cls[0] for cls in classes if cls[1].__module__ == 'src.tools']

    return class_names

def get_fewshot_examples(num_fewshot):
    """return a list of few shot examples"""
    example_path = os.path.join(script_dir, 'prompt_assets', 'few_shot.json')
    with open(example_path, 'r') as file:
        examples = json.load(file)  # Use json.load with the file object, not the file path
    if num_fewshot > len(examples):
        raise ValueError(f"Not enough examples (got {num_fewshot}, but there are only {len(examples)} examples).")
    return examples[:num_fewshot]

def validate_and_extract_tool_calls(assistant_content):
    validation_result = False
    tool_calls = []
    error_message = None
    scratchpad_text = None

    try:
        # Use regular expression to find the text within <scratchpad> tags
        scratchpad_pattern = r'<scratchpad>(.*?)</scratchpad>'
        scratchpad_match = re.search(scratchpad_pattern, assistant_content, re.DOTALL)
        if scratchpad_match:
            scratchpad_text = scratchpad_match.group(1).strip()

        # Use regular expression to find all <tool_call> tags and their contents
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, assistant_content, re.DOTALL)

        if not tool_call_matches:
            error_message = None
        else:
            for match in tool_call_matches:
                json_text = match.strip()

                try:
                    json_data = json.loads(json_text)
                    tool_calls.append(json_data)
                    validation_result = True
                except json.JSONDecodeError as json_err:
                    error_message = f"JSON parsing failed:\n"\
                                    f"- JSON Decode Error: {json_err}\n"\
                                    f"- Problematic JSON text: {json_text}"
                    inference_logger.error(error_message)
                    continue

    except Exception as err:
        error_message = f"Error during tool call extraction: {err}"
        inference_logger.error(error_message)

    return validation_result, tool_calls, scratchpad_text, error_message

def extract_json_from_markdown(text):
    """
    Extracts the JSON string from the given text using a regular expression pattern.
    
    Args:
        text (str): The input text containing the JSON string.
        
    Returns:
        dict: The JSON data loaded from the extracted string, or None if the JSON string is not found.
    """
    json_pattern = r'```json\r?\n(.*?)\r?\n```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON string: {e}")
    else:
        print("JSON string not found in the text.")
    return None

def embedding_search(url, query):
    text = download_form_html(url)
    elements = partition_html(text=text)
    content = "\n".join([str(el) for el in elements])
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([content])

    # Load a pre-trained sentence transformer model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS index and retriever
    index = FAISS.from_documents(docs, embedding_model)
    retriever = index.as_retriever()

    answers = retriever.invoke(query, top_k=4)
    chunks = []
    for i, doc in enumerate(answers):
        chunk = f"\n<chunk index={i}>\n{doc.page_content}\n</chunk>\n"
        chunks.append(chunk)

    result = "".join(chunks)
    return f"<documents>\n{result}</documents>"

def download_form_html(url):
    headers = {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
      'Accept-Encoding': 'gzip, deflate, br',
      'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
      'Cache-Control': 'max-age=0',
      'Dnt': '1',
      'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
      'Sec-Ch-Ua-Mobile': '?0',
      'Sec-Ch-Ua-Platform': '"macOS"',
      'Sec-Fetch-Dest': 'document',
      'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Site': 'none',
      'Sec-Fetch-User': '?1',
      'Upgrade-Insecure-Requests': '1',
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    return response.text