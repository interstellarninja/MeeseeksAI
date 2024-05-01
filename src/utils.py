import ast
import datetime
import importlib
import inspect
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from typing import List

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