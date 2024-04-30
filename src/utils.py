import inspect
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

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