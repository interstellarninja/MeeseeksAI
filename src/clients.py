import os
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env")

class Clients:
    def __init__(self):
        self.ollama = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',
        )
        self.groq = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.anthropic = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

CLIENTS = Clients()
