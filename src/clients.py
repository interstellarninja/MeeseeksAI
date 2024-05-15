import os
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env")

class Clients:
    def __init__(self):
        self.clients = {}

    def initialize_ollama(self):
        self.clients["ollama"] = {
            "client": OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama',
            ),
            "model": os.getenv("OLLAMA_MODEL")
        }

    def initialize_groq(self):
        self.clients["groq"] = {
            "client": Groq(
                api_key=os.getenv('GROQ_API_KEY')
            ),
            "model": os.getenv("GROQ_MODEL")
        }

    def initialize_anthropic(self):
        self.clients["anthropic"] = {
            "client": Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
            "model": os.getenv("ANTHROPIC_MODEL")
        }

    def initialize_lmstudio(self):
        self.clients["lmstudio"] = {
            "client": OpenAI(
                base_url="http://localhost:1234/v1",
                #base_url="http://192.168.1.2:1234/v1",
                api_key="lm-studio"
            ),
            "model": os.getenv("LMSTUDIO_MODEL")
        }

    def initialize_localllama(self):
        from src.inference import ModelInference
        self.clients["localllama"] = {
            "client": ModelInference(
                model_path=os.getenv("LOCAL_MODEL_PATH"),
                load_in_4bit=os.getenv("LOAD_IN_4BIT", "False")
            ),
            "model": None
        }

    def chat_completion(self, client, messages):
        if client == "ollama":
            self.initialize_ollama()
        elif client == "groq":
            self.initialize_groq()
        elif client == "anthropic":
            self.initialize_anthropic()
        elif client == "lmstudio":
            self.initialize_lmstudio()
        elif client == "localllama":
            self.initialize_localllama()
        else:
            raise ValueError(f"Unsupported client: {client}")

        response = self.clients[client]["client"].chat.completions.create(
            model=self.clients[client]["model"],
            messages=messages,
        )
        completion = response.choices[0].message.content

        return completion

CLIENTS = Clients()

