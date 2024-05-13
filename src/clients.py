import os
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
from dotenv import load_dotenv
from src.inference import ModelInference

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
        self.lmstudio = OpenAI(
            base_url="http://localhost:1234/v1",
            #base_url="http://192.168.1.2:1234/v1",
            api_key="lm-studio"
        )
        self.localllama = ModelInference(
            model_path=os.getenv("LOCAL_MODEL_PATH"),
            load_in_4bit=os.getenv("LOAD_IN_4BIT", "False")
        )

    def chat_completion(self, client, messages):
        if client == "ollama":
            response = self.ollama.chat.completions.create(
                model=os.getenv("OLLAMA_MODEL"),
                messages=messages,
            )
            completion = response.choices[0].message.content

        elif client == "anthropic":
            response = self.anthropic.messages.create(
                model=os.getenv("ANTHROPIC_MODEL"),
                max_tokens=1000,
                temperature=0.5,
                messages=messages,
            )
            completion = response.content[0].text
            
        elif client == "groq":
            response = self.groq.chat.completions.create(
                model=os.getenv("GROQ_MODEL"),
                messages=messages,
            )
            completion = response.choices[0].message.content           

        elif client == "lmstudio":
            response = self.lmstudio.chat.completions.create(
                model=os.getenv("LMSTUDIO_MODEL"),
                messages=messages,
                temperature=0.7,
            )
            completion = response.choices[0].message.content
            
        elif client == "localllama":
            completion = self.localllama.run_inference(messages)
            
        else:
            raise ValueError(f"Unsupported client: {client}")
        
        return completion

CLIENTS = Clients()
