from typing import List
from pydantic import BaseModel, Field

class Tool(BaseModel):
    name: str

class Agent(BaseModel):
    role: str
    goal: str
    persona: str
    tools: List[str] = Field(..., alias="tools")
    verbose: bool