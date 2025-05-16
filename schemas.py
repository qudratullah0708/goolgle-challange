from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str
    user_id: str

class EmbedRequest(BaseModel):
    user_id: str
