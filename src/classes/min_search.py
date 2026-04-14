from pydantic import BaseModel
from .mininal_source import MinimalSource


class MinimalSearchResults(BaseModel):
    question_id: str
    question_str: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str
