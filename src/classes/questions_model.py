import uuid
from mininal_source import MinimalSource
from pydantic import BaseModel, Field


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str
