from pydantic import BaseModel
from typing import Sequence
from .min_search import MinimalSearchResults, MinimalAnswer


class StudentSearchResults(BaseModel):
    search_results: Sequence[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: Sequence[MinimalAnswer]
