import bm25s
import os
import json
from json import JSONDecodeError
from typing import Any
from ollama import chat, ChatResponse
from bm25s import BM25
from .classes import (MinimalAnswer, RetrieveError, MinimalSearchResults,
                      MinimalSource, StudentSearchResults,
                      StudentSearchResultsAndAnswer)


def get_retriever() -> tuple[BM25, list[dict[str, int | str]]]:
    """This function instantiates a BM25 object that will have loaded the
    indexes saved during data indexing. Then I retrieve the chunks in a python
    object and return both.
    """

    try:
        retriever: BM25 = BM25.load('data/processed/bm25_index/')
        metadatas_chunks: list[dict[str, int | str]]

        with open('data/processed/chunks/chunks.json') as f:
            metadatas_chunks = json.load(f)
    except FileNotFoundError:
        raise RetrieveError()
    return (retriever, metadatas_chunks)


def write_output_search(min_search_res: list[MinimalSearchResults],
                        save_directory: str, file_name: str, k: int) -> None:
    """I instantiate a StudentSearchResults object and save it to a JSON file

    Args:
        min_search_res (list[MinimalSearchResults]): All chunks usefull to
        find the answer
        k (int): The number of chunks we kept
    """

    stud_search_res: StudentSearchResults = StudentSearchResults(
        search_results=min_search_res, k=k)
    os.makedirs(save_directory, exist_ok=True)
    with open(save_directory + '/' + file_name, 'w') as f:
        f.write(stud_search_res.model_dump_json(indent=2))
    print(
        f"Saved student_search_results to {save_directory + '/' + file_name}")


def write_output_answer(min_answer: list[MinimalAnswer], save_directory: str,
                        file_name: str, k: int) -> None:

    stud_answer: StudentSearchResultsAndAnswer =\
            StudentSearchResultsAndAnswer(
                search_results=min_answer, k=k)
    os.makedirs(save_directory, exist_ok=True)
    with open(save_directory + file_name, 'w') as f:
        f.write(stud_answer.model_dump_json(indent=2))
    print(f'Saved answers in {save_directory + file_name}')


def get_min_source(
    pack_datas: tuple[BM25, list[dict[str, int | str]]],
    question: str,
    k: int,
) -> list[MinimalSource]:

    retriever: BM25
    metadatas_chunks: list[dict[str, int | str]]
    retriever, metadatas_chunks = pack_datas
    query_tokens = bm25s.tokenize(question)
    docs, _ = retriever.retrieve(query_tokens, k=k)
    final_list: list[MinimalSource] = []

    for i in range(k):
        final_list.append(
            MinimalSource.model_validate(metadatas_chunks[docs[0, i]]))

    return final_list


def get_search_res(question: str,
                   k: int,
                   pack_datas: tuple[BM25, list[dict[str, int | str]]],
                   id: str = 'q1') -> MinimalSearchResults:

    final_list: list[MinimalSource] = get_min_source(pack_datas=pack_datas,
                                                     question=question,
                                                     k=k)

    min_search_res: MinimalSearchResults = MinimalSearchResults(
        question_id=id, question_str=question, retrieved_sources=final_list)

    return min_search_res


def get_answer(question: str,
               final_list: list[MinimalSource],
               id: str = 'q1') -> MinimalAnswer:
    context: str = '\n'.join([min_src.chunk for min_src in final_list])

    messages = [{
        "role":
        "system",
        "content":
        "You are an extraction tool.\nAnswer"
        " using EXACTLY the text from the Context.\nOutput the answer"
        " immediately"
    }, {
        "role":
        "user",
        "content":
        f"Context:\n{context}\n\nQuestion:\
            \n{question}\n"
    }]

    response: ChatResponse = chat(model='qwen3:0.6b',
                                  messages=messages,
                                  options={
                                      'num_predict': 1024,
                                      'temperature': 0.0
                                  })

    if response.message.content:
        message: str = response.message.content.split('</think>')[-1].split(
            '**Answer**')[-1].strip()
    else:
        message = 'No answer'

    min_answer: MinimalAnswer = MinimalAnswer(question_id=id,
                                              question_str=question,
                                              retrieved_sources=final_list,
                                              answer=message)

    return min_answer


def check_cache(question: str,
                cache_file: dict[str, Any],
                k: str = '10',
                id: str = 'q1') -> MinimalAnswer | None:

    key_str: str = f'{question.lower()}_{k}'
    if key_str in cache_file:
        response: dict[str, Any] = cache_file[key_str]
        min_answer: MinimalAnswer = MinimalAnswer(
            question_id=id,
            question_str=question,
            retrieved_sources=[
                MinimalSource.model_validate(src)
                for src in response['retrieved_sources']
            ],
            answer=response['answer'])
        return min_answer
    else:
        return None


def get_cache() -> dict[str, Any]:
    if not os.path.exists('data/cache/cache.json'):
        os.makedirs('data/cache/', exist_ok=True)
        with open('data/cache/cache.json', 'w') as f:
            f.write("{}")

    with open('data/cache/cache.json', 'r') as f:
        try:
            cache_file: dict[str, Any] = json.load(f)
        except JSONDecodeError:
            print('The cache was cleared after an error was detected inside'
                  ' it.')
            cache_file = {}
    return cache_file
