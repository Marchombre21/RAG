import bm25s
import os
import json
from transformers import Pipeline
from bm25s import BM25
from .classes import (MinimalAnswer, MinimalSearchResults, MinimalSource,
                      StudentSearchResults, StudentSearchResultsAndAnswer)


def get_retriever() -> tuple[BM25, list[dict[str, int | str]]]:
    retriever = BM25.load('data/processed/bm25_index/')
    metadatas_chunks: list[dict[str, int | str]]

    with open('data/processed/chunks/chunks.json') as f:
        metadatas_chunks = json.load(f)
    return (retriever, metadatas_chunks)


def write_output_search(min_search_res: list[MinimalSearchResults],
                        save_directory: str, k: int) -> None:

    stud_search_res: StudentSearchResults = StudentSearchResults(
        search_results=min_search_res, k=k)
    os.makedirs(save_directory, exist_ok=True)
    with open(save_directory + '/dataset_docs_public.json', 'w') as f:
        f.write(stud_search_res.model_dump_json(indent=2))
    print(f"Saved student_search_results to {save_directory}/"
          "dataset_docs_public.json")


def write_output_answer(min_answer: list[MinimalAnswer], save_directory: str,
                        k: int) -> None:

    stud_answer: StudentSearchResultsAndAnswer =\
            StudentSearchResultsAndAnswer(
                search_results=min_answer, k=k)
    os.makedirs(save_directory, exist_ok=True)
    with open(save_directory + '/dataset_docs_public.json', 'w') as f:
        f.write(stud_answer.model_dump_json(indent=2))


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

    final_list: list[MinimalSource] = get_min_source(
        pack_datas=pack_datas, question=question, k=k)

    min_search_res: MinimalSearchResults = MinimalSearchResults(
        question_id=id, question_str=question, retrieved_sources=final_list)

    return min_search_res


def get_answer(question: str,
               final_list: list[MinimalSource],
               generator: Pipeline,
               id: str = 'q1') -> MinimalAnswer:

    context: str = '\n'.join([min_src.chunk for min_src in final_list])

    messages = [{
        "role":
        "system",
        "content":
        "You are an extraction tool.\nAnswer"
        "using EXACTLY the text from the Context.\nOutput the answer"
        "immediately."
    }, {
        "role":
        "user",
        "content":
        f"Context:\n{context}\n\nQuestion:\
            \n{question}\n"
    }, {
        "role": "system",
        "content": "**Answer**:\n"
    }]

    prompt = generator.tokenizer.apply_chat_template(messages,
                                                     enable_thinking=False,
                                                     tokenize=False)

    preds = generator(text_inputs=prompt,
                      return_full_text=False,
                      max_length=None,
                      max_new_tokens=1024,
                      do_sample=False,
                      repetition_penalty=1.2)

    min_answer: MinimalAnswer = MinimalAnswer(
        question_id=id,
        question_str=question,
        retrieved_sources=final_list,
        answer=preds[0]['generated_text'].split("**Answer**:")[-1].strip())

    return min_answer
