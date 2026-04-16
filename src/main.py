import bm25s
import json
import os
import fire
from typing import Any
from tqdm import tqdm
from pydantic import ValidationError
from pydantic_core import ErrorDetails
from src.classes.errors import RagError
from bm25s import BM25
from src.classes import (FilePathError, AnsweredQuestion, StudentSearchResults,
                         RagDataset, MinimalSource, MinimalSearchResults,
                         MinimalAnswer, Indexer)
from .utils import (get_answer, get_min_source, get_retriever,
                    write_output_answer, write_output_search, get_search_res)


class CliCommands:

    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
        indexer: Indexer = Indexer(chunk_size=max_chunk_size)
        indexer.init_splitter()
        indexer.read_all_files()
        indexer.store()

        retriever = BM25()
        corpus_tokens = bm25s.tokenize(indexer.corpus)
        retriever.index(corpus_tokens)
        retriever.save('data/processed/bm25_index/', corpus=indexer.corpus)
        print("Ingestion complete! Indices saved under data/processed/")

    @staticmethod
    def search(question: str, k: int, save_directory: str) -> None:

        min_search_res: MinimalSearchResults = get_search_res(
            question, k, get_retriever())

        write_output_search([min_search_res], save_directory,
                            question + '.json', k)

    @staticmethod
    def search_dataset(dataset_path: str, k: int, save_directory: str)\
            -> None:

        pack_datas: tuple[BM25, list[dict[str, int | str]]] =\
            get_retriever()
        list_min_search: list[MinimalSearchResults] = []
        try:
            with open(dataset_path, 'r') as f:
                python_object: dict[str, Any] = json.load(f)
                quest_dict: RagDataset = RagDataset.model_validate(
                    python_object)
                if 'UnansweredQuestions' not in dataset_path:
                    for question in python_object.get('rag_questions', ['']):
                        if question:
                            AnsweredQuestion.model_validate(question)
                        else:
                            raise RagError()
        except FileNotFoundError:
            raise FilePathError(dataset_path)
        for question in quest_dict.rag_questions:
            list_min_search.append(
                get_search_res(question.question, k, pack_datas,
                               question.question_id))
        file_name: str = dataset_path.split('/')[-1]
        write_output_search(list_min_search, save_directory, file_name, k)

    @staticmethod
    def answer(question: str,
               k: int = 10,
               save_directory: str = 'data/output/answer_results') -> None:

        final_list: list[MinimalSource] = get_min_source(
            pack_datas=get_retriever(), question=question, k=k)

        min_answer: MinimalAnswer = get_answer(question=question,
                                               final_list=final_list)

        write_output_answer([min_answer], save_directory,
                            '/single_answer.json', k)

    @staticmethod
    def answer_dataset(student_search_results_path: str,
                       save_directory: str) -> None:

        list_min_answer: list[MinimalAnswer] = []
        try:
            with open(student_search_results_path, 'r') as f:
                stud_search_res: StudentSearchResults =\
                    StudentSearchResults.model_validate(json.load(f))
                size_list: int = len(stud_search_res.search_results)
                print(f'Loaded {size_list} questions'
                      f' from {student_search_results_path}')
                for search in tqdm(stud_search_res.search_results):
                    list_min_answer.append(
                        get_answer(question=search.question_str,
                                   final_list=search.retrieved_sources,
                                   id=search.question_id))
                print(f'Processed {size_list} of {size_list} questions.')
        except FileNotFoundError:
            raise FilePathError(student_search_results_path)

        file_name: str = student_search_results_path.split('/')[-1]
        write_output_answer(list_min_answer, save_directory, '/' + file_name,
                            stud_search_res.k)

    @staticmethod
    def evaluate(student_answer_path: str,
                 dataset_path: str,
                 k: int = 10,
                 max_context_length: int = 2000) -> None:
        os.system(
            "./src/moulinette-ubuntu evaluate_student_search_results"
            f" --student_answer_path {student_answer_path}"
            f" --dataset_path {dataset_path} --k {k} --max_context_length"
            f" {max_context_length}")


if __name__ == "__main__":
    try:
        fire.Fire(CliCommands)
    except ValidationError as e:
        errors: list[ErrorDetails] = e.errors()
        for error in errors:
            print(f'A Pydantic error of type {error["type"]} occurs.'
                  f' {error["msg"]} in {error["loc"][-1]} attribut.')
    except Exception as e:
        print(e)
