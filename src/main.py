import bm25s
import json
import os
import fire
from tqdm import tqdm
from transformers import pipeline, Pipeline
from bm25s import BM25
from src.classes import (StudentSearchResults, RagDataset, MinimalSource,
                         MinimalSearchResults, MinimalAnswer, Indexer)
from .utils import (get_answer, get_min_source, get_retriever,
                    write_output_answer, write_output_search, get_search_res)


class CliCommands:

    @staticmethod
    def index(max_chunk_size: int = 2000):
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

        write_output_search([min_search_res], save_directory, k)

    @staticmethod
    def search_dataset(dataset_path: str, k: int, save_directory: str)\
            -> None:

        pack_datas: tuple[BM25, list[dict[str, int | str]]] =\
            get_retriever()
        list_min_search: list[MinimalSearchResults] = []
        with open(dataset_path, 'r') as f:
            quest_dict: RagDataset = RagDataset.model_validate(json.load(f))
        for question in quest_dict.rag_questions:
            list_min_search.append(
                get_search_res(question.question, k, pack_datas,
                               question.question_id))
        write_output_search(list_min_search, save_directory, k)

    @staticmethod
    def answer(question: str,
               k: int,
               save_directory: str = 'data/output/answer_results') -> None:

        final_list: list[MinimalSource] = get_min_source(
            pack_datas=get_retriever(), question=question, k=k)

        generator: Pipeline = pipeline('text-generation',
                                       'Qwen/Qwen3-0.6B')

        min_answer: MinimalAnswer = get_answer(question=question,
                                               final_list=final_list,
                                               generator=generator)

        write_output_answer([min_answer], save_directory, k)

    @staticmethod
    def answer_dataset(student_search_results_path: str, save_directory: str):

        list_min_answer: list[MinimalAnswer] = []
        generator: Pipeline = pipeline('text-generation',
                                       'Qwen/Qwen3-0.6B')
        with open(student_search_results_path, 'r') as f:
            stud_search_res: StudentSearchResults =\
                StudentSearchResults.model_validate(json.load(f))
            for search in tqdm(stud_search_res.search_results):
                list_min_answer.append(
                    get_answer(question=search.question,
                               final_list=search.retrieved_sources,
                               generator=generator,
                               id=search.question_id))

        write_output_answer(list_min_answer, save_directory, stud_search_res.k)

    @staticmethod
    def evaluate(student_answer_path: str,
                 dataset_path: str,
                 k: int = 10,
                 max_context_length: int = 2000):
        os.system(
            "./src/moulinette-ubuntu evaluate_student_search_results"
            f" --student_answer_path {student_answer_path}"
            f" --dataset_path {dataset_path} --k {k} --max_context_length"
            f" {max_context_length}")


if __name__ == "__main__":
    # try:
    fire.Fire(CliCommands)
    # except Exception as e:
    #     print(e)
