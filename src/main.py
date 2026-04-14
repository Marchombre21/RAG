import bm25s
import json
import os
import fire
from transformers import pipeline, Pipeline
from bm25s import BM25
from src.classes.stud_search import (StudentSearchResults,
                                     StudentSearchResultsAndAnswer)
from src.classes.rag_model import RagDataset
from src.classes.min_search import (MinimalSource, MinimalSearchResults,
                                    MinimalAnswer)
from src.classes.indexer import Indexer


class CliCommands:

    @staticmethod
    def get_retriever() -> tuple[BM25, list[dict[str, int | str]]]:
        retriever = BM25.load('data/processed/bm25_index/')
        metadatas_chunks: list[dict[str, int | str]]

        with open('data/processed/chunks/chunks.json') as f:
            metadatas_chunks = json.load(f)
        return (retriever, metadatas_chunks)

    @staticmethod
    def write_output_search(min_search_res: list[MinimalSearchResults],
                            save_directory: str, k: int) -> None:

        stud_search_res: StudentSearchResults = StudentSearchResults(
            search_results=min_search_res, k=k)
        os.makedirs(save_directory, exist_ok=True)
        with open(save_directory + '/dataset_docs_public.json', 'w') as f:
            f.write(stud_search_res.model_dump_json(indent=2))

    @staticmethod
    def write_output_answer(min_answer: list[MinimalAnswer],
                            save_directory: str, k: int) -> None:

        stud_answer: StudentSearchResultsAndAnswer =\
              StudentSearchResultsAndAnswer(
                search_results=min_answer, k=k)
        os.makedirs(save_directory, exist_ok=True)
        with open(save_directory + '/dataset_docs_public.json', 'w') as f:
            f.write(stud_answer.model_dump_json(indent=2))

    @staticmethod
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

    def get_search_res(self,
                       question: str,
                       k: int,
                       pack_datas: tuple[BM25, list[dict[str, int | str]]],
                       id: str = 'q1') -> MinimalSearchResults:

        final_list: list[MinimalSource] = self.get_min_source(
            pack_datas=pack_datas, question=question, k=k)

        min_search_res: MinimalSearchResults = MinimalSearchResults(
            question_id=id, question=question, retrieved_sources=final_list)

        return min_search_res

    def get_answer(self,
                   question: str,
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
            question=question,
            retrieved_sources=final_list,
            answer=preds[0]['generated_text'].split("**Answer**:")[-1].strip())

        return min_answer

    def index(self, max_chunk_size: int = 2000):
        indexer: Indexer = Indexer(chunk_size=max_chunk_size)
        indexer.init_splitter()
        indexer.read_all_files()
        indexer.store()

        retriever = BM25()
        corpus_tokens = bm25s.tokenize(indexer.corpus)
        retriever.index(corpus_tokens)
        retriever.save('data/processed/bm25_index/', corpus=indexer.corpus)

    def search(self, question: str, k: int, save_directory: str) -> None:

        min_search_res: MinimalSearchResults = self.get_search_res(
            question, k, self.get_retriever())

        self.write_output_search([min_search_res], save_directory, k)

    def search_dataset(self, dataset_path: str, k: int, save_directory: str)\
            -> None:

        pack_datas: tuple[BM25, list[dict[str, int | str]]] =\
            self.get_retriever()
        list_min_search: list[MinimalSearchResults] = []
        with open(dataset_path, 'r') as f:
            quest_dict: RagDataset = RagDataset.model_validate(json.load(f))
        for question in quest_dict.rag_questions:
            list_min_search.append(
                self.get_search_res(question.question, k, pack_datas,
                                    question.question_id))
        self.write_output_search(list_min_search, save_directory, k)

    def answer(self,
               question: str,
               k: int,
               save_directory: str = 'data/output/answer_results') -> None:

        final_list: list[MinimalSource] = self.get_min_source(
            pack_datas=self.get_retriever(), question=question, k=k)

        generator: Pipeline = pipeline('text-generation', 'Qwen/Qwen3-0.6B')

        min_answer: MinimalAnswer = self.get_answer(question=question,
                                                    final_list=final_list,
                                                    generator=generator)

        self.write_output_answer([min_answer], save_directory, k)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str):

        list_min_answer: list[MinimalAnswer] = []
        generator: Pipeline = pipeline('text-generation', 'Qwen/Qwen3-0.6B')
        with open(student_search_results_path, 'r') as f:
            stud_search_res: StudentSearchResults =\
                StudentSearchResults.model_validate(json.load(f))
            for search in stud_search_res.search_results:
                list_min_answer.append(
                    self.get_answer(question=search.question,
                                    final_list=search.retrieved_sources,
                                    generator=generator,
                                    id=search.question_id))

        self.write_output_answer(list_min_answer, save_directory,
                                 stud_search_res.k)


if __name__ == "__main__":
    # try:
    fire.Fire(CliCommands)
    # except Exception as e:
    #     print(e)
