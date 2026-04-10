import bm25s
import json
import fire
from src.classes.stud_search import StudentSearchResults
from src.classes.rag_model import RagDataset
from src.classes.min_search import MinimalSource, MinimalSearchResults
from src.classes.indexer import Indexer


class CliCommands:

    def write_ouput_search(self, min_search_res: list[MinimalSearchResults],
                           save_directory: str, k: int) -> None:
        stud_search_res: StudentSearchResults = StudentSearchResults(
            search_results=min_search_res, k=k)
        with open(save_directory + '/search_results.json', 'w') as f:
            f.write(stud_search_res.model_dump_json(indent=2))

    def get_search_res(self, question: str, k: int, id: str = 'q1')\
            -> MinimalSearchResults:

        retriever = bm25s.BM25.load('data/processed/bm25_index/')
        metadatas_chunks: list[dict[str, int | str]]

        with open('data/processed/chunks/chunks.json') as f:
            metadatas_chunks = json.load(f)
        query_tokens = bm25s.tokenize(question)
        docs, _ = retriever.retrieve(query_tokens, k=k)
        final_list: list[MinimalSource] = []

        for i in range(k):
            final_list.append(
                MinimalSource.model_validate(metadatas_chunks[docs[0, i]]))

        min_search_res: MinimalSearchResults = MinimalSearchResults(
            question_id=id, question=question,
            retrieved_sources=final_list)

        return min_search_res

    def index(self, max_chunk_size: int = 2000):
        indexer: Indexer = Indexer(chunk_size=max_chunk_size)
        indexer.init_splitter()
        indexer.read_all_files()
        indexer.store()

        retriever = bm25s.BM25()
        corpus_tokens = bm25s.tokenize(indexer.corpus)
        retriever.index(corpus_tokens)
        retriever.save('data/processed/bm25_index/', corpus=indexer.corpus)

    def search(self, question: str, k: int, save_directory: str) -> None:

        # retriever = bm25s.BM25.load('data/processed/bm25_index/')
        # metadatas_chunks: list[dict[str, int | str]]
        # with open('data/processed/chunks/chunks.json') as f:
        #     metadatas_chunks = json.load(f)
        # query_tokens = bm25s.tokenize(question)
        # docs, _ = retriever.retrieve(query_tokens, k=k)
        # final_list: list[MinimalSource] = []
        # for i in range(k):
        #     final_list.append(
        #         MinimalSource.model_validate(metadatas_chunks[docs[0, i]]))
        # min_search_res: MinimalSearchResults = MinimalSearchResults(
        #     question_id='q1', question=question,
        #     retrieved_sources=final_list)
        # stud_search_res: StudentSearchResults = StudentSearchResults(
        #     search_results=[min_search_res], k=k)
        min_search_res: MinimalSearchResults = self.get_search_res(question, k)

        self.write_ouput_search([min_search_res], save_directory, k)

    def search_dataset(self, dataset_path: str, k: int, save_directory: str)\
            -> None:

        list_min_search: list[MinimalSearchResults] = []
        with open(dataset_path, 'r') as f:
            quest_dict: RagDataset = RagDataset.model_validate(json.load(f))
        for question in quest_dict.rag_questions:
            list_min_search.append(self.get_search_res(question.question, k,
                                                       question.question_id))
        self.write_ouput_search(list_min_search, save_directory, k)


if __name__ == "__main__":
    # try:
    fire.Fire(CliCommands)
    # except Exception as e:
    #     print(e)
