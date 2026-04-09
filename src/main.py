import bm25s
import json
import fire
from src.classes.indexer import Indexer


class CliCommands:

    def index(self, max_chunk_size: int = 2000):
        indexer: Indexer = Indexer(chunk_size=max_chunk_size)
        indexer.init_splitter()
        indexer.read_all_files()
        indexer.store()

        retriever = bm25s.BM25()
        corpus_tokens = bm25s.tokenize(indexer.corpus)
        retriever.index(corpus_tokens)
        retriever.save('data/processed/bm25_index/', corpus=indexer.corpus)

    def search(self, question: str, k: int, save_directory: str):
        retriever = bm25s.BM25.load('data/processed/bm25_index/')
        metadatas_chunks: list[dict[str, int | str]]
        with open('data/processed/chunks/chunks.json') as f:
            metadatas_chunks = json.load(f)
        query_tokens = bm25s.tokenize(question)
        docs, _ = retriever.retrieve(query_tokens, k=k)
        with open(save_directory + '/search_results.json', 'w') as f:
            final_list: list[dict[str, str | int]] = []
            for i in range(k):
                final_list.append(metadatas_chunks[docs[0, i]])
            json.dump(final_list, f, indent=2)

    # def truc(self, max_chunk_size: int = 2000):
    #     if len(sys.argv) > 2:
    #         print(sys.argv[2])
    #     print(max_chunk_size)
    # query = "What are the default values for FP8_MIN and FP8_MAX constants in vLLM's triton_flash_attention module?"


if __name__ == "__main__":
    # try:
    fire.Fire(CliCommands)
    # except Exception as e:
    #     print(e)
