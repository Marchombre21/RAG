import bm25s
import sys
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

    # def truc(self, max_chunk_size: int = 2000):
    #     if len(sys.argv) > 2:
    #         print(sys.argv[2])
    #     print(max_chunk_size)
    # query = "What are the default values for FP8_MIN and FP8_MAX constants in vLLM's triton_flash_attention module?"
    # query_tokens = bm25s.tokenize(query)
    # docs, scores = retriever.retrieve(query_tokens, k=2)
    # print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")
    # print(f"Dans le fichier {indexer.metadatas_chunks[docs[0, 0]]}")


if __name__ == "__main__":
    # try:
    fire.Fire(CliCommands)
    # except Exception as e:
    #     print(e)
