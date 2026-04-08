import bm25s
from src.classes.indexer import Indexer


def main():

    indexer: Indexer = Indexer()
    indexer.init_splitter()
    indexer.read_all_files()
    retriever = bm25s.BM25()
    corpus_tokens = bm25s.tokenize(indexer.corpus)
    retriever.index(corpus_tokens)
    query = "What are the default values for FP8_MIN and FP8_MAX constants in vLLM's triton_flash_attention module?"
    query_tokens = bm25s.tokenize(query)
    docs, scores = retriever.retrieve(query_tokens, k=2)
    print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")
    print(f"Dans le fichier {indexer.metadatas_chunks[docs[0, 0]]}")


if __name__ == "__main__":
    # try:
    main()
    # except Exception as e:
    #     print(e)
