import os
from ast import parse, FunctionDef, ClassDef, get_source_segment
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, PrivateAttr


class Indexer(BaseModel):
    __start_id: int = PrivateAttr(-1)
    __end_id: int = PrivateAttr()
    __chunk: str = PrivateAttr("")
    __text_splitter: RecursiveCharacterTextSplitter = PrivateAttr()
    chunk_size: int = Field(2000, le=2000)
    corpus: list[str]
    metadatas_chunks: list[tuple[str, int, int]]

    def init_splitter(self) -> None:
        self.__text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size
        )

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        return self.__text_splitter

    @property
    def start_id(self) -> int:
        return self.__start_id

    @start_id.setter
    def start_id(self, new_start: int):
        self.__start_id = new_start

    @property
    def end_id(self) -> int:
        return self.__end_id

    @end_id.setter
    def end_id(self, new_end: int):
        self.__end_id = new_end

    @property
    def chunk(self) -> int:
        return self.__chunk

    @chunk.setter
    def chunk(self, new_chunk: int):
        self.__chunk = new_chunk

    def add_chunk(self, file_content: str, index: tuple[int, int]):
        start: int
        end: int
        start, end = index
        self.corpus.append(file_content[start:end])

    def split_text(self, file_content: str, file: str):
        texts: list[str] = self.text_splitter.create_documents([file_content])
        for n in range(len(texts) - 2):
            self.end_id = self.start_id + len(texts[n])
            self.metadatas_chunks.append((file, self.start_id, self.end_id - 1))
            self.add_chunk(file_content, (self.start_id, self.end_id))
            self.start_id = self.end_id
        if len(texts[-1]) == self.chunk_size:
            self.end_id = self.start_id + self.chunk_size
            self.metadatas_chunks.append((file, self.start_id, self.end_id - 1))
            self.add_chunk(file_content, (self.start_id, self.end_id))
            self.start_id = -1
        else:
            self.chunk = texts[-1]
            self.end_id = self.start_id + len(texts[n])

    def parse_py(self, file_content: str, file: str) -> None:
        try:
            file_tree = parse(file_content)
        except SyntaxError:

        for node in file_tree.body:
            if isinstance(node, ClassDef) or\
                    isinstance(node, FunctionDef):
                if start_id != -1:
                    self.metadatas_chunks.append((file, start_id, end_id - 1))
                    self.add_chunk(file_content, (start_id, end_id))

                chunk = get_source_segment(file_content, node)
                start_id = file_content.find(chunk)
                end_id = start_id + len(chunk)
                self.metadatas_chunks.append((file, start_id, end_id))
                self.corpus.append(chunk)
                start_id = -1

            else:
                chunk = get_source_segment(file_content, node)
                if start_id == -1:
                    start_id = file_content.find(chunk)
                end_id = file_content.find(chunk, start_id) +\
                    len(chunk)
                if (end_id - start_id) > self.chunk_size:
                    self.split_text(file_content[start_id:end_id])
                if end_id == len(file_content):
                    self.metadatas_chunks.append((file, start_id, end_id - 1))
                    self.add_chunk(file_content, (start_id, end_id))


    def read_all_files(self) -> None:

        for root, dirs, files in os.walk("vllm-0.10.1/"):
            # print('root', root)
            # print('dirs', dirs)
            # print('files', files)
            for file in files:
                if file.endswith(".py") or file.endswith(".md") or\
                        file.endswith(".txt"):
                    with open(root + '/' + file, "r") as f:
                        file_content: str = f.read()
                    if file.endswith(".py"):
                        