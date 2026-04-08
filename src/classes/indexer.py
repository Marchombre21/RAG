import os
from ast import parse, FunctionDef, ClassDef, get_source_segment
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field, PrivateAttr


class Indexer(BaseModel):
    __start_id: int = PrivateAttr(-1)
    __end_id: int = PrivateAttr()
    __chunk: str = PrivateAttr("")
    __text_splitter: RecursiveCharacterTextSplitter = PrivateAttr()
    chunk_size: int = Field(2000, le=2000)
    __corpus: list[str] = PrivateAttr(default_factory=list)
    __metadatas_chunks: list[tuple[str, int, int]] =\
        PrivateAttr(default_factory=list)

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
    def corpus(self) -> int:
        return self.__corpus

    @corpus.setter
    def corpus(self, new_start: int):
        self.__corpus = new_start

    @property
    def metadatas_chunks(self) -> int:
        return self.__metadatas_chunks

    @metadatas_chunks.setter
    def metadatas_chunks(self, new_start: int):
        self.__metadatas_chunks = new_start

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

    # def add_chunk(self, file_content: str, index: tuple[int, int]):
    #     start: int
    #     end: int
    #     start, end = index
    #     self.corpus.append(file_content[start:end])

    def split_text(self, file_content: str, file: str):
        if self.start_id == -1:
            self.start_id = 0
        texts: list[Document] = self.text_splitter.\
            create_documents([file_content])
        offset: int = 1 if len(texts[-1].page_content) == self.chunk_size\
            else 0

        for n in range(len(texts) - (2 - offset)):
            self.end_id = self.start_id + len(texts[n].page_content)
            self.metadatas_chunks.append((file, self.start_id,
                                          self.end_id - 1))
            # self.add_chunk(file_content, (self.start_id, self.end_id))
            self.corpus.append(file_content[self.start_id:self.end_id])
            self.start_id = self.end_id

        if not offset:
            self.end_id = self.start_id + len(texts[-1].page_content)
        else:
            self.start_id = -1

    def parse_py(self, file_content: str, file: str) -> None:
        try:
            file_tree = parse(file_content)
            for node in file_tree.body:
                if isinstance(node, ClassDef) or isinstance(node, FunctionDef):
                    if self.start_id != -1:
                        self.metadatas_chunks.append(
                            (file, self.start_id, self.end_id - 1)
                        )
                        self.corpus.append(file_content[self.start_id:
                                                        self.end_id])

                    self.chunk = get_source_segment(file_content, node)
                    self.start_id = file_content.find(self.chunk)
                    self.end_id = self.start_id + len(self.chunk)
                    self.metadatas_chunks.append((file, self.start_id,
                                                  self.end_id))
                    self.corpus.append(self.chunk)
                    self.start_id = -1

                else:
                    self.chunk = get_source_segment(file_content, node)
                    if self.start_id == -1:
                        self.start_id = file_content.find(self.chunk)
                    self.end_id = file_content.find(self.chunk, self.start_id)\
                        + len(self.chunk)
                    if (self.end_id - self.start_id) > self.chunk_size:
                        self.split_text(file_content[self.start_id:
                                                     self.end_id], file)
                    if self.end_id == len(file_content):
                        self.metadatas_chunks.append(
                            (file, self.start_id, self.end_id - 1)
                        )
                        self.corpus.append(file_content[self.start_id:
                                                        self.end_id])

        except SyntaxError:
            self.split_text(file_content, file)

    def read_all_files(self) -> None:

        for root, dirs, files in os.walk("vllm-0.10.1/"):
            # print('root', root)
            # print('dirs', dirs)
            # print('files', files)
            for file in files:
                if (
                    file.endswith(".py")
                    or file.endswith(".md")
                    or file.endswith(".txt")
                ):
                    with open(root + "/" + file, "r") as f:
                        file_content: str = f.read()
                    if file.endswith(".py"):
                        self.parse_py(file_content, root + "/" + file)
