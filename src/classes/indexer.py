import os
import json
from ast import parse, FunctionDef, ClassDef, get_source_segment
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .mininal_source import MinimalSource
from pydantic import BaseModel, Field, PrivateAttr


class Indexer(BaseModel):
    __start_id: int = PrivateAttr(-1)
    __end_id: int = PrivateAttr()
    __chunk: str = PrivateAttr("")
    __text_splitter: RecursiveCharacterTextSplitter = PrivateAttr()
    chunk_size: int = Field(le=2000)
    __corpus: list[str] = PrivateAttr(default_factory=list)
    __metadatas_chunks: list[MinimalSource] =\
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
    def start_id(self, new_start: int) -> None:
        self.__start_id = new_start

    @property
    def corpus(self) -> list[str]:
        return self.__corpus

    # @corpus.setter
    # def corpus(self, new_start: list[str]) -> None:
    #     self.__corpus = new_start

    @property
    def metadatas_chunks(self) -> list[MinimalSource]:
        return self.__metadatas_chunks

    # @metadatas_chunks.setter
    # def metadatas_chunks(self, new_start: int) -> None:
    #     self.__metadatas_chunks = new_start

    @property
    def end_id(self) -> int:
        return self.__end_id

    @end_id.setter
    def end_id(self, new_end: int) -> None:
        self.__end_id = new_end

    @property
    def chunk(self) -> str:
        return self.__chunk

    @chunk.setter
    def chunk(self, new_chunk: str) -> None:
        self.__chunk = new_chunk

    def split_text(self, file_content: str, file: str):
        if self.start_id == -1:
            self.start_id = 0
        texts: list[Document] = self.text_splitter.\
            create_documents([file_content])
        offset: int = 1 if len(texts[-1].page_content) == self.chunk_size\
            else 0

        for n in range(len(texts) - (2 - offset)):
            self.end_id = self.start_id + len(texts[n].page_content)
            self.metadatas_chunks.append(
                MinimalSource(
                    file_path=file,
                    first_character_index=self.start_id,
                    last_character_index=self.end_id - 1,
                    chunk=file_content[self.start_id:self.end_id]))
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
                            MinimalSource(
                                file_path=file,
                                first_character_index=self.start_id,
                                last_character_index=self.end_id - 1,
                                chunk=file_content[self.start_id:self.end_id])
                        )
                        self.corpus.append(file_content[self.start_id:
                                                        self.end_id])

                    self.chunk = get_source_segment(file_content, node)
                    if len(self.chunk) > self.chunk_size:
                        self.split_text(self.chunk, file)
                    else:
                        self.start_id = file_content.find(self.chunk)
                        self.end_id = self.start_id + len(self.chunk)
                        self.metadatas_chunks.append(
                            MinimalSource(
                                file_path=file,
                                first_character_index=self.start_id,
                                last_character_index=self.end_id - 1,
                                chunk=file_content[self.start_id:self.end_id]))
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
                            MinimalSource(
                                file_path=file,
                                first_character_index=self.start_id,
                                last_character_index=self.end_id - 1,
                                chunk=file_content[self.start_id:self.end_id]))
                        self.corpus.append(file_content[self.start_id:
                                                        self.end_id])
        except SyntaxError:
            self.split_text(file_content, file)

    def read_all_files(self) -> None:

        for root, dirs, files in os.walk("data/vllm-0.10.1/"):
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
                    if file.endswith('.md') or file.endswith('.txt'):
                        file_size: int = len(file_content)
                        if file_size > self.chunk_size:
                            self.split_text(file_content, root + '/' + file)
                            if self.start_id != -1:
                                self.metadatas_chunks.\
                                    append(
                                        MinimalSource(
                                            file_path=root + '/' + file,
                                            first_character_index=self.
                                            start_id,
                                            last_character_index=self.end_id-1,
                                            chunk=file_content[self.start_id:
                                                               self.end_id]))
                                self.corpus.append(file_content[self.start_id:
                                                                self.end_id])
                                self.start_id = -1
                        else:
                            self.metadatas_chunks.append(
                                MinimalSource(
                                            file_path=root + '/' + file,
                                            first_character_index=0,
                                            last_character_index=len(
                                                file_content) - 1,
                                            chunk=file_content)
                                            )
                            self.corpus.append(file_content)

    def store(self):
        final_array: list[dict[str, int | str]] =\
            [chunk.model_dump() for chunk in self.metadatas_chunks]
        with open('data/processed/chunks/chunks.json', 'w') as f:
            json.dump(final_array, f)
