import os
import json
from tqdm import tqdm
from .errors import ImpossibleStoreError, IndexerError
from ast import parse, FunctionDef, ClassDef, get_source_segment
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .mininal_source import MinimalSource
from pydantic import BaseModel, Field, PrivateAttr


class Indexer(BaseModel):
    """This class handle the split into chunks
    """
    __start_id: int = PrivateAttr(0)
    __end_id: int = PrivateAttr(0)
    __chunk: str | None = PrivateAttr("")
    __text_splitter: RecursiveCharacterTextSplitter = PrivateAttr()
    chunk_size: int = Field(le=2000)
    __corpus: list[str] = PrivateAttr(default_factory=list)
    __metadatas_chunks: list[MinimalSource] =\
        PrivateAttr(default_factory=list)

    def init_splitter(self) -> None:
        overlap_size: int = int(self.chunk_size * 0.1)
        self.__text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=overlap_size)

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

    @property
    def metadatas_chunks(self) -> list[MinimalSource]:
        return self.__metadatas_chunks

    @property
    def end_id(self) -> int:
        return self.__end_id

    @end_id.setter
    def end_id(self, new_end: int) -> None:
        self.__end_id = new_end

    @property
    def chunk(self) -> str | None:
        return self.__chunk

    @chunk.setter
    def chunk(self, new_chunk: str | None) -> None:
        self.__chunk = new_chunk

    def split_text(self, file_content: str, file: str, end: int) -> None:
        """Split the text with RecursiveCharacterTextSplitter class from
        langchain librairie

        Args:
            file_content (str): The file content
            file (str): The file path
            end (int): The last index of the part of text we want to truncate
        """

        text_to_split: str = file_content[self.start_id:end]
        texts: list[Document] = self.text_splitter.\
            create_documents([text_to_split])
        offset: int = self.start_id
        local_id: int = 0

        for n in range(len(texts)):
            chunk_text: str = texts[n].page_content
            local_start: int = text_to_split.find(chunk_text, local_id)
            local_end: int = local_start + len(chunk_text)

            abs_start: int = offset + local_start
            abs_end: int = offset + local_end - 1

            self.add_meta(file, abs_start, abs_end, chunk_text)
            local_id = local_start

    def parse_py(self, file_content: str, file: str) -> None:
        try:
            self.chunk = ''
            self.start_id = 0
            self.end_id = 0

            # Parse function split all file content into nodes which represents
            # all logical parts of file (classes, functions, commentaries,
            # etc...)
            file_tree = parse(file_content)
            curr_search_index: int = 0
            for i, node in enumerate(file_tree.body):

                if isinstance(node, ClassDef) or isinstance(node, FunctionDef):

                    if self.chunk:
                        self.add_meta(file, self.start_id, self.end_id - 1,
                                      self.chunk)
                        curr_search_index = self.end_id
                    self.chunk = get_source_segment(file_content, node)
                    if self.chunk is None:
                        raise IndexerError()
                    self.start_id = file_content.find(self.chunk,
                                                      curr_search_index)
                    self.end_id = self.start_id + len(self.chunk)
                    if len(self.chunk) > self.chunk_size:
                        self.split_text(file_content, file, self.end_id)
                    else:
                        self.end_id = self.start_id + len(self.chunk) - 1
                        self.add_meta(file, self.start_id, self.end_id,
                                      self.chunk)
                    self.chunk = ''

                else:

                    # I retrieve the text that makes up the node
                    text_to_add: str | None = get_source_segment(
                        file_content, node)
                    if text_to_add is None:
                        raise IndexerError()

                    # I retrieve the index of the first and last characters of
                    # the text to be stored
                    node_start_id: int = file_content.find(
                        text_to_add, curr_search_index)
                    node_end_id: int = node_start_id + len(text_to_add)

                    # If there is already text to be stored and adding this
                    # text would cause the chunk to exceed its maximum size, I
                    # store the text that has already been saved and clear the
                    # self.chunk
                    if self.chunk and (node_end_id - self.start_id)\
                            > self.chunk_size:
                        self.add_meta(file, self.start_id, self.end_id - 1,
                                      self.chunk)
                        curr_search_index = self.start_id + len(self.chunk)
                        self.chunk = ''

                    # If this text is the first in the sequence, I set
                    # `start_id` to the index of the first character in this
                    # text.
                    if not self.chunk:
                        self.start_id = node_start_id
                    self.end_id = node_end_id
                    self.chunk = file_content[self.start_id:self.end_id]
                    if len(self.chunk) > self.chunk_size:
                        self.split_text(file_content, file, self.end_id)
                        curr_search_index = self.end_id
                        self.chunk = ''
                    elif i == len(file_tree.body) - 1:
                        self.add_meta(file, self.start_id, self.end_id - 1,
                                      self.chunk)
                        self.chunk = ''
                    else:
                        curr_search_index = self.end_id
        except SyntaxError:
            self.start_id = 0
            self.split_text(file_content, file, len(file_content))

    def add_meta(self, file_path: str, first_char: int, last_char: int,
                 text: str) -> None:
        """Adds the chunk to the metadata that will be stored and used later
        """
        self.metadatas_chunks.append(
            MinimalSource(file_path=file_path,
                          first_character_index=first_char,
                          last_character_index=last_char,
                          chunk=text))
        new_path: str = file_path.split('/')[-1].replace('.py', '')
        self.corpus.append('Keywords:' + new_path + '\n' + text)

    def read_all_files(self) -> None:

        file_counter: int = 0
        for _ in os.walk("data/raw/"):
            file_counter += 1
        if file_counter == 0:
            raise ImpossibleStoreError()
        for root, _, files in tqdm(os.walk("data/raw/"),
                                   total=file_counter,
                                   unit='files'):
            for file in files:
                if (file.endswith(".py") or file.endswith(".md")
                        or file.endswith(".txt")):
                    with open(root + "/" + file, "r") as f:
                        file_content: str = f.read()
                    if file.endswith(".py"):
                        self.parse_py(file_content, root + "/" + file)
                    if file.endswith('.md') or file.endswith('.txt'):
                        file_size: int = len(file_content)
                        if file_size > self.chunk_size:
                            self.split_text(file_content, root + '/' + file,
                                            file_size)
                        else:
                            self.add_meta(root + '/' + file, 0,
                                          len(file_content) - 1, file_content)
                    self.start_id = 0

    def store(self) -> None:
        """Store all metadatas in a file in chunks directory
        """
        final_array: list[dict[str, int | str]] =\
            [chunk.model_dump() for chunk in self.metadatas_chunks]
        os.makedirs('data/processed/chunks/', exist_ok=True)
        with open('data/processed/chunks/chunks.json', 'w') as f:
            json.dump(final_array, f, indent=2)
