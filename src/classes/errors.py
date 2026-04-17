class FilePathError(Exception):

    def __init__(self, path: str) -> None:
        message: str = f'\nERROR:\n{path} does not lead to any file.'
        super().__init__(message)


class RetrieveError(Exception):

    def __init__(self) -> None:
        message: str = '\nERROR:\nBe sure to index the data using the ‘index’'\
                ' command before searching for results.'
        super().__init__(message)


class ImpossibleStoreError(Exception):

    def __init__(self) -> None:
        message: str = '\nERROR:\nPlease put the files you want to index in'\
            ' the data/raw directory. Don\'t be THAT kind of people.'
        super().__init__(message)


class IndexerError(Exception):

    def __init__(self) -> None:
        message: str = '\nERROR:\nAn error occurs during indexing. Please'\
            'retry later.'
        super().__init__(message)


class RagError(Exception):

    def __init__(self) -> None:
        message: str = '\nERROR:\nNo Rag_questions key in JSON file.'
        super().__init__(message)
