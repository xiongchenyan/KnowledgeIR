import rdflib
import bz2


def parse_nif(data):
    # Currently we create a graph for each piece of data processed.
    __g = rdflib.Graph()
    __g.parse(data=data, format="n3")
    return __g


def parse_nif_as_list(data):
    return list(parse_nif(data))


class NIFParser:
    def __init__(self, url, batch=True):
        self.__url = url

        self.__batch_size = 100

        if self.__url.endswith(".bz2"):
            self.__nif = bz2.BZ2File(self.__url)
        else:
            self.__nif = open(self.__url)

        if batch:
            self.next = self.batch_read
        else:
            self.next = self.read

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        # Python 3 compatibility.
        pass

    def next(self):
        pass

    def batch_read(self):
        lines = self.__nif.readlines(self.__batch_size)
        if not lines:
            raise StopIteration
        return parse_nif_as_list("".join(lines))

    def read(self):
        return parse_nif_as_list(self.__nif.next())

    def close(self):
        self.__nif.close()
