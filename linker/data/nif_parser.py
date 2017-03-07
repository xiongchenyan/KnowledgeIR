import rdflib
import bz2


class NIFParser:
    def __init__(self, url):
        self.__nif = bz2.BZ2File(url)
        self.__statements = []

    def __iter__(self):
        return self

    def parse_bz2_nif(self):
        for line in self.__nif:
            self.parse_nif(line)

    def parse_nif(self, data):
        # Currently we create a graph for each piece of data processed.
        __g = rdflib.Graph()
        __g.parse(data=data, format="n3")
        return __g

    def next(self):
        done = False

        while not self.__statements:
            try:
                graph = self.parse_nif(self.__nif.next())
                self.__statements = list(graph)
            except StopIteration as _:
                done = True

        if done:
            raise StopIteration

        return self.__statements.pop(0)

    def close(self):
        self.__nif.close()
