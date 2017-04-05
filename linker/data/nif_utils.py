import urlparse
import sys


class NifRelationCollector:
    def __init__(self, *relation_names):
        self.__target_fields = relation_names
        self.__validator = set(relation_names)
        self.__info = {}
        self.__count = {}

        sys.stdout.write("Will collection following relations:\n")
        for n in relation_names:
            sys.stdout.write("\t%s\n" % n)

    def add_arg(self, s, relation, o):
        r = str(relation)

        if r not in self.__validator:
            return False

        try:
            self.__info[s][r] = o
        except KeyError:
            self.__info[s] = {r: o}

        try:
            self.__count[s] += 1
        except KeyError:
            self.__count[s] = 1

        if self.__count[s] == len(self.__target_fields):
            return True
        else:
            return False

    def pop(self, s):
        if self.__count[s] == len(self.__target_fields):
            self.__count.pop(s)
            return self.__info.pop(s)


def get_resource_name(url):
    parsed = urlparse.urlparse(url)
    return parsed.path.split("/")[-1]


def get_resource_attribute(url, param_name):
    parsed = urlparse.urlparse(url)
    return urlparse.parse_qs(parsed.query)[param_name][0]
