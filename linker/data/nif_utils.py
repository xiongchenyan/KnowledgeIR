import urlparse


class NifRelationCollector:
    def __init__(self, *args):
        self.__target_fields = args
        self.__info = {}
        self.__count = {}

    def add_arg(self, key, field_name, value):
        self.__info[key][field_name] = value

        try:
            self.__count[key] = 1
        except KeyError:
            self.__count[key] += 1

        if self.__count[key] == len(self.__target_fields):
            return True
        else:
            return False

    def pop(self, key):
        if self.__count[key] == len(self.__target_fields):
            self.__count.pop(key)
            return self.__info.pop(key)


def get_resource_name(url):
    parsed = urlparse.urlparse(url)
    return parsed.path.split("/")[-1]


def get_resource_attribute(url, param_name):
    parsed = urlparse.urlparse(url)
    return urlparse.parse_qs(parsed.query)[param_name][0]
