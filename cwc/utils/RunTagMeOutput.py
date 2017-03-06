import sys
import json


def run_file(path):
    with open(path) as input:
        for line in input:
            parts = line.split("\t")
            doc_id = parts[0]
            tagged_json = parts[1]
            tagged_results = json.load(tagged_json)


if __name__ == '__main__':
    path = sys.argv[1]
