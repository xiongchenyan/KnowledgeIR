import sys
import os
import json
import codecs
from operator import itemgetter
from cwc.utils import nlp_utils
import argparse
from cwc.coref.coref_engine_runner import BerkeleyEntityCoref, StanfordEntityCoref


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_dataset(path, out_dir):
    text_dir = os.path.join(out_dir, "text")

    create_dir(out_dir)
    create_dir(os.path.join(out_dir, "text"))
    create_dir(os.path.join(out_dir, "preprocessed"))
    create_dir(os.path.join(out_dir, "joint"))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    with open(path) as infile:
        for line in infile:
            all_spot_results = json.loads(line)
            title = all_spot_results['title']
            docno = all_spot_results['docno']
            body_text = all_spot_results['bodyText']

            formatted_text = nlp_utils.reformat_text(body_text)

            text_out_path = os.path.join(text_dir, docno)

            with codecs.open(text_out_path, 'w', "utf-8") as o:
                o.write(title)
                o.write("\n")
                o.write(formatted_text)

    return text_dir


def run_coref(input_path, out_dir, engine_type):
    if "berkeley" == engine_type:
        engine = BerkeleyEntityCoref()
    elif "stanford" == engine_type:
        engine = StanfordEntityCoref()
    else:
        raise RuntimeError("Unknown engine type.")

    text_dir = prepare_dataset(input_path, out_dir)
    engine.run_directory(text_dir, out_dir)


def get_entity_replaced_text(text, spots):
    tokens = text.split()

    sorted_spots = []

    for spot in spots:
        sorted_spots.append((tuple(spot['loc']), spot))

    sorted_spots.sort(key=itemgetter(0), reverse=True)

    for (begin, end), spot in sorted_spots:
        entities = spot['entities']
        tokens = tokens[0:begin] + [entities[0]['id']] + tokens[end:]

    return " ".join(tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run coreference for some text.')
    parser.add_argument('-i', '--input', help='The input path.')
    parser.add_argument('-o', '--output', help='The output directory.')
    parser.add_argument('-e', '--engine', choices=['berkeley', 'stanford'], help='The coreference engine type.')
    args = parser.parse_args()

    run_coref(args.input, args.output, args.engine)
