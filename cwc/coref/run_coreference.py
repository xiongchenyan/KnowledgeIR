import sys
import os
import json
import codecs
from operator import itemgetter
from cwc.utils import nlp_utils
import argparse
from cwc.coref.coref_engine import BerkeleyEntityCoref, StanfordEntityCoref


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_dataset(path, out_dir):
    for docno, title, formatted_text in get_data(path):
        text_out_path = os.path.join(out_dir, "text", docno)
        with codecs.open(text_out_path, 'w', "utf-8") as o:
            o.write(title)
            o.write("\n")
            o.write(formatted_text)


def get_data(path):
    with open(path) as infile:
        for line in infile:
            all_spot_results = json.loads(line)
            title = all_spot_results['title']
            docno = all_spot_results['docno']
            body_text = all_spot_results['bodyText']

            formatted_text = nlp_utils.reformat_text(body_text)

            yield docno, title, formatted_text


def get_engine(engine_type):
    if "berkeley" == engine_type:
        engine = BerkeleyEntityCoref(engine_type)
    elif "stanford" == engine_type:
        engine = StanfordEntityCoref(engine_type)
    else:
        raise RuntimeError("Unknown engine type.")

    return engine


def run_coref(input_path, out_dir, engine):
    create_dir(out_dir)
    if engine.get_name() == "berkeley":
        # We process all data and use Berkeley to run on the directory.
        text_dir = os.path.join(out_dir, "text")
        create_dir(text_dir)
        create_dir(os.path.join(out_dir, "preprocessed"))
        create_dir(os.path.join(out_dir, "joint"))
        prepare_dataset(input_path, out_dir)
        engine.run_directory(text_dir, out_dir)
    elif engine.get_name() == "stanford":
        # Stanford can be run as a server so we could run each file incrementally.
        for docno, title, formatted_text in get_data(input_path):
            combined = title + "\n" + formatted_text
            engine.run_coref(combined)


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

    engine = get_engine(args.engine)
    run_coref(args.input, args.output, engine)
