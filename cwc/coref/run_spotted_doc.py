import sys
import os
import json
import codecs
from operator import itemgetter
from cwc.utils import nlp_utils


def run_file(path, out_dir):
    text_dir = os.path.join(out_dir, "text")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    n = 100
    with open(path) as infile:
        for line in infile:
            all_spot_results = json.loads(line)
            title = all_spot_results['title']
            docno = all_spot_results['docno']
            body_text = all_spot_results['bodyText']

            title_spots = all_spot_results['spot']['title']
            body_spots = all_spot_results['spot']['bodyText']

            sentences = nlp_utils.reformmat_text(body_text)

            text_out_path = os.path.join(text_dir, docno)

            with codecs.open(text_out_path, 'w', "utf-8") as o:
                o.write(title)
                o.write("\n")
                for sent in sentences:
                    o.write(sent)
                    o.write("\n")

            n -= 1
            if n == 0:
                break


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
    run_file(sys.argv[1], sys.argv[2])
