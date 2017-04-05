import logging
from nif_parser import NIFParser
import ahocorasick
import json
import io
import sys
import os
import pickle
from multiprocessing import Pool, Value
import time, datetime
import nif_utils


class SurfaceLinkMap:
    def __init__(self):
        self.__surfaces = []
        self.__surface_indices = {}
        self.__links = []
        self.__index = 0

    def add_surface_link(self, anchor, target):
        try:
            anchor_index = self.__surface_indices[anchor]
        except KeyError:
            anchor_index = self.__index
            self.__surfaces.append(anchor)
            self.__surface_indices[anchor] = anchor_index
            self.__index += 1

        if anchor_index < len(self.__links):
            try:
                self.__links[anchor_index][target] += 1
            except KeyError:
                self.__links[anchor_index][target] = 1
        else:
            self.__links.append({target: 1})

    def get_links(self):
        return self.__links

    def get_anchors(self):
        return self.__surfaces

    def get_anchor_indices(self):
        return self.__surface_indices


class TrieTextMatcher:
    def __init__(self, keys):
        logging.info("Building the Aho Corasick tree with %d provided keys." % (len(keys)))
        self.surface_dict = ahocorasick.Automaton()

        for key in keys:
            self.surface_dict.add_word(key, key)
        self.surface_dict.make_automaton()
        logging.info("Done building the keys.")

    def count_surfaces(self, text):
        matched_surfaces = {}

        for end_index, key in self.surface_dict.iter(text):
            try:
                matched_surfaces[key] += 1
            except KeyError:
                matched_surfaces[key] = 1

        return matched_surfaces


def search_context(path):
    all_surface_count = [0] * len(surface_indices)

    start = time.clock()
    for statements in NIFParser(path):
        for statement in statements:
            for index, count in surface_search(statement).iteritems():
                all_surface_count[index] += count
        sys.stdout.write("\r[%s] Searched %d articles." % (datetime.datetime.now().time(), context_counter.value))

    print("")
    print("Elapsed: %.2f" % (time.clock() - start))

    return all_surface_count


def search_context_parallel(path):
    all_surface_count = [0] * len(surface_indices)

    pool = Pool(4)

    start = time.clock()

    for statements in NIFParser(path):
        results = pool.map(surface_search, statements)
        # Store batch result from the workers.
        for r in results:
            for index, count in r.iteritems():
                all_surface_count[index] += count

        sys.stdout.write("\r[%s] Searched %d articles." % (datetime.datetime.now().time(), context_counter.value))

    print("")
    print("Elapsed: %.2f" % (time.clock() - start))

    return all_surface_count


def surface_search(data):
    global surface_matcher
    global surface_indices
    global context_counter

    s, v, o = data
    nif_type = nif_utils.get_resource_attribute(s, "nif")

    found_surfaces = {}

    if nif_type and nif_type == "context" and v.endswith("nif-core#isString"):
        for surface, count in surface_matcher.count_surfaces(o.encode('utf-8')).iteritems():
            index = surface_indices[surface]
            found_surfaces[index] = count

        with context_counter.get_lock():
            context_counter.value += 1

    return found_surfaces


def parse_links(path):
    parser = NIFParser(path)

    sl = SurfaceLinkMap()

    anchors = {}
    link_to = {}

    start = time.clock()

    count = 0
    for statements in parser:
        for s, v, o in statements:
            if v.endswith("#taIdentRef"):
                resource_name = nif_utils.get_resource_name(o)
                if s in anchors:
                    anchor = anchors.pop(s)
                    sl.add_surface_link(anchor.encode('utf-8'), resource_name)
                else:
                    link_to[s] = resource_name

            if v.endswith("nif-core#anchorOf"):
                if s in link_to:
                    resource_name = link_to.pop(s)
                    sl.add_surface_link(o.encode('utf-8'), resource_name)
                else:
                    anchors[s] = o

            count += 1
            sys.stdout.write("\r[%s] Parsed %d lines." % (datetime.datetime.now().time(), count))

    print("")
    print("Elapsed: %.2f" % (time.clock() - start))

    parser.close()

    return sl


def div_or_nan(numerator, divisor):
    return float('nan') if divisor == 0 else numerator * 1.0 / divisor


def write_as_json(surfaces, surface_links, surface_text_count, output_path):
    readme = {"surface": 0, "appearance": 1, "num_linked": 2, "link_prob": 3, "targets": 4}

    with io.open(output_path, mode='w', encoding='utf-8') as f:
        f.write(json.dumps(readme).decode('utf-8'))
        f.write(u"\n")

    count = 0
    with io.open(output_path, mode='a', encoding='utf-8') as f:
        for index, surface in enumerate(surfaces):
            surface_info = []

            links = surface_links[index]
            num_appearance = surface_text_count[index]

            num_linked = 0

            surface_info.append(surface)
            surface_info.append(num_appearance)

            for link, link_count in links.iteritems():
                num_linked += link_count

            surface_info.append(num_linked)
            surface_info.append(div_or_nan(num_linked, num_appearance))

            targets = {}
            for link, link_count in links.iteritems():
                targets[link] = (link_count, div_or_nan(link_count, num_linked))
            surface_info.append(targets)

            f.write(json.dumps(surface_info).decode('utf-8'))
            f.write(u"\n")

            count += 1
            sys.stdout.write("\rWrote %d surfaces." % count)
        print("")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logging.info('Started scanning Wiki links from NIF.')

    if len(sys.argv) != 6:
        print("Usage: scan_wiki_links.py [wiki_links_NIF] [wiki_abstract_NIF] [Output path] [Pickle path]")
        sys.exit(1)

    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-text-links_en.ttl.bz2"
    wiki_links = sys.argv[1]
    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-abstract-context_en.ttl.bz2"
    wiki_context = sys.argv[2]

    out_path = sys.argv[3]
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # surface_links.pickle
    surface_pickle_path = sys.argv[4]

    if os.path.exists(surface_pickle_path):
        logging.info("Loading wiki surface linking count from %s." % surface_pickle_path)
        surface_link_map = pickle.load(open(surface_pickle_path))
    else:
        logging.info("Reading raw wiki links from [%s]." % wiki_links)

        logging.info("Parsing wiki links.")
        surface_link_map = parse_links(wiki_links)
        logging.info("Done parsing wiki links.")

        logging.info("Pickling links as middle results.")
        with open(surface_pickle_path, 'w') as link_pickle_f:
            pickle.dump(surface_link_map, link_pickle_f)
        logging.info("Done writing down the links.")

    surface_matcher = TrieTextMatcher(surface_link_map.get_anchors())
    context_counter = Value('d', 0)
    surface_indices = surface_link_map.get_anchor_indices()

    logging.info("Parsing surface counts.")
    logging.info("Reading wiki context from [%s]." % wiki_context)
    surface_count = search_context_parallel(wiki_context)

    logging.info("Done parsing surface counts.")

    logging.info("Write out as JSON.")
    write_as_json(surface_link_map.get_anchors(), surface_link_map.get_links(), surface_count, out_path)

    logging.info('Finished.')
