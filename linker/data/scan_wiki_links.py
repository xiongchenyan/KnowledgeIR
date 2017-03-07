import logging
from nif_parser import NIFParser
import urlparse
import ahocorasick
import json
import io
import sys
import os


class SurfaceLinks:
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


class SurfaceMatcher:
    def __init__(self, keys):
        logging.info("Building the surface Aho Corasick tree with the provided keys.")
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


def get_resource_name(url):
    parsed = urlparse.urlparse(url)
    return parsed.path.split("/")[-1]


def get_resource_attribute(url, param_name):
    parsed = urlparse.urlparse(url)
    return urlparse.parse_qs(parsed.query)[param_name][0]


def search_in_context(path, matcher, surface_indices):
    parser = NIFParser(path)

    all_surface_count = [0] * len(surface_indices)

    count = 0
    for s, v, o in parser:
        nif_type = get_resource_attribute(s, "nif")
        if nif_type and nif_type == "context":
            for surface, count in matcher.count_surfaces(o.encode('utf-8')).iteritems():
                index = surface_indices[surface]
                all_surface_count[index] += count
        count += 1
        sys.stdout.write("\rSearched %d lines." % count)

    return all_surface_count


def parse_links(path):
    parser = NIFParser(path)

    surface_links = SurfaceLinks()

    anchors = {}
    link_to = {}

    count = 0
    for s, v, o in parser:
        if v.endswith("#taIdentRef"):
            resource_name = get_resource_name(o)
            if s in anchors:
                anchor = anchors.pop(s)
                surface_links.add_surface_link(anchor.encode('utf-8'), resource_name)
            else:
                link_to[s] = resource_name

        if v.endswith("nif-core#anchorOf"):
            if s in link_to:
                resource_name = link_to.pop(s)
                surface_links.add_surface_link(o.encode('utf-8'), resource_name)
            else:
                anchors[s] = o
        count += 1
        sys.stdout.write("\rParsed %d lines." % count)

    return surface_links


def write_as_json(surfaces, surface_links, surface_count, out_path):
    parent_dir = os.path.dirname(out_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    readme = {"surface": 0, "appearance": 1, "num_linked": 2, "link_prob": 3, "targets": 4}

    with io.open(out_path, mode='w', encoding='utf-8') as f:
        f.write(json.dumps(readme).decode('utf-8'))
        f.write(u"\n")

    count = 0
    with io.open(out_path, mode='a', encoding='utf-8') as f:
        for index, surface in enumerate(surfaces):
            surface_info = []

            links = surface_links[index]
            num_appearance = surface_count[index]

            num_linked = 0

            surface_info.append(surface)
            surface_info.append(num_appearance)

            for link, link_count in links.iteritems():
                num_linked += link_count

            surface_info.append(num_linked)
            surface_info.append(num_linked * 1.0 / num_appearance)

            targets = {}
            for link, link_count in links.iteritems():
                targets[link] = (link_count, link_count * 1.0 / num_linked)
            surface_info.append(targets)

            f.write(json.dumps(surface_info).decode('utf-8'))
            f.write(u"\n")

            count += 1
            sys.stdout.write("\rWrote %d surfaces." % count)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logging.info('Started scanning Wiki links from NIF.')

    if len(sys.argv) != 4:
        print("Usage: scan_wiki_links.py [wiki_links_NIF] [wiki_abstract_NIF] [Output path]")
        sys.exit(1)

    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-text-links_en.ttl.bz2"
    wiki_links = sys.argv[1]
    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-abstract-context_en.ttl.bz2"
    wiki_context = sys.argv[2]

    logging.info("Reading wiki links from [%s]." % wiki_links)
    logging.info("Reading wiki context from [%s]." % wiki_context)

    logging.info("Parsing wiki links.")
    surface_links = parse_links(wiki_links)
    logging.info("Done parsing wiki links.")

    matcher = SurfaceMatcher(surface_links.get_anchors())

    logging.info("Parsing surface counts.")
    surface_count = search_in_context(wiki_context, matcher, surface_links.get_anchor_indices())
    logging.info("Done parsing surface counts.")

    logging.info("Write out as JSON.")
    write_as_json(surface_links.get_anchors(), surface_links.get_links(), surface_count, sys.argv[3])

    logging.info('Finished.')
