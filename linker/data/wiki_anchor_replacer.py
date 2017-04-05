import sys, os, time, datetime
from freebase_wiki_mapper import FreebaseWikiMapper
from nif_parser import NIFParser
from nif_utils import NifRelationCollector
import nif_utils

wiki_prefix = "http://en.wikipedia.org/wiki/"


class AnchorPositions:
    def __init__(self):
        self.__articles = []
        self.__article_indices = {}
        self.__index = 0
        self.__anchor_positions = []

    def add_surface_link(self, article, begin, end, target):
        try:
            article_index = self.__article_indices[article]
        except KeyError:
            article_index = self.__index
            self.__articles.append(article)
            self.__article_indices[article] = article_index
            self.__index += 1

        if article_index < len(self.__anchor_positions):
            self.__anchor_positions[article_index].append((begin, end, target))
        else:
            self.__anchor_positions.append([(begin, end, target)])

    def get_article_anchors(self, article_name):
        article_index = self.__article_indices[article_name]
        return self.__anchor_positions[article_index]


def write_origin(context_nif, out_path):
    with open(out_path, 'w') as out:
        for statements in context_nif:
            for s, v, o in statements:
                nif_type = nif_utils.get_resource_attribute(s, "nif")
                if nif_type and nif_type == "context" and v.endswith("nif-core#isString"):
                    out.write(o.encode('utf-8'))


def parse_links(links):
    parser = NIFParser(links)

    ap = AnchorPositions()

    anchors = {}
    link_to = {}
    begins = {}
    ends = {}

    start = time.clock()

    count = 0

    nif_relation_collector = NifRelationCollector(
        "http://www.w3.org/2005/11/its/rdf#taIdentRef",
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf",
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#beginIndex",
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#endIndex"
    )

    for statements in parser:
        for s, v, o in statements:

            if v.endswith("rdf#taIdentRef"):

                print v
                print v.__class__

                sys.stdin.readline()

                resource_name = nif_utils.get_resource_name(o)
                if s in anchors:
                    anchor = anchors.pop(s)
                else:
                    link_to[s] = resource_name
            elif v.endswith("nif-core#anchorOf"):
                if s in link_to:
                    resource_name = link_to.pop(s)
                else:
                    anchors[s] = o
            elif v.endswith("nif-core#beginIndex"):
                print o
            elif v.endswith("nif-core#endIndex"):
                print o

            count += 1
            sys.stdout.write("\r[%s] Parsed %d lines." % (datetime.datetime.now().time(), count))

            print("")
            print("Elapsed: %.2f" % (time.clock() - start))

    parser.close()
    return ap


def write_context_replaced(wiki_2_fb_map, context, anchor_positions, out_path):
    wiki_context_nif = NIFParser(context)

    article_ids = {}
    context_texts = {}

    context_info_count = {}

    with open(out_path, 'w') as out:
        for statements in wiki_context_nif:
            for s, v, o in statements:
                nif_type = nif_utils.get_resource_attribute(s, "nif")
                if nif_type and nif_type == "context":
                    if v.endswith("nif-core#isString"):
                        context_texts[s] = o.encode('utf-8')

                        # TODO write this as a class to collect info.
                        try:
                            context_info_count[s] += 1
                        except KeyError:
                            context_info_count[s] = 1

                        if context_info_count[s] == 2:
                            anchor_positions.get_article_anchors(article_ids[s])

                    elif v.endswith("nif-core#sourceUrl"):
                        article_ids[s] = str(o).replace(wiki_prefix, "")

                        try:
                            context_info_count[s] += 1
                        except KeyError:
                            context_info_count[s] = 1



def write_both(wiki_2_fb_map, context, anchor_positions, out_path):
    pass


if __name__ == '__main__':
    mapper_dir = sys.argv[1]

    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-text-links_en.ttl.bz2"
    wiki_links = sys.argv[2]
    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-abstract-context_en.ttl.bz2"
    wiki_context = sys.argv[3]

    output_dir = sys.argv[4]

    mapper = FreebaseWikiMapper(mapper_dir)
    # wiki_context_nif = NIFParser(wiki_context)
    # wiki_links_nif = NIFParser(wiki_links)

    anchor_positions = parse_links(wiki_links)

    write_origin(wiki_context, os.path.join(output_dir, "origin.txt"))
    write_context_replaced(mapper, wiki_context, anchor_positions, os.path.join(output_dir, "fb_replaced.txt"))
    write_context_replaced(mapper, wiki_context, anchor_positions,
                           os.path.join(output_dir, "origin_and_replaced.txt"))
