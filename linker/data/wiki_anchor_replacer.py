import sys, os, time, datetime
from freebase_wiki_mapper import FreebaseWikiMapper
from nif_parser import NIFParser
from nif_utils import NifRelationCollector
import nif_utils, data_utils
import logging

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
    logging.info("Reading context string from %s." % context_nif)
    with open(out_path, 'w') as out:
        count = 0
        for statements in NIFParser(context_nif):
            for s, v, o in statements:
                nif_type = nif_utils.get_resource_attribute(s, "nif")
                if nif_type and nif_type == "context" and v.endswith("nif-core#isString"):
                    out.write(o.encode('utf-8'))
                    count += 1
                    sys.stdout.write("\r[%s] Wrote %d articles." % (datetime.datetime.now().time(), count))


def parse_anchor_position_info(info):
    begin_index = int(info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#beginIndex"])
    end_index = int(info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#endIndex"])
    uri = nif_utils.get_resource_name(info["http://www.w3.org/2005/11/its/rdf#taIdentRef"])
    anchor = info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf"].encode('utf-8')
    article = nif_utils.get_resource_name(
        info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#referenceContext"])
    return begin_index, end_index, uri, anchor, article


def parse_anchor_positions(links):
    parser = NIFParser(links)

    ap = AnchorPositions()

    start = time.clock()
    count = 0

    nif_relation_collector = NifRelationCollector(
        "http://www.w3.org/2005/11/its/rdf#taIdentRef",
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf",
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#beginIndex",
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#endIndex",
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#referenceContext"
    )

    for statements in parser:
        for s, v, o in statements:

            ready = nif_relation_collector.add_arg(s, v, o)
            if ready:
                info = nif_relation_collector.pop(s)
                begin, end, resource_name, _, article = parse_anchor_position_info(info)
                ap.add_surface_link(article, begin, end, resource_name)

            count += 1
            sys.stdout.write("\r[%s] Parsed %d lines." % (datetime.datetime.now().time(), count))

    print("")
    logging.info("Elapsed: %.2f" % (time.clock() - start))
    parser.close()

    return ap


def parse_context_string_info(info):
    uri = nif_utils.get_resource_name(info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#sourceUrl"])
    text = info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#sourceUrl"].encode("UTF-8")
    return uri, text


def write_context_replaced(wiki_2_fb_map, context, anchor_positions, out_path, both_version=False):
    wiki_context_nif = NIFParser(context)

    start = time.clock()
    count = 0

    collector = NifRelationCollector(
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString",
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#sourceUrl"
    )

    with open(out_path, 'w') as out:
        for statements in wiki_context_nif:
            for s, v, o in statements:

                ready = collector.add_arg(s, v, o)

                if ready:
                    uri, text = parse_context_string_info(collector.pop(s))

                    replacement = text

                    positions = anchor_positions.get_article_anchors(uri)

                    sorted(positions)

                    for begin, end, wiki_id in positions:
                        m = wiki_2_fb_map.read_wiki_fb_mapping()

                        if wiki_id in m:
                            fb_id = m[wiki_id]
                            replacement[begin:end] = fb_id

                    out.write(text)

                    if both_version:
                        out.write(replacement)

                    count += 1
                    sys.stdout.write("\r[%s] Wrote %d articles." % (datetime.datetime.now().time(), count))

    print("")
    logging.info("Elapsed: %.2f" % (time.clock() - start))
    wiki_context_nif.close()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

    # "/media/hdd/hdd0/data/freebase_wiki"
    output_dir = sys.argv[1]

    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-text-links_en.ttl.bz2"
    wiki_links = sys.argv[2]
    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-abstract-context_en.ttl.bz2"
    wiki_context = sys.argv[3]

    # "/media/hdd/hdd0/data/Freebase/fb2w.nt"
    fb2w = sys.argv[4]

    # logging.info("Mapping Freebase to Wikipedia.")
    mapper = FreebaseWikiMapper(output_dir)
    # mapper.create_mapping(fb2w, "wikidatawiki_wb_items_per_site", "hector", "hector")
    # logging.info("Done.")

    logging.info("Reading anchors.")
    anchor_positions = data_utils.run_or_load(os.path.join(output_dir, "anchor_positions.pickle"),
                                              parse_anchor_positions, wiki_links)
    logging.info("Done.")

    logging.info("Writing down the text.")
    # write_origin(wiki_context, os.path.join(output_dir, "origin.txt"))
    write_context_replaced(mapper, wiki_context, anchor_positions, os.path.join(output_dir, "fb_replaced.txt"))
    write_context_replaced(mapper, wiki_context, anchor_positions,
                           os.path.join(output_dir, "origin_and_replaced.txt"), True)
    logging.info("Done.")
