import sys, os, time, datetime
from freebase_wiki_mapper import FreebaseWikiMapper
from nif_parser import NIFParser
from nif_utils import NifRelationCollector
import nif_utils, data_utils
import logging, urlparse

wiki_prefix = "http://en.wikipedia.org/wiki/"
dbpedia_prefix = "http://dbpedia.org/resource/"


class AnchorPositions:
    def __init__(self):
        self.__articles = []
        self.__article_indices = {}
        self.__index = 0
        self.__anchor_positions = []

    def add_surface_link(self, article, begin, end, target, anchor_text):
        try:
            article_index = self.__article_indices[article]
        except KeyError:
            article_index = self.__index
            self.__articles.append(article)
            self.__article_indices[article] = article_index
            self.__index += 1

        if article_index < len(self.__anchor_positions):
            self.__anchor_positions[article_index].append((begin, end, target, anchor_text))
        else:
            self.__anchor_positions.append([(begin, end, target, anchor_text)])

    def get_article_anchors(self, article_name):
        if article_name in self.__article_indices:
            article_index = self.__article_indices[article_name]
            return self.__anchor_positions[article_index]
        else:
            return None


def load_redirects(redirect_nif):
    collector = NifRelationCollector(
        "http://dbpedia.org/ontology/wikiPageRedirects",
    )
    redirect_to = {}
    count = 0
    for statements in NIFParser(redirect_nif):
        for s, v, o in statements:
            ready = collector.add_arg(s, v, o)

            if ready:
                count += 1
                from_page = s.replace(dbpedia_prefix, "")
                redirect_page = collector.pop(s)[
                    "http://dbpedia.org/ontology/wikiPageRedirects"
                ].replace(dbpedia_prefix, "")
                redirect_to[from_page] = redirect_page

                sys.stdout.write("\r[%s] Parsed %d lines." % (datetime.datetime.now().time(), count))

    sys.stdout.write("\nFinish loading redirects.")

    return redirect_to


def write_origin(context_nif, out_path):
    logging.info("Reading context string from %s." % context_nif)
    with open(out_path, 'w') as out:
        count = 0
        for statements in NIFParser(context_nif):
            for s, v, o in statements:
                nif_type = nif_utils.get_resource_attribute(s, "nif")
                if nif_type and nif_type == "context" and v.endswith("nif-core#isString"):
                    out.write(o.encode('utf-8'))
                    out.write("\n")
                    count += 1
                    sys.stdout.write("\r[%s] Wrote %d articles." % (datetime.datetime.now().time(), count))
    sys.stdout.write("\nFinish writing origin texts.")


def parse_anchor_position_info(info):
    begin_index = int(info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#beginIndex"])
    end_index = int(info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#endIndex"])
    uri = info["http://www.w3.org/2005/11/its/rdf#taIdentRef"].replace(dbpedia_prefix)
    anchor = info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf"].encode('utf-8')
    full_article_url = info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#referenceContext"]
    parsed_url = urlparse.urlparse(full_article_url)
    article = parsed_url[0] + ":/" + parsed_url[1] + parsed_url[2]

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
                begin, end, resource_name, anchor_text, article = parse_anchor_position_info(info)
                ap.add_surface_link(article, begin, end, resource_name, anchor_text)

            count += 1
            sys.stdout.write("\r[%s] Parsed %d lines." % (datetime.datetime.now().time(), count))

    print("")
    logging.info("Elapsed: %.2f" % (time.clock() - start))
    parser.close()

    return ap


def parse_context_string_info(info):
    uri = info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#sourceUrl"].replace(wiki_prefix, "")
    # text = info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString"].encode("UTF-8")
    text = unicode(info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString"])
    return uri, text


def find_fb_id(wiki_id, wiki_2_fb_map, redirects):
    target = wiki_id.encode('utf-8')

    if wiki_id in redirects:
        target = redirects[wiki_id]
        print "Found redirects for ", wiki_id, " as ", target

    if target in wiki_2_fb_map:
        fb_id = wiki_2_fb_map[target]
        return fb_id
    else:
        print "Cannot find target: ", target


def write_context_replaced(wiki_2_fb_map, context, article_anchors, redirects, out_path, both_version=False):
    wiki_context_nif = NIFParser(context)

    start = time.clock()
    article_count = 0

    seen_ids = set()
    anchor_count = 0
    wiki_missed_counter = {}

    anchor_miss_count = 0

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

                    positions = article_anchors.get_article_anchors(uri)

                    if not positions:
                        continue

                    positions = sorted(article_anchors.get_article_anchors(uri), reverse=True)

                    anchor_count += len(positions)

                    for begin, end, wiki_id in positions:
                        fb_id = find_fb_id(wiki_id, wiki_2_fb_map, redirects)
                        seen_ids.add(wiki_id)

                        if fb_id:
                            replacement = replacement[:begin] + fb_id + replacement[end:]
                        else:
                            print "Missing wiki id: ", wiki_id, " on page", uri, " at ", text[begin: end]
                            raw_input("Press Enter to continue...")

                            anchor_miss_count += 1
                            try:
                                wiki_missed_counter[wiki_id] += 1
                            except KeyError:
                                wiki_missed_counter[wiki_id] = 1

                    out.write(replacement.encode("utf-8"))

                    if both_version:
                        out.write(text.encode("utf-8"))

                    out.write("\n")

                    article_count += 1

                    missed_id_count = len(wiki_missed_counter)

                    total_id_referred = missed_id_count + len(seen_ids)

                    # sys.stdout.write("\r[%s] Wrote %d articles, "
                    #                  "%d/%d anchor misses (%.4f), "
                    #                  "%d/%d resource misses (%.4f)."
                    #                  % (datetime.datetime.now().time(), article_count,
                    #                     anchor_miss_count, anchor_count, 1.0 * anchor_miss_count / anchor_count,
                    #                     missed_id_count, total_id_referred, 1.0 * missed_id_count / total_id_referred))

    print("")
    logging.info("Elapsed: %.2f" % (time.clock() - start))
    wiki_context_nif.close()

    return len(seen_ids), anchor_count, wiki_missed_counter


def print_replacement_stats(num_wiki_seen, num_anchor, missed_counts, out_path):
    num_wiki_missed = len(missed_counts)

    num_wiki_referred = num_wiki_seen + num_wiki_missed
    with open(out_path, 'w') as stat_out:
        stat_out.write("Number of wiki resources seen: %s.\n" % num_wiki_seen)
        stat_out.write("Showing name and counts for missed resources:\n")
        num_anchor_missed = 0
        for resource, miss_count in missed_counts.iteritems():
            num_anchor_missed += miss_count
            stat_out.write("Wikipedia resource %s is not replaced successfully %d times.\n" % (resource, miss_count))
        stat_out.write("Percentage of resources missed: %.4f.\n" % (1.0 * num_wiki_missed / num_wiki_referred))
        stat_out.write("Percentage of anchors missed: %.4f.\n" % (1.0 * num_anchor_missed / num_anchor))


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

    # "/media/hdd/hdd0/data/freebase_wiki"
    output_dir = sys.argv[1]

    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-text-links_en.ttl.bz2"
    wiki_links = sys.argv[2]
    # "/media/hdd/hdd0/data/DBpedia/NIF_Abstract_Datasets/nif-abstract-context_en.ttl.bz2"
    wiki_context = sys.argv[3]

    # "/media/hdd/hdd0/data/Freebase/fb2w.nt"
    fb2w = sys.argv[4]

    logging.info("Mapping Freebase to Wikipedia.")
    mapper = FreebaseWikiMapper(output_dir)
    mapper.create_mapping(fb2w, "wikidatawiki_wb_items_per_site", "hector", "hector")
    wiki2fb = mapper.read_wiki_fb_mapping()
    logging.info("Done.")

    logging.info("Reading anchors.")
    anchor_positions = data_utils.run_or_load(os.path.join(output_dir, "anchor_positions.pickle"),
                                              parse_anchor_positions, wiki_links)
    logging.info("Done.")

    logging.info("Loading redirect pages.")
    redirects = load_redirects("/media/hdd/hdd0/data/DBpedia/201604_datasets/redirects_en.ttl.bz2")
    logging.info("Done")

    logging.info("Writing down the text.")
    write_origin(wiki_context, os.path.join(output_dir, "origin.txt"))
    num_wiki_seen, num_anchor, missed_counts = write_context_replaced(wiki2fb, wiki_context, anchor_positions,
                                                                      redirects,
                                                                      os.path.join(output_dir, "fb_replaced.txt"))
    write_context_replaced(wiki2fb, wiki_context, anchor_positions, redirects,
                           os.path.join(output_dir, "origin_and_replaced.txt"), True)

    print_replacement_stats(num_wiki_seen, num_anchor, missed_counts, os.path.join(output_dir, "replacement_stat"))

    logging.info("All Done.")


if __name__ == '__main__':
    main()
