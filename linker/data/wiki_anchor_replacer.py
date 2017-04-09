import datetime
import logging
import os
import sys
import time
import codecs

import data_utils
import nif_utils
from freebase_wiki_mapper import FreebaseWikiMapper
from nif_parser import NIFParser
from nif_utils import NifRelationCollector

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

    def get_articles(self):
        return self.__articles


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
    uri = info["http://www.w3.org/2005/11/its/rdf#taIdentRef"].replace(dbpedia_prefix, '')
    anchor = info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf"].encode('utf-8')
    full_article_url = info["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#referenceContext"]
    article = nif_utils.strip_url_params(full_article_url).replace(dbpedia_prefix, '')

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
                sys.stdout.write("\r[%s] Collected from %d articles." % (datetime.datetime.now().time(), count))

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
        # print "Using redirect %s in place of %s" %(target, wiki_id)

    if target in wiki_2_fb_map:
        fb_id = wiki_2_fb_map[target]
        return fb_id


def do_replace(text, begin, end, replacement, expected_text):
    status = 0

    if text[begin:end].encode('utf-8') == expected_text:
        text = text[:begin] + replacement + text[end:]
    else:
        # print "Text not matching (%s != %s), doing left search." %(text_at_span, expected_text)
        for left_offset in range(1, 11):
            # Conduct left search. If succeed, status will be 1, if not , will be 2
            new_begin = begin - left_offset
            new_end = end - left_offset

            if new_begin < 0:
                break

            fragment = text[new_begin: new_end].encode('utf-8')

            # print "Trying new fragment ", fragment

            if expected_text == fragment:
                # print "Matches using %d:%d instead" % (new_begin, new_end)
                text = text[:new_begin] + replacement + text[new_end:]
                status = 1
                return status, text

        for right_offset in range(1, 6):
            new_begin = begin + right_offset
            new_end = end + right_offset

            if new_end > len(text):
                break

            fragment = text[new_begin: new_end].encode('utf-8')

            if expected_text == fragment:
                text = text[:new_begin] + replacement + text[new_end:]
                status = 2
                return status, text

        status = 3

    return status, text


def write_context_replaced(wiki_2_fb_map, context, article_anchors, redirects, error_log, out_path, both_version=False):
    wiki_context_nif = NIFParser(context)

    start = time.clock()
    article_count = 0

    seen_ids = set()
    anchor_count = 0
    wiki_missed_counter = {}

    anchor_miss_count = 0
    anchor_left_search_count = 0
    anchor_right_search_count = 0
    anchor_replace_failures = 0

    collector = NifRelationCollector(
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString",
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#sourceUrl"
    )

    with open(out_path, 'w') as out, codecs.open(error_log, 'w', 'utf-8') as error:
        for statements in wiki_context_nif:
            for s, v, o in statements:
                ready = collector.add_arg(s, v, o)

                if ready:
                    article_name, text = parse_context_string_info(collector.pop(s))
                    replaced_text = text

                    positions = article_anchors.get_article_anchors(article_name)

                    if not positions:
                        continue

                    positions = sorted(article_anchors.get_article_anchors(article_name), reverse=True)

                    anchor_count += len(positions)

                    for begin, end, wiki_id, anchor_text in positions:
                        fb_id = find_fb_id(wiki_id, wiki_2_fb_map, redirects)
                        seen_ids.add(wiki_id)

                        if fb_id:
                            status, replaced_text = do_replace(replaced_text, begin, end, fb_id, anchor_text)

                            if status == 1:
                                error.write('[Warning] %s replaces %s at [%d:%d] on page %s, done by left search.\n' % (
                                    fb_id, anchor_text.decode('utf-8'), begin, end, article_name))
                                anchor_left_search_count += 1
                            elif status == 2:
                                error.write('[Warning] %s replaces %s at [%d:%d] on page %s, done by right search.\n' % (
                                    fb_id, anchor_text.decode('utf-8'), begin, end, article_name))
                                anchor_right_search_count += 1

                            elif status == 3:
                                error.write('[Warning] %s cannot replace %s at [%d:%d] on page %s.\n' % (
                                    fb_id, anchor_text.decode('utf-8'), begin, end, article_name))
                                # sys.stdout.write('[Warning] %s cannot replace %s at [%d:%d] on page %s, '
                                #                  'origin span maps to %s\n' % (
                                #                      fb_id, anchor_text.decode('utf-8'), begin, end, article_name,
                                #                      text[begin:end].encode('utf-8')))
                                # sys.stdout.write(replaced_text.encode('utf-8'))
                                #
                                # raw_input("Wait.")

                                anchor_replace_failures += 1
                        else:
                            # print "Missing wiki id: ", wiki_id, " on page", article_name, " at ", text[begin: end]
                            error.write('[Warning] Missing wiki id: %s on page %s at [%d:%d].\n' % (
                                wiki_id, article_name, begin, end))
                            # raw_input("Press Enter to continue...")

                            anchor_miss_count += 1
                            try:
                                wiki_missed_counter[wiki_id] += 1
                            except KeyError:
                                wiki_missed_counter[wiki_id] = 1

                    out.write(replaced_text.encode("utf-8"))

                    if both_version:
                        out.write(text.encode("utf-8"))

                    out.write("\n")

                    article_count += 1

                    missed_id_count = len(wiki_missed_counter)

                    total_id_referred = missed_id_count + len(seen_ids)

                    sys.stdout.write("\r[%s] Wrote %d articles, "
                                     "%d/%d anchor misses (%.4f), "
                                     "%d/%d resource misses (%.4f), "
                                     "%d/%d anchor replaced with left search (%.4f), "
                                     "%d/%d anchor replaced with right search (%.4f), "
                                     "%d/%d anchor replace failures (%.4f)."
                                     % (datetime.datetime.now().time(), article_count,
                                        anchor_miss_count, anchor_count, 1.0 * anchor_miss_count / anchor_count,
                                        missed_id_count, total_id_referred, 1.0 * missed_id_count / total_id_referred,
                                        anchor_left_search_count, anchor_count,
                                        1.0 * anchor_left_search_count / anchor_count,
                                        anchor_right_search_count, anchor_count,
                                        1.0 * anchor_right_search_count / anchor_count,
                                        anchor_replace_failures, anchor_count,
                                        1.0 * anchor_replace_failures / anchor_count))

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

    # "/media/hdd/hdd0/data/DBpedia/201604_datasets/redirects_en.ttl.bz2"
    redirect_path = sys.argv[5]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    redirects = data_utils.run_or_load(os.path.join(output_dir, "redirects.pickle"), load_redirects, redirect_path)
    logging.info("Done")

    logging.info("Writing down the text.")
    write_origin(wiki_context, os.path.join(output_dir, "origin.txt"))
    num_wiki_seen, num_anchor, missed_counts = write_context_replaced(wiki2fb, wiki_context, anchor_positions,
                                                                      redirects,
                                                                      os.path.join(output_dir, "fb_replace.log"),
                                                                      os.path.join(output_dir, "fb_replaced.txt"))
    write_context_replaced(wiki2fb, wiki_context, anchor_positions, redirects,
                           os.path.join(output_dir, "fb_replace_both.log"),
                           os.path.join(output_dir, "origin_and_replaced.txt"), True)

    print_replacement_stats(num_wiki_seen, num_anchor, missed_counts, os.path.join(output_dir, "replacement_stat"))

    logging.info("All Done.")


if __name__ == '__main__':
    main()
