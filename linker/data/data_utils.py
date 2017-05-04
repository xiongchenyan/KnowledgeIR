import logging
import os
import pickle
from .nif_parser import NIFParser
from .nif_utils import NifRelationCollector
import datetime
import sys


def run_or_load(path, func, *args):
    if os.path.exists(path):
        logging.info("Loading data from %s." % path)
        return pickle.load(open(path, 'rb'))
    else:
        logging.info("No saved data found.")
        result = func(*args)
        logging.info("Done processing, pickling as middle results.")

        with open(path, 'wb') as link_pickle_f:
            pickle.dump(result, link_pickle_f)
        logging.info("Done pickling.")

        return result


def canonical_freebase_id(freebase_id):
    if freebase_id.startswith("m."):
        return "/m/" + freebase_id[2:]
    else:
        return freebase_id


def load_redirects(redirect_nif):
    dbpedia_prefix = "http://dbpedia.org/resource/"

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

    sys.stdout.write("\nFinish loading redirects.\n")

    return redirect_to
