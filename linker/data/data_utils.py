import logging
import os
import pickle


def run_or_load(path, func, *args):
    if os.path.exists(path):
        logging.info("Loading data from %s." % path)
        return pickle.load(open(path))
    else:
        logging.info("No saved data found.")
        result = func(*args)
        logging.info("Done processing, pickling as middle results.")

        with open(path, 'w') as link_pickle_f:
            pickle.dump(result, link_pickle_f)
        logging.info("Done pickling.")

        return result


def canonical_freebase_id(freebase_id):
    if freebase_id.startswith("m."):
        return "/m/" + freebase_id[2:]
    else:
        return freebase_id
