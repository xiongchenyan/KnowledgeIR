"""

"""

from SPARQLWrapper import SPARQLWrapper, JSON
import logging
import xml.etree.ElementTree as ET
import json


def query_generator(xml_q_in):
    """

    :param xml_q_in: the qald-2 train and test xml sparql query file
    :return: yield a query id and a query SPARQL string each time
    """

    tree = ET.parse(xml_q_in)

    for child in tree.getroot():
        qid = child.attrib['id']
        query = ""

        for son in child:
            if son.tag == 'query':
                query = son.text
        if not query:
            continue
        if 'OUT OF SCOPE' in query:
            continue
        yield qid, query
    return


def fetch_res(sparql_str):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(sparql_str)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results


def run_given_sparql(xml_q_in_name, out_name):
    out = open(out_name, 'w')

    for qid, query in query_generator(xml_q_in_name):
        logging.info('running [%s]', qid)
        results = fetch_res(query)
        print >> out, '%s\t%s' % (qid, json.dumps(results))

    out.close()


if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log
    import sys
    set_basic_log()
    if 3 != len(sys.argv):
        print "2 para: xml sparql in + out"
        sys.exit(-1)

    run_given_sparql(*sys.argv[1:])
