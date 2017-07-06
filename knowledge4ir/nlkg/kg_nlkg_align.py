"""
align given kg and nlkg, see the coverage and examples
input:
    edges and nlss in resource
output:
    for each entity:
        # of nlss edge, # of kg, # of overlapped (kg's tail in the nlss)
        [kg edge, [nlss of this kg edge]]
"""

from knowledge4ir.utils.resource import JointSemanticResource
import json
import sys
import logging


def align_per_entity(e_id, l_nlss, l_edge):
    """
    align for one entity
    :param e_id:
    :param l_nlss:
    :param l_edge:
    :return:
    """
    l_nlss = [nlss for nlss in l_nlss if nlss[1]]
    l_edge = [edge for edge in l_edge if edge[1].startswith('/m/')]

    h_aligned_info = dict()
    h_aligned_info['id'] = e_id
    h_aligned_info['nb_nlss'] = len(l_nlss)
    h_aligned_info['nb_triple'] = len(l_edge)

    s_nlss_e = set(sum(
        [nlss[1] for nlss in l_nlss],
        []
    ))
    s_edge_e = set([tail for edge, tail in l_edge])
    h_aligned_info['nb_nlss_connected_e'] = len(s_nlss_e)
    h_aligned_info['nb_kg_connected_e'] = len(s_edge_e)
    h_aligned_info['nb_overlap_e'] = len(s_nlss_e.intersection(s_edge_e))
    l_aligned_edge_nlss = align_edge_nlss(l_edge, l_nlss)
    h_aligned_info['aligned'] = l_aligned_edge_nlss
    return h_aligned_info


def align_edge_nlss(l_edge, l_nlss):
    h_e_nlss = dict()
    s_edge_e = set([tail for edge, tail in l_edge])
    for sent, l_e in l_nlss:
        for e in l_e:
            if e not in s_edge_e:
                continue
            if e not in h_e_nlss:
                h_e_nlss[e] = []
            h_e_nlss[e].append(sent)

    l_aligned = []
    for edge, tail in l_edge:
        l_sent = h_e_nlss.get(tail, [])
        l_aligned.append([edge, tail, {'nlss_sent': l_sent}])
    return l_aligned


def process(resource, out_name):
    out = open(out_name, 'w')
    for e, h_edge_info in resource.h_e_edge.items():
        l_edge = h_edge_info.get('edges', [])
        logging.info('aligning for [%s]', e)
        l_nlss = resource.l_h_nlss[0].get(e, [])
        h_aligned_info = align_per_entity(e, l_nlss, l_edge)
        print >> out, json.dumps(h_aligned_info)
    out.close()
    logging.info('finished')

if __name__ == '__main__':
    from knowledge4ir.utils import set_basic_log, load_py_config
    set_basic_log()
    if 3 != len(sys.argv):
        print "2 para: resource config + output"
        JointSemanticResource.class_print_help()
        sys.exit(-1)

    resource = JointSemanticResource(config=load_py_config(sys.argv[1]))
    process(resource, sys.argv[2])




