import gzip
import json
from knowledge4ir.utils import add_svm_feature, mutiply_svm_feature
from knowledge4ir.salience.utils.evaluation import SalienceEva
import logging


def open_func(corpus_in):
    return gzip.open if corpus_in.endswith("gz") else open


class JointSplitter:
    def __init__(self, entity_vocab_size, content_field='bodyText'):
        self.entity_vocab_size = entity_vocab_size
        self.content_field = content_field

    def load_pairs(self, docs, f_predict):
        with open_func(docs)(docs) as origin, open_func(f_predict)(
                f_predict) as pred:
            while True:
                try:
                    inline = origin.next()
                    pred_line = pred.next()

                    doc = json.loads(inline)
                    predict_res = json.loads(pred_line)

                    gold_doc = doc['docno']
                    pred_doc = predict_res['docno']

                    while not gold_doc == pred_doc:
                        # Some results may have skipped empty lines.
                        doc = json.loads(origin.next())
                        gold_doc = doc['docno']
                        yield None

                    # Backward compatibility.
                    if 'predict' in predict_res:
                        predictions = predict_res['predict']
                    else:
                        predictions = predict_res[self.content_field]['predict']

                    l_e = doc['spot']['bodyText']['entities']
                    l_label_e = doc['spot']['bodyText']['salience']
                    s_e_label = dict(zip(l_e, l_label_e))

                    l_evm = doc['event']['bodyText']['sparse_features'].get(
                        'LexicalHead', [])
                    l_label_evm = doc['event']['bodyText']['salience']
                    s_evm_label = dict(zip(l_evm, l_label_evm))

                    yield doc, predictions, s_e_label, s_evm_label

                except StopIteration:
                    break

    def split_and_eval(self, docs, f_predict):
        print("Split and evaluating joint predictions [%s]." % f_predict)

        evaluator = SalienceEva()  # evaluator with default values.

        h_e_total_eva = dict()
        h_e_mean_eva = dict()

        h_evm_total_eva = dict()
        h_evm_mean_eva = dict()

        e_p = 0
        evm_p = 0
        p = 0

        with open(f_predict + '.entity.json', 'w') as entity_out, \
                open(f_predict + '.event.json', 'w') as event_out:
            for res in self.load_pairs(docs, f_predict):
                p += 1

                if not res:
                    continue

                doc, predictions, s_e_label, s_evm_label = res

                l_e_pack = self.get_e_labels(predictions, s_e_label)
                l_evm_pack = self.get_evm_labels(predictions, s_evm_label)

                pred_event = {'bodyText': {}}
                pred_entity = {'bodyText': {}}

                if l_e_pack:
                    h_e = evaluator.evaluate(l_e_pack[0], l_e_pack[1])
                    e_p += 1
                    h_e_total_eva = add_svm_feature(h_e_total_eva, h_e)

                    pred_entity['bodyText']['predict'] = [[eid, score] for
                                                          eid, score in
                                                          zip(l_e_pack[2],
                                                              l_e_pack[0])]
                    pred_entity['docno'] = doc['docno']
                    pred_entity['eval'] = h_e

                    entity_out.write(json.dumps(pred_entity))
                    entity_out.write('\n')

                if l_evm_pack:
                    h_evm = evaluator.evaluate(l_evm_pack[0], l_evm_pack[1])
                    evm_p += 1
                    h_evm_total_eva = add_svm_feature(h_evm_total_eva, h_evm)

                    pred_event['bodyText']['predict'] = [[eid, score] for
                                                         eid, score in
                                                         zip(l_evm_pack[2],
                                                             l_evm_pack[0])]
                    pred_event['docno'] = doc['docno']
                    pred_event['eval'] = h_evm

                    event_out.write(json.dumps(pred_event))
                    event_out.write('\n')

                if not e_p == 0:
                    h_e_mean_eva = mutiply_svm_feature(h_e_total_eva, 1.0 / e_p)
                if not evm_p == 0:
                    h_evm_mean_eva = mutiply_svm_feature(h_evm_total_eva,
                                                         1.0 / evm_p)

                ep1 = '%.4f' % h_e_mean_eva[
                    'p@01'] if 'p@01' in h_e_mean_eva else 'N/A'
                evmp1 = '%.4f' % h_evm_mean_eva[
                    'p@01'] if 'p@01' in h_evm_mean_eva else 'N/A'

                sys.stdout.write(
                    '\rEvaluated %d files, %d with entities and %d '
                    'with events, En P@1: %s, Evm P@1: %s, '
                    % (p, e_p, evm_p, ep1, evmp1))

            print('')

        h_e_mean_eva = {}
        if not e_p == 0:
            h_e_mean_eva = mutiply_svm_feature(h_e_total_eva, 1.0 / e_p)
            logging.info('finished predicted [%d] docs on entity, eva %s', e_p,
                         json.dumps(h_e_mean_eva))

        h_evm_mean_eva = {}
        if not evm_p == 0:
            h_evm_mean_eva = mutiply_svm_feature(h_evm_total_eva, 1.0 / evm_p)
            logging.info('finished predicted [%d] docs on event, eva %s', evm_p,
                         json.dumps(h_evm_mean_eva))

        with open(f_predict + '.entity.eval', 'w') as out:
            json.dump([[k, v] for k, v in h_e_mean_eva.items()], out, indent=1)

        with open(f_predict + '.event.eval', 'w') as out:
            json.dump([[k, v] for k, v in h_evm_mean_eva.items()], out,
                      indent=1)

    def get_e_labels(self, predictions, s_e_label):
        e_list = []

        for pred in predictions:
            eid = pred[0]
            score = pred[1]
            if eid < self.entity_vocab_size:
                e_list.append((score, s_e_label[eid], eid))
        return zip(*e_list)

    def get_evm_labels(self, predictions, s_evm_label):
        evm_list = []

        for pred in predictions:
            eid = pred[0]
            score = pred[1]
            if eid >= self.entity_vocab_size:
                evm_list.append((score,
                                 s_evm_label[eid - self.entity_vocab_size],
                                 eid - self.entity_vocab_size))
        return zip(*evm_list)

    def split_joint_list(self, entity_vocab_size, predictions, s_e_label,
                         s_evm_label):
        e_list = []
        evm_list = []

        for pred in predictions:
            eid = pred[0]
            score = pred[1]
            if eid >= entity_vocab_size:
                evm_list.append((score, s_evm_label[eid - entity_vocab_size],
                                 eid - entity_vocab_size))
            else:
                e_list.append((score, s_e_label[eid], eid))

        return zip(*e_list), zip(*evm_list)


if __name__ == '__main__':
    import sys

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    args = sys.argv
    if len(args) < 2:
        print(
            "Usage: [this script] [gold standard] [prediction]"
            "[Default: 723749 entity vocab size]")
        exit(1)

    vocab_size = 723749 if len(args) < 4 else int(args[3])

    spliiter = JointSplitter(vocab_size)

    spliiter.split_and_eval(args[1], args[2])
