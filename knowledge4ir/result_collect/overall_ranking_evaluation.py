"""
analysis ranking performance

overall table
p-value
win/tie/loss

input:
    for each run
        a folder/
            eval.d01,- eval.d20
    name of each run

12/22/2017
    add load json format results
"""

from traitlets import (
    Unicode,
    Int,
    List,
)
from traitlets.config import Configurable
from knowledge4ir.utils import (
    load_gdeval_res,
)
from knowledge4ir.result_collect import (
    randomization_test,
    win_tie_loss,
)
import logging
import os
import json


class RankingPerformanceCollector(Configurable):
    result_dir = Unicode(help='result dir').tag(config=True)
    l_run_name = List(Unicode, help='run names').tag(config=True)
    baseline_name = Unicode(help='base line name').tag(config=True)
    l_sig_test_name = List(Unicode,
                           help='list of runs for all others to test statistical significance'
                           ).tag(config=True)
    l_sig_symbol = List(Unicode,
                        default_value=['\\dagger', '\\ddagger',
                                       '\\mathsection', '\\mathparagraph',
                                       '*', '**'],
                        help='symbols to indicate significances'
                        ).tag(config=True)
    out_dir = Unicode(help='out directory').tag(config=True)
    res_format = Unicode('trec',
                         help='the format of the evaluation files, trec or json'
                         ).tag(config=True)

    l_target_metric = List(Unicode, default_value=['ndcg', 'err']).tag(config=True)
    l_target_depth = List(Int, default_value=[20]).tag(config=True)
    main_metric = Unicode('ndcg').tag(config=True)
    main_depth = Int(20).tag(config=True)

    eva_prefix = Unicode('eval.d')
    sig_str = Unicode('\dagger')

    def __init__(self, **kwargs):
        super(RankingPerformanceCollector, self).__init__(**kwargs)
        self.l_run_h_eval_per_q = []  # h_eval{'metric': l_q_score in qid order}
        self.l_run_h_eval = []  # h_eval{'metric': score}
        self.h_base_eval_per_q = []
        self.h_base_eval = []
        self.l_to_comp_run_p = [self.l_run_name.index(name) for name in self.l_sig_test_name]
        self.h_res_loader = {
            'trec': self._load_trec_eval_results,
            'json': self._load_json_eval_results,
        }
        if self.res_format == 'json':
            self.l_target_depth = [0]

    def _load_eva_results(self):
        self.h_base_eval_per_q, self.h_base_eval = self._load_per_run_results(self.baseline_name)
        for run_name in self.l_run_name:
            h_eval_per_q, h_eval = self._load_per_run_results(run_name)
            logging.info('[%s] loaded overall res [%s]', run_name, json.dumps(h_eval))
            self.l_run_h_eval_per_q.append(h_eval_per_q)
            self.l_run_h_eval.append(h_eval)

        return

    def _load_per_run_results(self, run_name):
        eval_fname = os.path.join(self.result_dir, run_name)
        return self.h_res_loader[self.res_format](eval_fname)

    def _load_trec_eval_results(self, run_dir):
        h_eval_per_q = dict()
        h_eval = dict()
        for depth in self.l_target_depth:
            eva_res_name = os.path.join(run_dir, self.eva_prefix + '%02d' % depth)
            l_q_eva, ndcg, err = load_gdeval_res(eva_res_name)
            l_q_eva.sort(key=lambda item: int(item[0]))
            l_ndcg = [item[1][0] for item in l_q_eva]
            l_err = [item[1][1] for item in l_q_eva]
            for metric in self.l_target_metric:
                name = metric + '%02d' % depth
                if metric == 'ndcg':
                    h_eval_per_q[name] = l_ndcg
                    h_eval[name] = ndcg
                elif metric == 'err':
                    h_eval_per_q[name] = l_err
                    h_eval[name] = err
                else:
                    logging.error('[%s] metric not implemented', metric)
                    raise NotImplementedError
        return h_eval_per_q, h_eval

    def _load_json_eval_results(self, eval_fname):
        per_data_eval_name = eval_fname + '.json'
        final_eval_name = per_data_eval_name + '.eval'
        h_eval = dict(
            [item for item in json.load(open(final_eval_name))
             if item[0] in self.l_target_metric]
        )

        h_eval_per_data = dict()
        l_key_h_eval = []
        for line in open(per_data_eval_name):
            h = json.loads(line)
            key = self._get_data_key(h)
            if 'eval' not in h:
                logging.warn('%s has no eval field', line)
                continue
            h_this_eval = h['eval']
            l_key_h_eval.append([key, h_this_eval])
        l_key_h_eval.sort(key=lambda item: item[0])
        l_h_eval = [item[1] for item in l_key_h_eval]
        for metric in self.l_target_metric:
            l_score = [h_eval.get(metric, 0) for h_eval in l_h_eval]
            h_eval_per_data[metric] = l_score
        return h_eval_per_data, h_eval

    def _get_data_key(self, h_data):
        l_key_name = ['qid', 'docno']
        for name in l_key_name:
            if name in h_data:
                return h_data[name]
        logging.warn('%s has no key field', json.dumps(h_data))
        return ""

    def csv_overall_performance_table(self):
        self._load_eva_results()
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        out = open(self.out_dir +'/overall.csv', 'w')

        header = "\\bf{Method}"
        for eva_metric in self.l_target_metric:
            for d in self.l_target_depth:
                metric = eva_metric.upper() + ('@%2d' % d if d else '')
                header += '& \\bf{%s}' % metric + '& &\\bf{W/T/L}'

        # for metric in [self.target_metric.upper() + '@%2d' % d for d in self.l_target_depth]:
        #     header += '& \\bf{%s}' % metric + '& &\\bf{W/T/L}'
        print >> out, header + '\\\\ \\hline'
        print header + '\\\\ \\hline'
        logging.info('header made')
        for run_name in self.l_run_name:
            logging.info("geneerating row for [%s]", run_name)
            print >> out, self._overall_performance_per_run(run_name) + '\\\\'
            print self._overall_performance_per_run(run_name) + '\\\\'
        out.close()

        return

    def _overall_performance_per_run(self, run_name):
        """
        score (with , relative %, w/t/l
        :param run_name:
        :return:
        """
        res_str = '\\texttt{%s}\n' % run_name.replace('_', "\\_")
        if run_name != self.baseline_name:
            p = self.l_run_name.index(run_name)
            h_eval_per_q = self.l_run_h_eval_per_q[p]
            h_eval = self.l_run_h_eval[p]
        else:
            h_eval = self.h_base_eval
            h_eval_per_q = self.h_base_eval_per_q
        wtl_str = ' & --/--/--'
        for d in self.l_target_depth:
            for eva_metric in self.l_target_metric:
                metric = eva_metric + ('@%2d' % d if d else '')
                score = h_eval[metric]
                if run_name == self.baseline_name:
                    res_str += ' & $%.4f$ & -- ' % score
                    continue
                l_base_q_score = self.h_base_eval_per_q[metric]
                base_score = self.h_base_eval[metric]
                l_q_score = h_eval_per_q[metric]
                rel = float(score) / base_score - 1

                sig_mark = self._calc_sig_mark(l_q_score, metric)
                if sig_mark:
                    score_str = '${%.4f}^%s$' % (score, sig_mark)
                else:
                    score_str = '${%.4f}$' % score
                res_str += ' & ' + ' & '.join([
                    score_str,
                    "$ {0:+.2f}\\%  $ ".format(rel * 100),
                ]) + '\n\n'
                if eva_metric == self.main_metric:
                    if (d != 0) & (d == self.main_depth):
                        w, t, l = win_tie_loss(l_q_score, l_base_q_score)
                        wtl_str = '& %02d/%02d/%02d\n\n' % (w, t, l)

        res_str += wtl_str

        return res_str

    def _calc_sig_mark(self, l_q_score, metric):
        sig_mark = ""
        logging.info("significant testing for [%s] [%d] data points", metric, len(l_q_score))
        assert len(self.l_to_comp_run_p) <= len(self.l_sig_symbol)

        for i in xrange(len(self.l_to_comp_run_p)):
            run_p = self.l_to_comp_run_p[i]
            l_cmp_q_score = self.l_run_h_eval_per_q[run_p][metric]
            if sum(l_q_score) <= sum(l_cmp_q_score):
                # only test improvements
                continue
            p_value = randomization_test(l_q_score, l_cmp_q_score)
            if p_value < 0.05:
                sig_mark += self.l_sig_symbol[i] + ' '

        if sig_mark:
            sig_mark = '{' + sig_mark + "}"
        return sig_mark


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import (
        load_py_config,
        set_basic_log,
    )
    set_basic_log()
    if 2 > len(sys.argv):
        RankingPerformanceCollector.class_print_help()
        print "can list baseline and run name after conf parameter"
        sys.exit()

    conf = load_py_config(sys.argv[1])
    collector = RankingPerformanceCollector(config=conf)
    if len(sys.argv) == 2:
        collector.csv_overall_performance_table()
    else:
        baseline = sys.argv[2]
        run_name = sys.argv[3]
        collector.baseline_name = baseline
        collector.l_run_name = [run_name]
        collector.out_dir = os.path.join(collector.out_dir, run_name)
        collector.csv_overall_performance_table()






