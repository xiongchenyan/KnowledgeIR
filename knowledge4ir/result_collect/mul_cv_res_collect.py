"""
collect cv results using condor_out. log
input:
    a dir with condor out logs
    qrel (defaulted)
    out_dir to keep results

do:
    read condor out log, get dirs with finished cross validation (folds all there)
    collect cv results for each finished cv_dir
    cp trec and eval to target folder, with the same base dir name as the cv_dir
    print summary results

"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int
)
from knowledge4ir.utils import (
    QREL_IN,
    load_gdeval_res,
    load_py_config,
    set_basic_log
)
from knowledge4ir.duet_model.att_ltr.cv_result_collect import collect_cv_results
import json
import subprocess
import logging
import os
import shutil
import ntpath


class CondorCvResCollector(Configurable):
    K = Int(10).tag(config=True)
    out_dir = Unicode('/bos/usr0/cx/tmp/knowledge4ir/att_ltr/results', help='out dir').tag(config=True)
    qrel = Unicode(QREL_IN, help='qrel in').tag(config=True)

    def collect_finished_cv_dir(self, log_dir):
        """
        read all files with condor_out.*
        get dir name in the last line
        :param log_dir:
        :return:
        """

        l_log_names = self._get_all_logs(log_dir)
        l_out_dir_names = [self._get_out_dir_per_log(name) for name in l_log_names]

        h_cv_cnt = {}
        for name in l_out_dir_names:
            if not 'Fold' in name:
                continue
            cv_dir, fold_d = self._split_dir_fold(name)
            if not cv_dir.startswith('/bos'):
                continue
            if cv_dir not in h_cv_cnt:
                h_cv_cnt[cv_dir] = 1
            else:
                h_cv_cnt[cv_dir] += 1
        l_finished_cv_dir = [item[0] for item in h_cv_cnt.items() if item[1] >= self.K]
        logging.info('finished cv dirs %s', json.dumps(l_finished_cv_dir))
        l_finished_cv_dir.sort()
        return l_finished_cv_dir

    def per_cv_dir_eval(self, cv_dir):
        logging.info('start [%s]', cv_dir)
        collect_cv_results(cv_dir, self.qrel)
        method_base = ntpath.basename(cv_dir.strip('/'))
        this_out_dir = os.path.join(self.out_dir, method_base)
        if not os.path.exists(this_out_dir):
            os.makedirs(this_out_dir)
        # subprocess.check_output(['cp', cv_dir + '/eval', cv_dir + '/trec', this_out_dir])
        shutil.copyfile(cv_dir + '/eval', this_out_dir + '/eval')
        shutil.copyfile(cv_dir + '/trec', this_out_dir + '/trec')
        logging.info('res moved to [%s]', this_out_dir)
        __, ndcg, err = load_gdeval_res(cv_dir + '/eval')
        return method_base, ndcg, err

    def collect_cv_res(self, log_dir):
        l_finished_cv_dir = self.collect_finished_cv_dir(log_dir)
        l_name_eva = []
        for cv_dir in l_finished_cv_dir:
            l_name_eva.append(self.per_cv_dir_eval(cv_dir))
        print "\n\n"
        for name, ndcg, err in l_name_eva:
            print "%s,%f,,%f" % (' '.join(name.split('_')), ndcg, err)
        return

    def _get_all_logs(self, log_dir):
        l_log_names = []
        for dir_name, sub_dirs, file_names in os.walk(log_dir):
            for fname in file_names:
                if fname.startswith('condor_out'):
                    l_log_names.append(os.path.join(dir_name, fname))
        logging.info('total [%d] condor out', len(l_log_names))
        return l_log_names

    def _get_out_dir_per_log(self, log_in):
        line = [s for s in open(log_in).read().splitlines() if "root - INFO" in s][-1]
        dir_name = line.split('[')[-1].split(']')[0]
        logging.debug('get out dir [%s]', dir_name)
        return dir_name

    def _split_dir_fold(self, dir_name):

        cv_dir = ntpath.dirname(dir_name)
        fold_d = ntpath.basename(dir_name)
        return cv_dir, fold_d


if __name__ == '__main__':
    import sys
    set_basic_log()

    if 2 > len(sys.argv):
        print "1+ para: log_dir + conf (if not default)"
        sys.exit(-1)
    if len(sys.argv) > 2:
        conf = load_py_config(sys.argv[2])
        collector = CondorCvResCollector(config=conf)
    else:
        collector = CondorCvResCollector()

    collector.collect_cv_res(sys.argv[1])


