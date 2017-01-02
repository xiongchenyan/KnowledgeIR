"""
collect cv results from log
very simple
input:
    a dir of condor_out
    from run_mul_kfolds.py
output:
    the results from the last line of condor_out
    also copy for backup

"""

from traitlets.config import Configurable
from traitlets import (
    Unicode
)
import os
import json
import ntpath
import logging
import shutil


class CVResCollector(Configurable):
    out_dir = Unicode("/bos/usr0/cx/tmp/knowledge4ir/cv/results", help='results collect out dir'
                      ).tag(config=True)

    def _seg_results(self, log_in):
        lines = open(log_in).read().splitlines()
        lines = [line for line in lines if 'INFO' in line]
        res_line = lines[-1]
        if 'cv evaluation' not in res_line:
            return None
        if 'bos' not in res_line:
            return None
        # logging.info('get res line [%s]', res_line)
        cv_dir = res_line.split(']')[0].split('[')[-1]
        base_name = ntpath.basename(ntpath.dirname(cv_dir.strip('/')))
        base_name.replace('.ranksvm', '')

        ndcg, err = res_line.split(' ')[-1].split(',')
        ndcg = float(ndcg)
        err = float(err)

        logging.info('[%s] dir [%s] res [%f, %f]', base_name, cv_dir, ndcg, err)
        return base_name, cv_dir, ndcg, err

    def collect(self, in_dir):
        l_res = []
        for dir_name, sub_dirs, file_names in os.walk(in_dir):
            for fname in file_names:
                if not fname.startswith('condor_out'):
                    continue
                l = self._seg_results(os.path.join(dir_name, fname))
                if not l:
                    continue
                base_name, cv_dir, ndcg, err = l
                l_res.append([base_name, ndcg, err])
                shutil.copytree(cv_dir, os.path.join(self.out_dir, base_name))
        l_res.sort(key = lambda item: item[0])
        for name, ndcg, err in l_res:
            print name + ',%f,,%f' % (ndcg, err)

        return


if __name__ == '__main__':
    import sys
    from knowledge4ir.utils import set_basic_log
    set_basic_log()
    if 2 != len(sys.argv):
        print 'dir of cv logs => results'
        sys.exit(-1)

    collector = CVResCollector()
    collector.collect(sys.argv[1])







