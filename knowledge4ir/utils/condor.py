"""
condor ope
"""

import subprocess
import logging
import json


def get_cx_job():
    out_str = subprocess.check_output(['condor_q', 'cx'])
    l_job_id = [line.split()[0] for line in out_str.splitlines() if 'cx' in line]
    return l_job_id


def qsub_job(l_cmd):
    out_str = subprocess.check_output(['qsub'] + l_cmd)
    l_job_id = [line.strip('.').split()[-1]
                for line in out_str.splitlines() if 'submitted to cluster' in line]
    logging.info('submit %s to %s', json.dumps(l_cmd), l_job_id[0])
    return l_job_id[0]
