"""
condor ope
"""

import subprocess


def get_cx_job():
    out_str = subprocess.check_output(['condor_q', 'cx'])
    l_job_id = [line.split()[0] for line in out_str if 'cx' in line]
    return l_job_id


def qsub_job(l_cmd):
    out_str = subprocess.check_output(l_cmd)
    l_job_id = [line.strip('.').split()[-1]
                for line in out_str.splitlines() if 'submitted to cluster' in line]
    return l_job_id
