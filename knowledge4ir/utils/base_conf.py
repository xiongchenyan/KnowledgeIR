"""
I am the basic config file to be used for all the packages
    as a default
all the project level constant are set here.
this is sth to be used for impoet
"""

from os import path
ROOT_PATH = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
GDEVAL_PATH = ROOT_PATH + '/knowledge4ir/utils/gdeval.pl'
RANKSVM_PATH = ROOT_PATH + '/knowledge4ir/cv/svm_rank'
RANKLIB_PATH = ROOT_PATH + '/knowledge4ir/cv/RankLib.jar'
TMP_DIR = ROOT_PATH + '/tmp/S2Ranking/'
DATA_PATH = "/bos/usr0/cx/tmp/knowledge4ir/data/"
body_field = 'bodyText'
title_field = 'title'
TARGET_TEXT_FIELDS = [title_field, body_field, ]
MS_ENTITY_KEY = '3c552f84af52430482da7285e4da8ad9'
QREL_IN = "/bos/usr0/cx/tmp/knowledge4ir/data/qrel.all"
e_desp_field = 'desp'
e_name_field = 'name'
e_alias_field = 'alias'
E_TEXT_FIELDS = [e_name_field, e_alias_field, e_desp_field]
