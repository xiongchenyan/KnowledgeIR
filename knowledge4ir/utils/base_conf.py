"""
I am the basic config file to be used for all the packages
    as a default
all the project level constant are set here.
this is sth to be used for impoet
"""

from os import path
ROOT_PATH = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
GDEVAL_PATH = ROOT_PATH + '/knowledge4ir/utils/gdeval.pl'
RANKSVM_PATH = ROOT_PATH + '/knowledge4ir/letor/svm_rank'
RANKLIB_PATH = ROOT_PATH + '/knowledge4ir/letor/RankLib.jar'
TMP_DIR = ROOT_PATH + '/tmp/knowledge4ir/'
DATA_PATH = "/bos/usr0/cx/tmp/knowledge4ir/data/"
body_field = 'bodyText'
paper_abstract_field = 'paperAbstract'
abstract_field = 'abstract'
title_field = 'title'
E_GRID_FIELD = 'e_grid'
TARGET_TEXT_FIELDS = [title_field, body_field, ]
QUERY_FIELD = 'query'
MS_ENTITY_KEY = '3c552f84af52430482da7285e4da8ad9'
QREL_IN = "/bos/usr0/cx/tmp/knowledge4ir/data/qrel.all"
e_desp_field = 'desp'
e_name_field = 'name'
e_alias_field = 'alias'
ENTITY_TEXT_FIELDS = [e_name_field, e_alias_field, e_desp_field]
GROUND_FIELD = 'ground'
SPOT_FIELD = 'spot'
MATCH_FIELD = 'match'
COREFERENCE_FIELD = 'coreferences'

