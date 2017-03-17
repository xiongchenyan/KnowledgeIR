"""
provide pipe line api to run JointSemanticModel

run one fold: train-test
run with given train_in and test_in, can be used to
run one fold with dev: train-dev-test

fold split by qid % k

"""

from traitlets.config import Configurable
from traitlets import (
    Unicode,
    List
    Int
)



