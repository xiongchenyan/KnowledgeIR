"""
the family of kernel neural ranking model
    especially for entity representations

version 1 models: 6/19/2017
    K-NRM with fixed embedding
    K-NRM with distance metric
    attention K-NRM

two stage setup:
    1: module
        define the neural network module
        a pairwise training wrapper
    2: pipe line class
        I/O (data generator
        train
        test
    and scripts to run
        (this will be mainly implemented to the model module)
        train_test of a fold
        cross validation
        evaluation
"""


