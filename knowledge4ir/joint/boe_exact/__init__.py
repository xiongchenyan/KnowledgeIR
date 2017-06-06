"""
boe exact match model
extract _flat_ learning to rank features

input:
    q info: query info with spotted or linked results
    doc info: doc info with spotted, linked, or co-reference results
    trec: candidate document ranking
output:
    the svm format letor features

06/06/2017
first set of features:
    base retrieval score
    BOE EF * doc fields
    BOE Coordinate match * doc fields

TODO:add co-reference features
"""


