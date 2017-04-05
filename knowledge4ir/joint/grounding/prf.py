"""
prf grounding
vote the entity using prf's infor
starting with RM3's entities

features:
    max, mean, bin 1 and bin 2 of the RM3's entities' vote
"""


import json
from knowledge4ir.joint.grounding import Grounder


class PrfGrounder(Grounder):

    def extract_for_entity(self, h_e_info, h_sf_info, h_info):
        return

