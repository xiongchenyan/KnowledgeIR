"""
overfit entities

sf feature only a bias
entity feature is the entity id

will only be used to overfit the results

serve as a subclass of base.Grounder
only need to replace the extract_for_surface and extract_for_entity

"""

from knowledge4ir.joint.grounding import Grounder


class EMemGrounder(Grounder):

    def extract_for_surface(self, h_sf_info, h_info):
        return {'b': 1}

    def extract_for_entity(self, h_e_info, h_sf_info, h_info):
        h_feature = dict()
        e_id = h_e_info['id']
        h_feature[e_id] = 1

        return h_feature

