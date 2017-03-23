"""
overfit entities

sf feature only a bias
entity feature is the entity id

will only be used to overfit the results

serve as a subclass of base.Grounder
only need to replace the extract_for_surface and extract_for_entity

"""

from knowledge4ir.joint.grounding import Grounder
import json
from traitlets import (
    Unicode,
)

class EMemGrounder(Grounder):
    feature_pre = Unicode



