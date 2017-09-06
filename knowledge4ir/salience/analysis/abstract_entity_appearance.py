"""
check
    the fraction of abstract entities appear in the body or title
    the fraction of #1 frequent body entity in abstract
    the fraction of first position entity (title|body) in abstract
input:
    hashed nyt
output:
    each line:
        docno, abs entity cnt, fraction of abs e in title, fraction of abs e in body
        whether #1 body entity is salient, whether #1 positioned entity is salient
"""

import json
