"""
Paring wikipedia dump using the MediaWiki utilities: https://github.com/mediawiki-utilities

Unfortunately this only works at Python3.
"""
import mwxml
import mwtypes


class WikipediaLinkExtractor:
    def __init__(self, dump_path):
        self.dump = mwxml.Dump.from_file(mwtypes.files.reader(dump_path))


