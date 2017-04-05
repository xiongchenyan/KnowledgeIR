import sqlite3
import sys


class WbItemsPerSite:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def page_query(self, wikidata_id, wiki_site):
        pure_id = wikidata_id[1:]

        command = '''
                    SELECT ips_item_id, ips_site_id, ips_site_page 
                    from wb_items_per_site, sites
                    WHERE ips_item_id = %s 
                    AND ips_site_id = site_global_key
                    AND ips_site_id = %s
                    ''' % (pure_id, wiki_site)

        for row in self.cursor.execute(command):
            print (row)

        # Rerutn the wiki page name for the given wikidata id.
        return ""


if __name__ == '__main__':
    database_path = sys.argv[1]
    wb = WbItemsPerSite(database_path)
    wb.page_query("Q5920298", "enwiki")
