import MySQLdb

"""
MySQLdb is required to read the Wikidata mapping
"""


class WbItemsPerSite:
    def __init__(self, user, passwd, db_name, host="localhost"):
        self.db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db_name)
        self.cursor = self.db.cursor()

    def page_query(self, wikidata_id, wiki_site):
        pure_id = wikidata_id[1:] if wikidata_id.startswith("Q") else wikidata_id

        command = """
                    SELECT ips_item_id, ips_site_id, ips_site_page 
                    from wb_items_per_site
                    WHERE ips_item_id = %s 
                    AND ips_site_id = '%s'
                    """ % (pure_id, wiki_site)

        self.cursor.execute(command)

        result = self.cursor.fetchall()

        if len(result) == 0:
            return None

        return result[0][2]

    def close(self):
        self.db.close()


if __name__ == '__main__':
    database_name = "wikidatawiki_wb_items_per_site"
    wb = WbItemsPerSite("hector", "hector", database_name)
    print wb.page_query("Q3079073", "enwiki")
