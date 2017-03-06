"""
A simple spotter based on surfarce forms
"""
import ahocorasick
import cPickle
import os


class Spotter:
    def __init__(self):
        pass

    def spot(self, text):
        pass


class SurfaceFormSpotter:
    def __init__(self, **kwargs):
        self._get_surface_dict(kwargs.get("surface_form"), kwargs.get("middle_data"))

    def _get_surface_dict(self, surface_form_path, middle_data_path):
        index_name = "ahocorasick"

        if not os.path.exists(middle_data_path):
            os.mkdir(middle_data_path)

        index_path = os.path.join(middle_data_path, index_name)

        if os.path.exists(index_path):
            self.surface_dict = cPickle.load(open(index_path))
        else:
            self._read_surface_form(surface_form_path)
            cPickle.dump(self.surface_dict, open(index_path, 'w'))

    def _read_surface_form(self, surface_form_path):
        surfaces = {}
        self.surface_dict = ahocorasick.Automaton()

        with open(surface_form_path) as surface_form_file:
            line_count = 0
            for line in surface_form_file.readlines():
                key, freebase_id, occurrences = line.strip().split("\t")
                key = key.lower()

                if key in surfaces:
                    surfaces[key].append((key, freebase_id, occurrences))
                else:
                    surfaces[key] = [(key, freebase_id, occurrences)]

                line_count += 1

                sys.stdout.write("%d\r" % line_count)

        sys.stdout.write("\n")

        for key, candidates in surfaces.iteritems():
            self.surface_dict.add_word(key, (candidates, key))

        self.surface_dict.make_automaton()

    def spot(self, text):
        all_candidates = []

        for end_index, (candidates, key) in self.surface_dict.iter(text):
            begin_index = end_index - len(key)
            all_candidates.append((begin_index, end_index, key, candidates))
        return all_candidates


if __name__ == '__main__':
    import sys

    surface_path = sys.argv[1]
    middle_output_path = sys.argv[2]

    spotter = SurfaceFormSpotter(surface_form=surface_path, middle_data=middle_output_path)
    print spotter.spot("Yahoo located at US.")
