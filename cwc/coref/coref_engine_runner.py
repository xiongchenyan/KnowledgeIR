import subprocess
from subprocess import Popen, PIPE
from config import BerkeleyConfig
import os
from shutil import copyfile
from stanfordcorenlp import StanfordCoreNLP
from config import StanfordConfig
import logging


class CorefEngineRunner:
    def __init__(self):
        pass

    def run_coref(self, text):
        pass

    def run_directory(self, dir):
        pass


class StanfordEntityCoref(CorefEngineRunner):
    def __init__(self):
        CorefEngineRunner.__init__(self)
        logging.info("Initializing Stanford CoreNLP models.")
        self.nlp = StanfordCoreNLP(StanfordConfig.system_path, memory='8g')
        self.props = {'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, depparse, dcoref'}
        logging.info("Initialization done.")

    def run_coref(self, text):
        result = self.nlp.annotate(text, properties=self.props)
        print result
        raw_input("Wait")

    def run_directory(self, dir):
        pass


def run_with_output(commands):
    print("The command to be run is:")
    print(" ".join(commands))

    p = Popen(
        commands, stdin=PIPE, stdout=PIPE, stderr=PIPE
    )

    out, err = p.communicate()

    print("Process output:")
    print(out)

    if not p.returncode == 0:
        print("Process error:")
        print(err)


class BerkeleyEntityCoref(CorefEngineRunner):
    def run_coref(self, text):
        pass

    def run_directory(self, input_dir, output_dir):
        os.chdir(BerkeleyConfig.system_path)
        self.preprocess(input_dir, output_dir)
        self.prepare_wiki(output_dir)
        self.run_joint(output_dir)

    def preprocess(self, input_dir, output_dir):
        print("Running preprocessing for Berkeley parser.")

        # You have to create this for Berkeley parser.
        scratch_path = os.path.join(output_dir, 'scratch')
        if not os.path.exists(scratch_path):
            os.makedirs(scratch_path)

        commands = [
            'java', '-Xmx2g', '-cp', '%s' % BerkeleyConfig.system_jar,
            'edu.berkeley.nlp.entity.preprocess.PreprocessingDriver', '++%s' % BerkeleyConfig.config_path,
            '-execDir', '%s/scratch/preprocess' % output_dir,
            '-inputDir', input_dir,
            '-outputDir', '%s/preprocessed' % output_dir
        ]

        run_with_output(commands)

    def prepare_wiki(self, output_dir):
        print("Prepareing relevant Wikipedia documents for the dataset.")

        preprocessed_dir = os.path.join(output_dir, "preprocessed")
        auto_conll_dir = os.path.join(output_dir, "preprocessed_conll")

        if not os.path.exists(auto_conll_dir):
            os.makedirs(auto_conll_dir)

        for f in os.listdir(preprocessed_dir):
            copyfile(os.path.join(preprocessed_dir, f), os.path.join(auto_conll_dir, f) + ".auto_conll")

        commands = ['java', '-Xmx4g', '-cp', '%s:lib/bliki-resources' % BerkeleyConfig.system_jar,
                    'edu.berkeley.nlp.entity.wiki.WikipediaInterface',
                    '-datasetPaths', '%s' % os.path.join(output_dir, "preprocessed_conll"),
                    '-wikipediaDumpPath', '%s' % BerkeleyConfig.wikipedia_dump,
                    '-outputPath', '%s' % BerkeleyConfig.wiki_db
                    ]

        run_with_output(commands)

    def run_joint(self, output_dir):
        commands = [
            'java', '-Xmx8g', '-cp', '%s' % BerkeleyConfig.system_jar,
            'edu.berkeley.nlp.entity.Driver', '++config/base.conf',
            '-execDir', '%s' % os.path.join(output_dir, "scratch/joint"),
            '-mode', 'PREDICT',
            '-modelPath', 'models/joint-onto.ser.gz',
            '-testPath', '%s' % os.path.join(output_dir, "preprocessed"),
            '-wikipediaPath', BerkeleyConfig.wiki_db
        ]

        run_with_output(commands)
