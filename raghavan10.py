#!/usr/bin/python3
import logging
import string
import jsonhandler
import argparse

import string as str
import math
from decimal import Decimal

# natural language toolkit:
from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree
from nltk.grammar import Nonterminal
from nltk.grammar import Production
from nltk.grammar import PCFG
import nltk

import multiprocessing
import os
import pickle


corpusdir = ""
cache_dir = ""
outputdir = ""
parser = None

class Database:

    """A database contains authors of texts."""

    def __init__(self):
        """
        Initialize a database.
        """

        # The following list contains all authors
        # of the database.
        self.authors = []
        self.minimum = -1

    def add_author(self, *authors):
        """Add authors to the database."""
        for author in authors:
            self.authors.append(author)

    def find_author(self, text):
        probabilities = {}

        if self.minimum < 0:
            self.compute_minimum()

        for author in self.authors:
            probabilities[author.name] = author.compute_probability(text, self.minimum)

        return max(probabilities, key=probabilities.get)

    def compute_minimum(self):
        self.minimum = 1 # find minimum probability over all productions and authors
        for author in self.authors:
            for lhs in author.pcfg_prob:
                self.minimum = min(self.minimum, min(author.pcfg_prob[lhs].values()))


class Author:

    """Represents an author with a collection of his texts."""

    def __init__(self, name):
        self.name = name      # the author's name
        self.texts = []       # a list of the author's text
        self.pcfg = None      # a PCFG for all sentences of the author
        self.pcfg_prob = None # the probabilites of the PCFG by nonterminal symbol

        self.text_count = 0   # the number of texts by this author

    def add_text(self, *texts):
        """Add texts to this author."""
        for text in texts:
            self.texts.append(text)
            self.text_count += 1

    def buildPCFG(self):
        productions = []

        for text in self.texts:
            productions.extend(text.productions)

        self.texts = None    # free memory
        import gc
        gc.collect()

        self.pcfg = nltk.grammar.induce_pcfg(Nonterminal("ROOT"), productions)
        self.compute_pcfg_probabilites()

    def compute_pcfg_probabilites(self):
        pcfg_probabilities = {}

        for prod in self.pcfg.productions():
            if prod.lhs() not in pcfg_probabilities:
                pcfg_probabilities[prod.lhs()] = {}
            pcfg_probabilities[prod.lhs()][prod.rhs()] = prod.prob()

        self.pcfg_prob = pcfg_probabilities

    def compute_probability(self, text, minimum):
        probability_value = Decimal(0)  # log(1) = 0

        for prod in text.productions:
            if prod.lhs() in self.pcfg_prob and prod.rhs() in self.pcfg_prob[prod.lhs()]:
                # multiply probabilities - csp. to addition in logarithmic space
                probability_value += Decimal(math.log(self.pcfg_prob[prod.lhs()][prod.rhs()]))
            else:
                probability_value += Decimal(math.log(minimum))

        logging.info("computed:", probability_value, "for author", self.name)
        return probability_value

class Text:

    """Represents a single text."""

    def __init__(self, raw, name):
        """
        Initialize a text object with raw text.

        Keyword arguments:
        raw -- Raw text as a string.
        name -- Name of the text.
        """
        self.name = name

        # load production rules from cache if possible
        if os.path.exists(".text_cache/" + cache_dir + self.name):
            with open(".text_cache/" + cache_dir + self.name,'rb') as handle:
                self.productions = pickle.load(handle)
            print("loaded " + self.name + " from cache!")
        else:
            print("Processing text " + self.name)
            preprocessed = self.preprocess(raw)
            treebanked = self.treebank(preprocessed)
            self.productions = self.compute_productions(treebanked)

            with open(".text_cache/" + cache_dir + self.name,'wb') as handle:
                pickle.dump(self.productions, handle, protocol=2)

    def clean(self, text):
        '''
            cleans text from anything but alphanumerics and punctuation, replaces new line character
        '''
        text = text.replace('<NAME/>','Joe')
        text = text.replace('\n',' ')
        text = text.replace('...','-')
        text = text.replace('  ',' ')

        my_punctuation = str.punctuation[0] + str.punctuation[2:] + "’"
        text = ''.join(ch for ch in text if (ch.isalnum()) or (ch==' ') or (ch in my_punctuation))

        return text

    def preprocess(self, raw):
        """
        Preprocess the raw text.
        """
        return nltk.tokenize.sent_tokenize(self.clean(raw))

    def treebank(self, preprocessed):
        """
        Treebank the preprocessed text.
        """
        treebanked = []

        global parser

        for sentence in preprocessed:
            try:
                treebanked.append(parser.raw_parse(sentence))
            except ValueError:
                # throw away malformatted sentences
                logging.error("Malformatted sentence in text", self.name, ":", sentence)
            except OSError:
                # throw away malformatted sentences
                logging.error("Malformatted sentence in text", self.name, ":", sentence)

        return treebanked

    def compute_productions(self, treebanked):
        productions = []

        for root in treebanked:
            productions.extend(_traverse_productions(next(root)))
            if len(list(root)) > 0:
                logging.error("len(root) > 1 !!")

        return productions

def _traverse_productions(tree):
    productions = []
    if isinstance(tree, Tree) and tree.height() > 2: # not a leaf
        child_nonterminals = []
        for child in tree:
            productions.extend(_traverse_productions(child))
            child_nonterminals.append(Nonterminal(child.label()))
        productions.append(Production(Nonterminal(tree.label()), tuple(child_nonterminals)))
    return productions

def train_author(candidate):
    global parser

    author = Author(candidate)
    for training in jsonhandler.trainings[candidate]:
        logging.info(
            "Author '%s': Loading training '%s'", candidate, training)
        text = Text(jsonhandler.getTrainingText(candidate, training),
                candidate + "-" + training)
        author.add_text(text)

    author.buildPCFG()

    return author

def read_unknown(unknown):
    return Text(jsonhandler.getUnknownText(unknown),
                unknown)

def tira(corpusdir, outputdir):
    """
    Keyword arguments:
    corpusdir -- Path to a tira corpus
    outputdir -- Output directory
    """
    jsonhandler.loadJson(corpusdir)

    # load and process the training data
    logging.info("Load the training data...")
    jsonhandler.loadTraining()

    # initialize a stanford parser to treebank texts
    global parser
    parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

    # initialize a database of authors
    database = Database()

    # multithreading to parallelize text parsing
    number_processes = max(1, multiprocessing.cpu_count() - 1)
    text_pool = multiprocessing.Pool(number_processes)

    # parse author training documents
    for author in text_pool.imap_unordered(train_author, jsonhandler.candidates):
        database.add_author(author)

    # parse unknown test documents and run the testcases
    results = {} # unknown_text -> computed_author
    for unknown in text_pool.imap_unordered(read_unknown, jsonhandler.unknowns):
        results[unknown.name] = database.find_author(unknown)

    text_pool.close()
    text_pool.join()

    jsonhandler.storeJson(outputdir, list(results.keys()), list(results.values()))


def main():
    parser = argparse.ArgumentParser(description='Tira submission for the reimplementation of the authorship attribution approach using probabilistic context-free grammars.')
    parser.add_argument('-i', action='store',
        help='Path to input directory')
    parser.add_argument('-o', action='store',
        help='Path to output directory')

    args = vars(parser.parse_args())

    global corpusdir
    corpusdir = args['i']
    global outputdir
    outputdir = args['o']

    # initialize cache directory
    global cache_dir
    cache_dir = corpusdir.split('/')[-1] + "/"
    if cache_dir == "/":
        cache_dir = corpusdir.split('/')[-2] + "/"

    if not os.path.exists(".text_cache/" + cache_dir):
        os.makedirs(".text_cache/" + cache_dir)

    # start evaluation
    tira(corpusdir, outputdir)


if __name__ == "__main__":
    # execute only if run as a script
    logging.basicConfig(level=logging.ERROR,
        format='%(asctime)s %(levelname)s: %(message)s')
    main()
