#!/usr/bin/python3
import logging
import string
import jsonhandler
import argparse

import math
from decimal import Decimal

# natural language toolkit:
from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree
from nltk.grammar import Nonterminal
from nltk.grammar import Production
from nltk.grammar import PCFG
import nltk


class Database:

    """A database contains authors of texts."""

    def __init__(self):
        """
        Initialize a database.
        """

        # The following list contains all authors
        # of the database.
        self.authors = []

    def add_author(self, *authors):
        """Add authors to the database."""
        for author in authors:
            self.authors.append(author)

    def find_author(self, text):
        probabilities = {}

        minimum = 1 # find minimum probability over all productions and authors
        for author in self.authors:
            for lhs in author.pcfg_prob:
                minimum = min(minimum, min(author.pcfg_prob[lhs].values()))

        for author in self.authors:
            probabilities[author.name] = author.compute_probability(text, minimum)

        print("text", text.name, "by author", max(probabilities, key=probabilities.get), "prob:", max(probabilities.values()))
        return max(probabilities, key=probabilities.get)


class Author:

    """Represents an author with a collection of his texts."""

    def __init__(self, name):
        self.name = name      # the author's name
        self.texts = []       # a list of the author's text
        self.pcfg = None      # a PCFG for all sentences of the author
        self.pcfg_prob = None # the probabilites of the PCFG by nonterminal symbol

        self.text_count = 0   # the number of texts by this author
        self.word_count = 0   # the number of words by this author

    def add_text(self, *texts):
        """Add texts to this author."""
        for text in texts:
            self.texts.append(text)
            self.text_count += 1
            self.word_count += text.word_count

    def buildPCFG(self):
        productions = []

        for text in self.texts:
            productions.extend(text.productions)

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

        print("computed:", probability_value, "for author", self.name)
        return probability_value

class Text:

    """Represents a single text."""

    def __init__(self, raw, name, sent_detector, parser, append_hyphens=True):
        """
        Initialize a text object with raw text.

        Keyword arguments:
        raw -- Raw text as a string.
        name -- Name of the text.
        """
        self.name = name
        self.word_count = -1  # number of words of this text (after tokeni-
                              # zation and combining words due to hyphens)

        self.raw = raw

        print("Processing text", self.name)
        preprocessed = self.preprocess(sent_detector, append_hyphens)
        treebanked = self.treebank(preprocessed, parser)
        self.productions = self.compute_productions(treebanked)

    def preprocess(self, sent_detector, append_hyphens=True):
        """
        Preprocess the raw text.
        """
        # split into words to remove line feeds
        tokenized_words = nltk.word_tokenize(self.raw)
        # remove hyphens and combine words
        words = []
        if append_hyphens:
            skip = False  # allows to skip the next word after joining two
            for i in range(len(tokenized_words)):
                if skip:
                    skip = False
                    continue
                if (len(tokenized_words[i]) > 1 and tokenized_words[i][-1:] == "-"):
                    words.append(tokenized_words[i][:-1] + tokenized_words[i+1])
                    skip = True
                else:
                    words.append(tokenized_words[i])
        else:
            words = tokenized_words

        self.word_count = len(words)
        return sent_detector.sentences_from_tokens(words)

    def treebank(self, preprocessed, parser):
        """
        Treebank the preprocessed text.
        """
        treebanked = []

        for sentence in preprocessed:
            try:
                treebanked.extend(parser.parse_sents([sentence]))
            except ValueError:
                # throw away malformatted sentences
                logging.error("Malformatted sentence in text", self.name, ":", sentence)
            except OSError:
                # throw away malformatted sentences
                logging.error("Malformatted sentence in text", self.name, ":", sentence)

#       for tree in treebanked:
#           for t in tree:
#               t.pprint()
#               t.pretty_print()
#               print("label",t.label())         # tree's constituent type
#               sys.exit(0)
#           time.sleep(5)
#       sys.exit(0)

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

def tira(corpusdir, outputdir):
    """
    Keyword arguments:
    corpusdir -- Path to a tira corpus
    outputdir -- Output directory
    """
    jsonhandler.loadJson(corpusdir)
    jsonhandler.loadTraining()

    # load and process the training data
    logging.info("Load the training data...")
    database = Database()
    # initialize a stanford parser to treebank texts
    parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    j = 0 # TODO remove
    if j > 0: print(j, "author(s)") # TODO remove
    for candidate in jsonhandler.candidates:
        author = Author(candidate)
        k = 0 # TODO remove
        if k > 0: print(k, "text(s) per author") # TODO remove
        for training in jsonhandler.trainings[candidate]:
            logging.info(
                "Author '%s': Loading training '%s'", candidate, training)
            text = Text(jsonhandler.getTrainingText(candidate, training),
                    candidate + " " + training, sent_detector, parser)
            author.add_text(text)
            k -= 1 # TODO remove
            if k == 0: break # TODO remove
        author.buildPCFG()
        database.add_author(author)
        j -= 1 # TODO remove
        if j == 0: break # TODO remove

    # run the testcases
    results = []
    l = 0 # TODO remove
    if l > 0: print(j, "unknown(s)") # TODO remove
    for unknown in jsonhandler.unknowns:
        text = Text(jsonhandler.getUnknownText(unknown),
                        unknown, sent_detector, parser)
        results.append(database.find_author(text))
        l -= 1 # TODO remove
        if l == 0: break # TODO remove

    jsonhandler.storeJson(outputdir, jsonhandler.unknowns, results)
    

def main():
    parser = argparse.ArgumentParser(description='Tira submission for the reimplementation of the authorship attribution approach using probabilistic context-free grammars.')
    parser.add_argument('-i', action='store',
        help='Path to input directory')
    parser.add_argument('-o', action='store',
        help='Path to output directory')

    args = vars(parser.parse_args())

    corpusdir = args['i']
    outputdir = args['o']

    tira(corpusdir, outputdir)


if __name__ == "__main__":
    # execute only if run as a script
    logging.basicConfig(level=logging.ERROR,
        format='%(asctime)s %(levelname)s: %(message)s')
    main()
