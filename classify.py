# -*- coding: utf-8 -*-

import json
from normalizer import think


def classify(words, classes, sentence, synapse_0, synapse_1, verbose=False, threshold=0.3):
    results = think(words, sentence, synapse_0, synapse_1, verbose)
    results = [[i,r] for i,r in enumerate(results) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results


class Classifier(object):

    def __init__(self, words, classes, threshold, *synapses):
        assert len(synapses) >= 2
        self.synapses = synapses
        self.threshold = threshold
        self.words = words
        self.classes = classes
        
    def __call__(self, sentence, verbose=False):
        return classify(
            self.words, self.classes, sentence,
            self.synapses[0], self.synapses[1],
            verbose=verbose, threshold=self.threshold)
