# -*- coding: utf-8 -*-

import nltk
import os
import json
import datetime
from stemmer import stemmer


def learn(data, verbose=False):
    words = set()
    classes = set()
    documents = []
    ignore_words = ['?']

    # loop through each sentence in our training data
    for pattern in data:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern['sentence'])
        # add to our words list
        words |= set(w)
        # add to documents in our corpus
        documents.append((w, pattern['class']))
        # add to our classes list
        if pattern['class'] not in classes:
            classes.add(pattern['class'])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]

    # remove duplicates
    classes = list(classes)

    if verbose:
        print (len(documents), "documents")
        print (len(classes), "classes", classes)
        print (len(words), "unique stemmed words", words)

    return words, classes, documents


def prepare_training(words, classes, documents, verbose=False):
    # create our training data
    training = []
    output = []

    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:

        # initialize our bag of words
        bag = []

        # list of tokenized words for the pattern
        pattern_words = doc[0]

        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        training.append(bag)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        output.append(output_row)

    if verbose:
        # sample training/output
        i = 0
        w = documents[i][0]
        print ([stemmer.stem(word.lower()) for word in w])
        print (training[i])
        print (output[i])

    return training, output
