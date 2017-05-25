# -*- coding: utf-8 -*-

import time
import numpy as np
from training import train
from learning import prepare_training, learn
from classify import Classifier


# 3 classes of training data
data = []
data.append({"class":"python", "sentence":"We use the framework Cromlech."})
data.append({"class":"python", "sentence":"The Python interpreter segfaulted."})
data.append({"class":"python", "sentence":"Coding in Plone is fascinating."})

data.append({"class":"history", "sentence":"The Cromlech was collapsed."})
data.append({"class":"history", "sentence":"The feathered python is a Mayan symbol."})
data.append({"class":"history", "sentence":"Mayan code was decyphered."})

data.append({"class":"novareto", "sentence":"Novareto"})
data.append({"class":"novareto", "sentence":"Christian Klinger is here."})
data.append({"class":"novareto", "sentence":"Novareto works with Plone and Cromlech"})
data.append({"class":"novareto", "sentence":"Christian Klinger works at Novareto"})


words, classes, documents = learn(data)
training, output = prepare_training(words, classes, documents)

X = np.array(training)
y = np.array(output)

start_time = time.time()

synapse = train(classes, X, y, hidden_neurons=20, alpha=0.1,
                epochs=100000, dropout=False, dropout_percent=0.2)

synapse_0 = np.asarray(synapse['synapse0'])
synapse_1 = np.asarray(synapse['synapse1'])

identify = Classifier(words, classes, 0.2, synapse_0, synapse_1)

identify("Do we use Cromlech or Plone ?")
identify("Christian Klinger likes Cromlech")
identify("Novareto codes on Plone")
identify("Novareto uses only Python")
identify("Plone is a collapsed framework")
identify("Python does not segfault often")
