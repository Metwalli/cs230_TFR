from keras.callbacks import Callback
import json
import codecs
import os
class LossHistory(Callback):
    def __init__(self, filename):
        self.filename = filename
        self.history = self.loadHist(filename)
    # https://stackoverflow.com/a/53653154/852795
    def on_epoch_end(self, epoch, logs=None):
        new_history = {}
        for k, v in logs.items(): # compile new history from logs
            new_history[k] = [v] # convert values into lists
        self.history = self.appendHist(self.history, new_history) # append the logs
        self.saveHist(self.filename, self.history) # save history from current training

    def saveHist(self, path, history):
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4)

    def loadHist(self, path):
        n = {}  # set history to empty
        if os.path.exists(path):  # reload history if it exists
            with codecs.open(path, 'r', encoding='utf-8') as f:
                n = json.loads(f.read())
        return n

    def appendHist(h1, h2):
        if h1 == {}:
            return h2
        else:
            dest = {}
            for key, value in h1.items():
                dest[key] = value + h2[key]
            return dest