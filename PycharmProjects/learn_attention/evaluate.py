from sklearn.metrics import confusion_matrix
import numpy as np
import logging

class Evaluate(object):
    def __init__(self, grained=3):
        self.grained = grained
        self.cm = np.zeros((grained, grained), dtype=int)
        self.mat = [[], [], [0, 1], [0, 1, 2]]

    def statistic(self):
        cm = self.cm
        grained_total = float(np.sum(cm))
        ret = {}
        ret['three-way'] = np.sum([cm[i][i] for i in xrange(3)]) / grained_total
        if self.verbose:
            logging.info('Cm:\n%s' % self.cm)
        return ret

    def evalute(self, pre, val):
        def label(probs):
            return np.argmax(probs, axis=1)
        val = label(val)
        self.cm += confusion_matrix(val, pre, self.mat[self.grained])
        self.statistic()
        return val == pre

