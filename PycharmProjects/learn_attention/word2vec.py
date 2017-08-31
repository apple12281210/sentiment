import logging
import os
import numpy as np
logger = logging.getLogger('main.word2vec')

class Word2vec(object):
    def __init__(self, GLOVE_DIR, word_index, EMBEDDING_DIM = 300):
        self.GLOVE_DIR = GLOVE_DIR
        self.word_index = word_index
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def load(self):
        logger.info('load word2vec')
        self.embeddings_index = {}
        f = open(os.path.join(self.GLOVE_DIR, 'glove.840B.' + str(self.EMBEDDING_DIM) + 'd.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        not_exist = 0
        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
            else:
                not_exist = not_exist + 1
        logger.info('not exist word: {}'.format(not_exist))
        return self.embedding_matrix