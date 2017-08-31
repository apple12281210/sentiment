import pandas as pd
import collections
import os
import os.path as op
import numpy as np
import theano
import logging
import nltk
import collections
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import metrics
from keras.utils import to_categorical

logger = logging.getLogger('main')

pd.options.display.max_colwidth = 10000

class DataManager(object):
    def __init__(self, dataset, grained=2, use=1, max_sequence_length=300, max_nb_words=30000, validation_split=0.1):
        self.tasklist = ['entity', 'relation', 'event']
        self.contextlist = ['entity_mention_context3', 'rel_arg1_context3', 'trigger_context3']
        self.idlist = ['entity_id:source_id', 'rel_arg1_id:rel_arg2_id', 'event_mention_id:source_id:hopper_id']
        self.targetlist = ['entity_type', 'rel_arg1_entity_type', 'event_mention_type']
        self.grained = grained
        self.use = use
        self.texts = []
        self.labels = []
        self.data = pd.DataFrame()
        self.MAX_SEQUENCE_LENGTH = max_sequence_length
        self.MAX_NB_WORDS = max_nb_words
        self.VALIDATION_SPLIT = validation_split
        self.load_data(dataset)

    def load_data(self, dataset, use_stem=False):
        # stem context
        def stem(context):
            lemm = WordNetLemmatizer()
            words = []
            text = []
            sentences = nltk.sent_tokenize(str(context.lower()).decode('utf-8'))
            for sentence in sentences:
                words.extend(nltk.word_tokenize(sentence))
            for word in words:
                lemmed = lemm.lemmatize(word)
                if not lemmed == '':
                    text.append(lemmed)
            return ' '.join(text)

        def trans(label):
            if label == 'none':
                return -1
            elif label == 'pos':
                return 0
            elif label == 'neg':
                return 1

        def sta(text):
            return int(len(str(text).split(' '))/10)

        # run entity, relation, event
        sum = 0
        for i in range(len(self.tasklist)):
            taskfolder = op.join(dataset, self.tasklist[i]+'_info')
            logger.info(taskfolder)
            sonsum = 0; senlen = collections.defaultdict(int)
            # run all files
            for parent, dirnames, filenames in os.walk(taskfolder):
                for filename in filenames:
                    df = pd.read_csv(op.join(parent, filename))
                    data = pd.DataFrame()
                    sonsum = sonsum + df.shape[0]
                    sum = sum + df.shape[0]
                    df['whole_context'] = ''
                    df['whole_context'] = df[self.contextlist[i]]
                    df['senlen'] = df['whole_context'].apply(sta)
                    group = pd.DataFrame(df.groupby('senlen').size().rename('counts')).to_dict()
                    from collections import Counter
                    x, y = Counter(senlen), Counter(group['counts'])
                    senlen = dict(x+y)
                    self.texts.extend(df['whole_context'].tolist())
                    self.labels.extend(df['label_polarity'].apply(trans))

                    # save id
                    id_col_name = self.idlist[i].split(':')
                    for col_name in id_col_name:
                        data[col_name] = df[col_name]
                    # save train text
                    # data['seqs'] = df['whole_context']
                    # save label
                    data['label'] = df['label_polarity']
                    data['target'] = df[self.targetlist[i]]
                    data['type'] = self.tasklist[i]

                    self.data = self.data.append(data)

            logger.info('size: {}'.format(sonsum))
            ddict = sorted(senlen.iteritems(), key=lambda d: d[1], reverse=True)
            logger.info('senlen: {}'.format(ddict))
        logger.info('label: {}   text: {}'.format(len(self.labels), len(self.texts)))
        self.texts = self.texts[:int(len(self.texts)*self.use)]
        self.labels = self.labels[:int(len(self.labels) * self.use)]
        self.data = self.data.to_dict(orient='record')
        self.data = self.data[:int(len(self.data)*self.use)]

    def handle_2label(self):
        nb_validation_samples = int(self.VALIDATION_SPLIT * len(self.texts))
        train_x = self.texts[:-nb_validation_samples]
        test_x = self.texts[:-nb_validation_samples]
        train_y = self.labels[-nb_validation_samples:]
        test_y = self.labels[-nb_validation_samples:]
        train_data = self.data[:-nb_validation_samples]
        test_data = self.data[-nb_validation_samples:]
        logger.info(type(train_y))
        label2_index = [i for i in range(len(train_y)) if not(train_y[i] == -1)]
        train_x = [train_x[index] for index in label2_index]
        train_y = [train_y[index] for index in label2_index]
        train_data = [train_data[index] for index in label2_index]
        return train_x, train_y, test_x, test_y, train_data, test_data

    def gen_data(self):
        train_x, train_y, test_x, test_y, train_data, test_data = self.handle_2label()
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(train_x+test_x)
        train_sequences = tokenizer.texts_to_sequences(train_x)
        test_sequences = tokenizer.texts_to_sequences(test_x)

        word_index = tokenizer.word_index
        logger.info('Found %s unique tokens.' % len(word_index))

        train_x = pad_sequences(train_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        test_x = pad_sequences(test_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)

        train_y = to_categorical(np.asarray(train_y))
        logger.info('Shape of data tensor: {}'.format(train_x.shape))
        logger.info('Shape of label tensor: {}'.format(train_y.shape))

        # split the data into a training set and a validation set
        indices = np.arange(train_x.shape[0])
        np.random.shuffle(indices)
        train_x = train_x[indices]
        train_y = train_y[indices]

        logger.info('{}'.format(train_x))
        logger.info('{}'.format(train_y))
        logger.info('{}'.format(test_x))
        logger.info('{}'.format(test_y))
        return train_x, train_y, test_x, test_y, word_index, train_data, test_data

