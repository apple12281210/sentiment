import logging
import os
from gru_attention import GRU_Model
from KerasDataManager import DataManager
from word2vec import Word2vec
import logging.config
from evaluate import Evaluate
import pandas as pd

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('main')

def save_pre(probs, data, savepath):
    if not len(probs) == len(data):
        logger.error('predict not equal test data.')
    for index in range(len(data)):
        data[index]['probs'] = probs[index]
    data = pd.DataFrame(data)
    logger.info('columns:{}'.format(data.columns))
    savefile = os.path.join(savepath, 'pred.csv')
    data.to_csv(savefile, index=False)

if __name__ == '__main__':
    root = '/home/apple/best'
    use = 0.05
    dataset = os.path.join(root, 'data/middle_files')
    w2v_path = os.path.join(root, 'data')
    save_result_path = os.path.join(root, 'result')
    # load data
    logger.info('load data')
    data = DataManager(dataset, use=use)
    train_x, train_y, test_x, test_y, word_index, train_data, test_data = data.gen_data()
    '''
    # load w2v
    logger.info('load w2v')
    w2v = Word2vec(w2v_path, word_index)
    embedding_matrix = w2v.load()
    # train model
    logger.info('train model')
    epoch = 25; batch = 32
    model = GRU_Model(embedding_matrix)
    model.buildmodel_bigru_atten()
    history = model.fit(train_x, train_y, validation_split=0.1, nb_epoch=epoch, batch_size=batch, shuffle=True)
    # predict test
    logger.info('predict test')
    test_pre = model.predict(test_x, batch_size=batch, verbose=0)
    # evaluate
    
    logger.info('evaluate')
    eval = Evaluate(test_pre, test_y)
    eval.evalute(test_pre, test_y)
    '''
    # save file
    logger.info('save file')
    test_pre = [0.1 for i in range(len(test_data))]
    save_pre(test_pre, test_data, save_result_path)
