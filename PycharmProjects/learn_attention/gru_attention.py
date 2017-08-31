import keras.backend as K
from keras.layers.embeddings import Embedding
from keras.layers import Input, merge, Activation, Dense, Flatten, Permute, RepeatVector, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras.models import Model
from keras import optimizers, initializers
from keras import metrics
import logging

logger = logging.getLogger('main.model')

class GRU_Model(object):
    def __init__(self, embedding_matrix):
        self.w_embedding_matrix = embedding_matrix
        self.word_index, self.EMBEDDING_DIM = self.w_embedding_matrix.shape

    def buildmodel_bigru_atten(self):
        logger.info('build model.')

        w_embedding_layer = Embedding(len(self.word_index) + 1,
                                      self.EMBEDDING_DIM,
                                      weights=[self.w_embedding_matrix],
                                      input_length=self.word_len, trainable=False,
                                      embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                        seed=None))
        w_sequence_input = Input(shape=(self.word_len,), name="titleword_input")
        w_embedded_sequences = w_embedding_layer(w_sequence_input)
        w_z_pos = GRU(256, implementation=2, return_sequences=True, go_backwards=False)(w_embedded_sequences)
        w_z_neg = GRU(256, implementation=2, return_sequences=True, go_backwards=True)(w_embedded_sequences)
        w_z_concat = merge([w_z_pos, w_z_neg], mode='concat', concat_axis=-1)

        attention = Dense(1, activation='tanh')(w_z_concat)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(256 * 2)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = merge([w_z_concat, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(256 * 2,))(sent_representation)

        model_final = Dense(1999, activation='softmax')(sent_representation)

        self.model = Model(input=w_sequence_input, outputs=model_final)
        adam = optimizers.Adam()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=[metrics.categorical_accuracy])
        for layer in self.model.layers:
            logger.info(layer.get_output_at(0).get_shape().as_list())
        logger.info(self.model.summary())

    def get_activations(model, inputs, print_shape_only=False, layer_name=None):
        # Documentation is available online on Github at the address below.
        # From: https://github.com/philipperemy/keras-visualize-activations
        print('----- activations -----')
        activations = []
        inp = self.model.input
        if layer_name is None:
            outputs = [layer.output for layer in model.layers]
        else:
            outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
        funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
        layer_outputs = [func([inputs, 1.])[0] for func in funcs]
        for layer_activations in layer_outputs:
            activations.append(layer_activations)
            if print_shape_only:
                print(layer_activations.shape)
            else:
                print(layer_activations)
        return activations