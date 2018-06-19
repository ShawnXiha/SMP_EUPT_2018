from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Dropout, Conv1D, GlobalMaxPool1D, GlobalAvgPool1D

from keras.layers import CuDNNLSTM, CuDNNGRU
from .attentions import *


def getModel0(input_shape, classes, num_words, emb_size, emb_matrix,
              emb_dropout=0.5,
              attention=0, dense=False, emb_trainable=False, gru=True):
    x_input = Input(shape=(input_shape,))

    emb = Embedding(num_words, emb_size, weights=[emb_matrix],
                    trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)

    if gru:
        rnn1 = Bidirectional(CuDNNGRU(64, return_sequences=True))(emb)
        rnn2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(rnn1)
    else:
        rnn1 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(emb)
        rnn2 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(rnn1)

    x = concatenate([rnn1, rnn2])

    if attention == 1:
        x = AttentionWeightedAverage()(x)
    elif attention == 2:
        x = Attention()(x)
    else:
        x = GlobalMaxPooling1D()(x)

    if dense:
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)

    x_output = Dense(classes, activation='softmax')(x)
    return Model(inputs=x_input, outputs=x_output)


def getModel1(input_shape, classes, num_words, emb_size, emb_matrix,
              emb_dropout=0.5,
              attention=0, dense=False, emb_trainable=False, gru=True):
    x_input = Input(shape=(input_shape,))

    emb = Embedding(num_words, emb_size, weights=[emb_matrix],
                    trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)

    if gru:
        rnn, rnn_fw, rnn_bw = Bidirectional(
            CuDNNGRU(100, return_sequences=True, return_state=True))(emb)
    else:
        rnn, rnn_fw, rnn_bw = Bidirectional(
            CuDNNLSTM(100, return_sequences=True, return_state=True))(emb)

    rnn_max = GlobalMaxPool1D()(rnn)
    rnn_avg = GlobalAvgPool1D()(rnn)
    rnn_last = concatenate([rnn_fw, rnn_bw])

    x = concatenate([rnn_max, rnn_avg, rnn_last])

    if dense:
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)

    x_output = Dense(classes, activation='softmax')(x)
    return Model(inputs=x_input, outputs=x_output)


def getModel2(input_shape, classes, num_words, emb_size, emb_matrix,
              emb_dropout=0.5,
              attention=0, dense=False, emb_trainable=False, gru=True):
    x_input = Input(shape=(input_shape,))

    emb = Embedding(num_words, emb_size, weights=[emb_matrix],
                    trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)

    if gru:
        rnn = Bidirectional(CuDNNGRU(75, return_sequences=True))(emb)
    else:
        rnn = Bidirectional(CuDNNLSTM(75, return_sequences=True))(emb)

    cnn1 = Conv1D(filters=50, kernel_size=3, activation='relu',
                  padding='same')(rnn)
    cnn2 = Conv1D(filters=50, kernel_size=4, activation='relu',
                  padding='same')(rnn)
    cnn3 = Conv1D(filters=50, kernel_size=5, activation='relu',
                  padding='same')(rnn)

    x = concatenate([rnn, cnn1, cnn2, cnn3])

    if attention == 1:
        x = AttentionWeightedAverage()(x)
    elif attention == 2:
        x = Attention()(x)
    else:
        x = GlobalMaxPooling1D()(x)

    if dense:
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)

    x_output = Dense(classes, activation='softmax')(x)
    return Model(inputs=x_input, outputs=x_output)


def getModel3(input_shape, classes, num_words, emb_size, emb_matrix,
              emb_dropout=0.5,
              attention=0, dense=False, emb_trainable=False, gru=True):
    x_input = Input(shape=(input_shape,))

    emb = Embedding(num_words, emb_size, weights=[emb_matrix],
                    trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)

    if gru:
        rnn1 = Bidirectional(CuDNNGRU(64, return_sequences=True))(emb)
        rnn2 = Bidirectional(CuDNNGRU(64, return_sequences=False))(rnn1)
    else:
        rnn1 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(emb)
        rnn2 = Bidirectional(CuDNNLSTM(64, return_sequences=False))(rnn1)

    x = rnn2

    if dense:
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)

    x_output = Dense(classes, activation='softmax')(x)
    return Model(inputs=x_input, outputs=x_output)
