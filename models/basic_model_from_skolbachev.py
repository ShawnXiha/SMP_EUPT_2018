from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Dropout, Conv1D, GlobalMaxPool1D, GlobalAvgPool1D

from keras.layers import CuDNNLSTM, CuDNNGRU
from models.attentions import *


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

if __name__ == '__main__':
    import gc
    import pandas as pd
    import numpy as np
    from keras import backend as K
    from config import embed_npz, data_npz
    from keras.utils.np_utils import to_categorical
    from config import maxlen, max_features
    from utils import F1Evaluation
    from sklearn.model_selection import train_test_split
    from keras.optimizers import Adam
    my_dict = {0: '人类作者', 1: '机器作者', 2: '机器翻译', 3: '自动摘要'}
    test = pd.read_csv("../inputs/vali.tsv", sep='\t')


    embedding_matrix = np.load(embed_npz)['arr_0']
    input_shape, classes, num_words, emb_size, emb_matrix = maxlen, 4,  max_features, 300, embedding_matrix
    data = np.load(data_npz)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_train = to_categorical(y_train, num_classes=None)
    X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)
    batch_size = 256
    epochs = 100
    for i, m in enumerate([getModel0, getModel1, getModel2, getModel3]):
        print(f"starting training model{i}!!!")
        model_name = f"model{i}_of_sko"
        gc.collect()
        K.clear_session()
        model = m(input_shape, classes, num_words, emb_size, emb_matrix)
        model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
        f1_val = F1Evaluation(validation_data=(X_val, y_val), interval=1)
        model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[f1_val])
        gc.collect()
        model.load_weights("best_weights.h5")
        y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)

        y_p = np.argmax(y_pred, 1)
        test['标签'] = np.vectorize(my_dict.get)(y_p)
        test.to_csv(f'../inputs/{model_name}_withoutcv_{batch_size}_{epochs}.csv',
                    columns=['id', '标签'], header=False, index=False)

