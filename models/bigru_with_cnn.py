from keras.layers import Dense, Input, Bidirectional, Conv1D, CuDNNGRU, GRU
from keras.layers import Dropout, Embedding
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, \
    concatenate, SpatialDropout1D
from keras.models import Model
from keras import optimizers
import numpy as np
from config import *

embedding_matrix = np.load("./inputs/embedding.npz")['arr_0']


def build_model(dr_emb=0.2, gru_nums=128, conv_nums=64):
    sequence_input = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(sequence_input)
    x = SpatialDropout1D(dr_emb)(x)
    x = Bidirectional(CuDNNGRU(gru_nums, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Conv1D(conv_nums, kernel_size=3, padding="valid",
               kernel_initializer="glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    preds = Dense(4, activation="softmax")(x)
    model = Model(sequence_input, preds)
    adam = optimizers.adam(clipvalue=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    import gc
    import pandas as pd
    import numpy as np
    from keras import backend as K
    from config import embed_npz, data_npz
    from sklearn.model_selection import KFold
    from keras.utils.np_utils import to_categorical
    from config import maxlen, max_features, embed_size
    from utils import F1Evaluation

    test = pd.read_csv("../inputs/vali.tsv", sep='\t')

    model_name = 'bigru_withcnn'
    fold = 10
    batch_size = 256
    epochs = 100
    np.random.seed(233)
    embedding_matrix = np.load(embed_npz)['arr_0']
    data = np.load(data_npz)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_train = to_categorical(y_train, num_classes=None)
    y_pred = np.zeros((test.shape[0], 4))
    kf = KFold(n_splits=fold, shuffle=True, random_state=239)
    for train_index, test_index in kf.split(x_train):
        kfold_y_train, kfold_y_test = y_train[train_index], y_train[test_index]
        kfold_X_train = x_train[train_index]
        kfold_X_valid = x_train[test_index]

        gc.collect()
        K.clear_session()

        model = build_model()

        f1_val = F1Evaluation(validation_data=(kfold_X_valid, kfold_y_test),
                              interval=1)

        model.fit(kfold_X_train, kfold_y_train,
                  batch_size=batch_size,
                  epochs=epochs, verbose=1, callbacks=[f1_val])
        gc.collect()
        model.load_weights("best_weights.h5")

        y_pred += model.predict(x_test, batch_size=batch_size,
                                verbose=1) / fold

    my_dict = {0: '人类作者', 1: '机器作者', 2: '机器翻译', 3: '自动摘要'}
    y_p = np.argmax(y_pred, 1)
    test['标签'] = np.vectorize(my_dict.get)(y_p)
    test.to_csv(
        f'../inputs/{model_name}_sub_bilistmcnn_{fold}_{batch_size}_{epochs}_cv.csv',
        columns=['id', '标签'], header=False, index=False)
