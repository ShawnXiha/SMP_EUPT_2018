from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, MaxPooling1D, Flatten
from keras import optimizers
from utils import F1Evaluation


def inception(x, co, relu=True, norm=True):
    assert (co % 4 == 0)
    cos = [co // 4] * 4
    activatie = Sequential()
    if norm:
        activatie.add(BatchNormalization())
    if relu:
        activatie.add(Activation('relu'))

    branch1 = Sequential()
    branch2 = Sequential()
    branch3 = Sequential()
    branch4 = Sequential()
    branch1.add(Conv1D(cos[0], 1, strides=1))
    branch2.add(Conv1D(cos[1], 1))
    branch2.add(BatchNormalization())
    branch2.add(Activation('relu'))
    branch2.add(Conv1D(cos[1], 3, strides=1, padding='same'))
    branch3.add(Conv1D(cos[2], 3, padding='same'))
    branch3.add(BatchNormalization())
    branch3.add(Activation('relu'))
    branch3.add(Conv1D(cos[2], 5, strides=1, padding='same'))
    branch4.add(Conv1D(cos[3], 3, strides=1, padding='same'))
    branch1 = branch1(x)
    branch2 = branch2(x)
    branch3 = branch3(x)
    branch4 = branch4(x)
    result = activatie(concatenate([branch1, branch2, branch3, branch4]))
    return result


def getModelInception(maxlen, classes, max_features, emb_size, emb_matrix,
                      emb_dropout=0.5, inception_dim=256, clipvalue=1,
                      emb_trainable=False):
    x_input = Input(shape=(maxlen,))

    emb = Embedding(max_features, emb_size, weights=[emb_matrix],
                    trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)

    content_conv = inception(emb, inception_dim)
    content_conv = inception(content_conv, inception_dim)
    content_conv = inception(content_conv, inception_dim)
    content_conv = MaxPooling1D(maxlen)(content_conv)

    fc = Sequential()
    fc.add(Flatten())
    fc.add(Dense(128))
    fc.add(BatchNormalization())
    fc.add(Activation('relu'))
    fc.add(Dense(classes, activation='softmax'))

    outp = fc(content_conv)
    model = Model(inputs=x_input, outputs=outp)

    adam = optimizers.adam(lr=5e-3, clipvalue=clipvalue)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
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

    test = pd.read_csv("../inputs/vali.tsv", sep='\t')

    model_name = 'inception_big'
    fold = 5
    batch_size = 200
    epochs = 25
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

        model = getModelInception(maxlen, 4, max_features, embed_size,
                                  embedding_matrix)

        f1_val = F1Evaluation(validation_data=(kfold_X_valid, kfold_y_test),
                              interval=1)

        model.fit(kfold_X_train, kfold_y_train,
                  batch_size=batch_size,
                  epochs=epochs, verbose=1, callbacks=[f1_val])
        gc.collect()
        model.load_weights("best_weights.h5")

        y_pred += model.predict(x_test, batch_size=512,
                                verbose=1) / fold

    my_dict = {0: '人类作者', 1: '机器作者', 2: '机器翻译', 3: '自动摘要'}
    y_p = np.argmax(y_pred, 1)
    test['标签'] = np.vectorize(my_dict.get)(y_p)
    test.to_csv(
        f'../inputs/{model_name}_sub_bilistmcnn_{fold}_{batch_size}_{epochs}_cv.csv',
        columns=['id', '标签'], header=False, index=False)
