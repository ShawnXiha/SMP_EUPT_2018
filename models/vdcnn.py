from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda
from keras.layers.pooling import MaxPooling1D
from .vdcnn_layers import ConvBlockVDCNN
import tensorflow as tf

num_filters_default = [64, 128, 256, 512]  # from VDCNN paper


def VDCNN_model(input_shape, num_classes, num_words, emb_size, emb_matrix,
                num_filters=num_filters_default, top_k=8, emb_trainable=False):
    inputs = Input(shape=(input_shape,), dtype='int32', name='inputs')

    embedded_sent = Embedding(num_words, emb_size, weights=[emb_matrix],
                              trainable=emb_trainable, name='embs')(inputs)

    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(
        embedded_sent)

    for i in range(len(num_filters)):
        conv = ConvBlockVDCNN(conv.get_shape().as_list()[1:], num_filters[i])(
            conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    def k_max_pooling(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))

    k_max = Lambda(k_max_pooling, output_shape=(num_filters[-1] * top_k,))(
        conv)

    # fully-connected layers
    fc1 = Dropout(0.2)(
        Dense(4096, activation='relu', kernel_initializer='he_normal')(k_max))
    fc2 = Dropout(0.2)(
        Dense(2048, activation='relu', kernel_initializer='he_normal')(fc1))
    fc3 = Dense(num_classes, activation='softmax')(fc2)

    model = Model(inputs=inputs, outputs=fc3)
    return model


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
    input_shape, classes, num_words, emb_size, emb_matrix = maxlen, 4, max_features, 300, embedding_matrix
    data = np.load(data_npz)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_train = to_categorical(y_train, num_classes=None)
    X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train,
                                                  train_size=0.9,
                                                  random_state=233)
    batch_size = 256
    epochs = 100
    print("starting training model!!!")
    model_name = "vdcnn"
    gc.collect()
    K.clear_session()
    model = VDCNN_model(input_shape, classes, num_words, emb_size, emb_matrix)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['accuracy'])
    f1_val = F1Evaluation(validation_data=(X_val, y_val), interval=1)
    f1_val.set_name(f"{model_name}_withoutcv")
    model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=1,
              callbacks=[f1_val])
    gc.collect()
    model.load_weights(
        f"~/total_data/saved_weight/{f1_val.name}best_weights.h5")
    y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)

    y_p = np.argmax(y_pred, 1)
    test['标签'] = np.vectorize(my_dict.get)(y_p)
    np.save(
        f'~/total_data/pred_output/{model_name}_withoutcv_{batch_size}_{epochs}.npy',
        y_pred)
    test.to_csv(
        f'~/total_data/pred_output/{model_name}_withoutcv_{batch_size}_{epochs}.csv',
        columns=['id', '标签'], header=False, index=False)
