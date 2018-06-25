from keras.layers import Dense, LSTM, Bidirectional, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D, Reshape, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, \
    GlobalMaxPooling1D


def BiLSTM_2DCNN(maxlen, classes, max_features, embed_size, embedding_matrix,
                 lstm_units=256):
    conv_filters = 32
    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = Embedding(max_features, embed_size,
                                   input_length=maxlen,
                                   weights=[embedding_matrix],
                                   trainable=False)(sequence_input)
    x = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(
        embedded_sequences)
    x = Dropout(0.1)(x)
    x = Reshape((2 * maxlen, lstm_units, 1))(x)
    x = Conv2D(conv_filters, (3, 3))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    preds = Dense(classes, activation='softmax')(x)
    model = Model(sequence_input, preds)
    return model


def Conv2D_block(reshape, sequence_length, embedding_dim):
    filter_sizes = [3, 4, 5]
    num_filters = 32

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim),
                    padding='valid', kernel_initializer='he_uniform',
                    activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim),
                    padding='valid', kernel_initializer='he_uniform',
                    activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim),
                    padding='valid', kernel_initializer='he_uniform',
                    activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1),
                          strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1),
                          strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1),
                          strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)(
        [maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    return flatten


def Art_CNN(maxlen, classes, max_features, embed_size, embedding_matrix):
    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(max_features, embed_size, weights=[embedding_matrix],
                          input_length=maxlen, trainable=False)(sequence_input)
    x = SpatialDropout1D(0.2)(embedding)
    reshape = Reshape((maxlen, embed_size, 1))(x)
    x = Conv2D_block(reshape, maxlen, embed_size)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    preds = Dense(classes, activation='softmax')(x)
    model = Model(sequence_input, preds)
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
    for model, model_name in {BiLSTM_2DCNN: 'BiLSTM_2DCNN',
                              Art_CNN: 'Art_CNN'}.items():
        print(f"starting training model:{model_name}!!!")
        gc.collect()
        K.clear_session()
        model = model(input_shape, classes, num_words, emb_size, emb_matrix)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                      metrics=['accuracy'])
        f1_val = F1Evaluation(validation_data=(X_val, y_val), interval=1)
        f1_val.set_name(f"{model_name}_withoutcv")
        model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs,
                  verbose=1,
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
