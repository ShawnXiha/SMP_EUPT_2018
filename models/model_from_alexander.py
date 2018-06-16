import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense,  Embedding, SpatialDropout1D, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from config import maxlen, max_features

def get_model_ala(features_in, embedding_matrix, clipvalue=1.,num_filters=40,dropout=0.5,embed_size=300):
    features_input = Input(shape=(features_in.shape[1],))
    inp = Input(shape=(maxlen, ))

    # Layer 1: concatenated fasttext and glove twitter embeddings.
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    # Uncomment for best result
    # Layer 2: SpatialDropout1D(0.5)
    x = SpatialDropout1D(dropout)(x)

    # Uncomment for best result
    # Layer 3: Bidirectional CuDNNLSTM
    x = Bidirectional(CuDNNLSTM(num_filters, return_sequences=True))(x)


    # Layer 4: Bidirectional CuDNNGRU
    x, x_h, x_c = Bidirectional(CuDNNGRU(num_filters, return_sequences=True, return_state = True))(x)

    # Layer 5: A concatenation of the last state, maximum pool, average pool and
    # two features: "Unique words rate" and "Rate of all-caps words"
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    x = concatenate([avg_pool, x_h, max_pool,features_input])

    # Layer 6: output dense layer.
    outp = Dense(4, activation="softmax")(x)

    model = Model(inputs=[inp, features_input], outputs=outp)
    adam = optimizers.adam(clipvalue=clipvalue)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    from config import embed_npz, features_npz, data_npz
    from sklearn.model_selection import train_test_split
    from keras.utils.np_utils import to_categorical

    model_name = 'alexander'
    np.random.seed(233)
    embedding_matrix = np.load(embed_npz)['arr_0']
    features = np.load(features_npz)
    train_features = features['train']
    data = np.load(data_npz)
    x_train = data['x_train']
    y_train = data['y_train']
    y_train = to_categorical(y_train, num_classes=None)
    filepath = f"../inputs/{model_name}.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
    callbacks_list = [checkpoint, early]
    model = get_model_ala(train_features, embedding_matrix)
    model.fit([x_train, train_features], y_train, validation_split=0.1,
              batch_size=512, epochs=30, verbose=1,
              callbacks=callbacks_list)
    # model.load_weights(filepath)
    # print('Predicting....')
    # y_pred = model.predict([data['x_test'], features['test']],batch_size=1024,verbose=1)
    #