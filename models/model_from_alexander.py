import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, \
    GlobalMaxPooling1D
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras import optimizers
from config import maxlen, max_features
from keras.callbacks import Callback
from sklearn.metrics import f1_score


class F1Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

        self.y_val = np.argmax(self.y_val, 1)
        self.max_score = 0
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=1024, verbose=1)

            score = f1_score(self.y_val, np.argmax(y_pred, 1), average='macro')
            print("\n F1-score - epoch: %d - score: %.6f \n" % (
                epoch + 1, score))
            if score > self.max_score:
                print(
                    "*** New High Score (previous: %.6f) \n" % self.max_score)
                self.model.save_weights("best_weights.h5")
                self.max_score = score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > 5:
                    print("Epoch %05d: early stopping, high score = %.6f" % (
                        epoch, self.max_score))
                    self.model.stop_training = True


def get_model_ala(features_in, embedding_matrix, clipvalue=1., num_filters=40,
                  dropout=0.5, embed_size=300):
    features_input = Input(shape=(features_in.shape[1],))
    inp = Input(shape=(maxlen,))

    # Layer 1: concatenated fasttext and glove twitter embeddings.
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)

    # Uncomment for best result
    # Layer 2: SpatialDropout1D(0.5)
    x = SpatialDropout1D(dropout)(x)

    # Uncomment for best result
    # Layer 3: Bidirectional CuDNNLSTM
    x = Bidirectional(CuDNNLSTM(num_filters, return_sequences=True))(x)

    # Layer 4: Bidirectional CuDNNGRU
    x, x_h, x_c = Bidirectional(
        CuDNNGRU(num_filters, return_sequences=True, return_state=True))(x)

    # Layer 5: A concatenation of the last state, maximum pool, average pool and
    # two features: "Unique words rate" and "Rate of all-caps words"
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    x = concatenate([avg_pool, x_h, max_pool, features_input])

    # Layer 6: output dense layer.
    outp = Dense(4, activation="softmax")(x)

    model = Model(inputs=[inp, features_input], outputs=outp)
    adam = optimizers.adam(clipvalue=clipvalue)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    import gc
    from keras import backend as K
    from config import embed_npz, features_npz, data_npz
    from sklearn.model_selection import KFold
    from keras.utils.np_utils import to_categorical

    test = pd.read_csv("../inputs/vali.tsv", sep='\t')

    model_name = 'alexander'
    np.random.seed(233)
    embedding_matrix = np.load(embed_npz)['arr_0']
    features = np.load(features_npz)
    train_features = features['train']
    data = np.load(data_npz)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    test_features = features['test']
    y_train = to_categorical(y_train, num_classes=None)
    y_pred = np.zeros((test.shape[0], 4))
    kf = KFold(n_splits=10, shuffle=True, random_state=239)
    for train_index, test_index in kf.split(x_train):
        kfold_y_train, kfold_y_test = y_train[train_index], y_train[test_index]
        kfold_X_train = x_train[train_index]
        kfold_X_features = train_features[train_index]
        kfold_X_valid = x_train[test_index]
        kfold_X_valid_features = train_features[test_index]

        gc.collect()
        K.clear_session()

        model = get_model_ala(train_features, embedding_matrix)

        f1_val = F1Evaluation(validation_data=(
            [kfold_X_valid, kfold_X_valid_features], kfold_y_test),
            interval=1)

        model.fit([kfold_X_train, kfold_X_features], kfold_y_train,
                  batch_size=512,
                  epochs=100, verbose=1, callbacks=[f1_val])
        gc.collect()
        model.load_weights("best_weights.h5")

        y_pred += model.predict([x_test, test_features], batch_size=1024,
                                verbose=1) / 10

    my_dict = {0: '人类作者', 1: '机器作者', 2: '机器翻译', 3: '自动摘要'}
    y_p = np.argmax(y_pred, 1)
    test['标签'] = np.vectorize(my_dict.get)(y_p)
    test.to_csv(f'../inputs/{model_name}_sub_bilistmcnn_10_cv.csv',
                columns=['id', '标签'], header=False, index=False)
