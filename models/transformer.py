from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dropout, \
    Dense
from keras import Model


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                2 * K.arange(self.size / 2, dtype='float32' \
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]),
                              1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)],
                                    2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (
            -1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (
            -1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (
            -1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def getModel(maxlen, classes, max_features, embedding_size, embedding_matrix):
    S_inputs = Input(shape=(maxlen,), dtype='int32')
    embeddings = Embedding(max_features, embedding_size,
                           weights=[embedding_matrix])(S_inputs)
    embeddings = Position_Embedding()(embeddings)
    O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    outputs = Dense(classes, activation='softmax')(O_seq)

    model = Model(inputs=S_inputs, outputs=outputs)
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
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
    model_name = "tranformer"
    gc.collect()
    K.clear_session()
    model = getModel(input_shape, classes, num_words, emb_size, emb_matrix)
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
