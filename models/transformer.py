import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(
            lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)(
            [q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(
                    TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(
                    TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(
                    TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1],
                                   d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1],
                                   n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = [];
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head);
                attns.append(attn)
            head = Concatenate()(heads)
            attn = Concatenate()(attns)

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v,
                                                 dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid,
                                                     dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input,
                                               mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, \
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _
            in range(layers)]

    def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
        x = self.emb_layer(src_seq)
        if src_pos is not None:
            pos = self.pos_layer(src_pos)
            x = Add()([x, pos])
        if return_att: atts = []
        mask = Lambda(lambda x: GetPadMask(x, x))(src_seq)
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att: atts.append(att)
        return (x, atts) if return_att else x


class Transformer:
    def __init__(self, max_features, len_limit, d_model=256, d_out=4,
                 d_inner_hid=512, n_head=4, d_k=64, d_v=64, layers=2,
                 dropout=0.1):
        self.max_features = max_features
        self.len_limit = len_limit
        self.src_loc_info = True
        self.d_model = d_model
        self.model = None
        d_emb = d_model

        pos_emb = Embedding(len_limit, d_emb, trainable=False,
                            weights=[GetPosEncodingMatrix(len_limit, d_emb)])

        i_word_emb = Embedding(max_features, d_emb)
        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers,
                               dropout, \
                               word_emb=i_word_emb, pos_emb=pos_emb)
        self.flatten = Flatten()
        self.encoder2label = Dense(d_out, activation='softmax')

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile(self, optimizer='adam', active_layers=999):
        src_seq = Input(shape=(None,), dtype='int32')
        src_pos = Lambda(self.get_pos_seq)(src_seq)
        if not self.src_loc_info: src_pos = None

        enc_output = self.encoder(src_seq, src_pos,
                                  active_layers=active_layers)
        enc_output = self.flatten(enc_output)
        pred_p = self.encoder2label(enc_output)

        self.model = Model(inputs=src_seq, outputs=pred_p)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])


if __name__ == '__main__':
    from config import maxlen, max_features
    import gc
    import pandas as pd
    import numpy as np
    from keras import backend as K
    from config import embed_npz, features_npz, data_npz
    from sklearn.model_selection import KFold
    from keras.utils.np_utils import to_categorical
    from config import maxlen, max_features, embed_size
    from utils import F1Evaluation

    test = pd.read_csv("../inputs/vali.tsv", sep='\t')

    model_name = 'transformer'
    data = np.load(data_npz)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_train = to_categorical(y_train, num_classes=None)
    y_pred = np.zeros((test.shape[0], 4))
    kf = KFold(n_splits=10, shuffle=True, random_state=239)
    for train_index, test_index in kf.split(x_train):
        kfold_y_train, kfold_y_test = y_train[train_index], y_train[test_index]
        kfold_X_train = x_train[train_index]
        kfold_X_valid = x_train[test_index]

        gc.collect()
        K.clear_session()
        trans = Transformer(max_features, maxlen)
        trans.compile()

        f1_val = F1Evaluation(validation_data=(kfold_X_valid, kfold_y_test),
                              interval=1)

        trans.model.fit(kfold_X_train, kfold_y_train,
                        batch_size=512,
                        epochs=100, verbose=1, callbacks=[f1_val])
        gc.collect()
        trans.model.load_weights("best_weights.h5")

        y_pred += trans.model.predict(x_test, batch_size=1024,
                                      verbose=1) / 10

    my_dict = {0: '人类作者', 1: '机器作者', 2: '机器翻译', 3: '自动摘要'}
    y_p = np.argmax(y_pred, 1)
    test['标签'] = np.vectorize(my_dict.get)(y_p)
    test.to_csv(f'../inputs/{model_name}_sub_bilistmcnn_10_cv.csv',
                columns=['id', '标签'], header=False, index=False)
