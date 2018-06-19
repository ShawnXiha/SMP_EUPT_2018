from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, MaxPooling1D
from keras.engine.topology import Layer


def inception(x, co, relu=True, norm=True):
    assert (co % 4 == 0)
    cos = [co / 4] * 4
    activatie = Sequential()
    if norm:
        activatie.add(BatchNormalization())
    if relu:
        activatie.add(Activation('relu'))

    branch1 = Sequential()
    branch2 = Sequential()
    branch3 = Sequential()
    branch4 = Sequential()
    branch1.add(Conv1D(cos[0], 1, stride=1))
    branch2.add(Conv1D(cos[1], 1))
    branch2.add(BatchNormalization())
    branch2.add(Activation('relu'))
    branch2.add(Conv1D(cos[1], 3, stride=1, padding=1))
    branch3.add(Conv1D(cos[2], 3, padding=1))
    branch3.add(BatchNormalization())
    branch3.add(Activation('relu'))
    branch3.add(Conv1D(cos[2], 5, stride=1, padding=2))
    branch4.add(Conv1D(cos[3], 3, stride=1, padding=1))
    branch1 = branch1(x)
    branch2 = branch2(x)
    branch3 = branch3(x)
    branch4 = branch4(x)
    result = activatie(concatenate([branch1, branch2, branch3, branch4]))
    return result


class Inception(Layer):
    def __init__(self, cin, co, relu=True, norm=True):
        super().__init__()
        assert (co % 4 == 0)
        cos = [co / 4] * 4


def getModelInception(input_shape, classes, num_words, maxlen, emb_size, emb_matrix,
                      emb_dropout=0.5, inception_dim=64, dense=False, emb_trainable=False):
    x_input = Input(shape=(input_shape,))

    emb = Embedding(num_words, emb_size, weights=[emb_matrix],
                    trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)

    content_conv = inception(emb, inception_dim)
    content_conv = inception(content_conv, inception_dim)
    content_conv = MaxPooling1D(maxlen)(content_conv)

    fc = Sequential()
    fc.add(Dense(32))
    fc.add(BatchNormalization())
    fc.add(Activation('relu'))
    fc.add(Dense(classes, activation='softmax'))
    outp = fc(content_conv)

    return Model(inputs=x_input, outputs=outp)
