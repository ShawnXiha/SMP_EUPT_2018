import pandas as pd
import numpy as np
from config import max_features, maxlen, embed_size, EMBEDDING_FILE, data_npz
from config import embed_npz

from keras.preprocessing import text, sequence

train = pd.read_csv("../inputs/train.tsv", sep='\t')
test = pd.read_csv("../inputs/vali.tsv", sep='\t')
X_train = train["内容"].fillna("无").str.lower()
y_train = train["标签"].values

X_test = test["内容"].fillna("无").str.lower()

tok = text.Tokenizer(num_words=max_features,
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n+——！，。？、~@#￥%……&*（）‘’”“：《》=【】；`')
tok.fit_on_texts(list(X_train) + list(X_test))
X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tok.word_index
# prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

y_lookup, y_train = np.unique(y_train, return_inverse=True)
np.savez_compressed(data_npz, x_test=x_test, x_train=x_train, y_train=y_train,
                    y_lookup=y_lookup)

np.savez_compressed(embed_npz, embedding_matrix)
