import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from config import features_npz


def add_features(df):
    df['内容'] = df['内容'].apply(lambda x: str(x))
    df['total_length'] = df['内容'].apply(len)
    df['chinese'] = df['内容'].apply(lambda comment: sum(1 for c in comment if
                                                       re.fullmatch(
                                                           r'[\u4e00-\u9fff]+',
                                                           c)))
    df['chi_vs_length'] = df.apply(
        lambda row: float(row['chinese']) / float(row['total_length']),
        axis=1)
    df['num_words'] = df.内容.str.count('\S+')
    df['num_unique_words'] = df['内容'].apply(
        lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    return df


if __name__ == '__main__':
    train = pd.read_csv("../inputs/train.tsv", sep='\t')
    test = pd.read_csv("../inputs/vali.tsv", sep='\t')
    train = add_features(train)
    test = add_features(test)
    train_features = train[['chi_vs_length', 'words_vs_unique']].fillna(0)
    test_features = test[['chi_vs_length', 'words_vs_unique']].fillna(0)
    ss = StandardScaler()
    ss.fit(np.vstack((train_features, test_features)))
    train_features = ss.transform(train_features)
    test_features = ss.transform(test_features)
    np.savez_compressed(features_npz, train=train_features,
                        test=test_features)
