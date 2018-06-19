import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from config import features_npz


def add_features(train):
    train["内容"] = train["内容"].fillna("无")
    train['num_words'] = train.内容.str.count('\S+')
    train['num_sents'] = train.内容.str.count(r'[。！？]')
    train['num_sents'] = train['num_sents'].fillna(0)
    train['num_sents'] = train['num_sents'] + 1
    train['words_per_sent'] = train['num_words'].astype(float)/train['num_sents'].astype(float)
    train['str_len'] = train.内容.apply(lambda x: np.sum([len(w) for w in str(x).split()]))
    train['mean_word_len'] = train.内容.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    train['chinese_word_count'] = train['内容'].apply(lambda comment: sum(1 for c in comment.split() if
                                                           re.fullmatch(r'[\u4e00-\u9fff]+', c)))
    train['num_unique_words'] = train['内容'].apply(lambda comment: len(set(w for w in comment.split())))
    train['words_vs_unique'] = train['num_unique_words'].astype(float) / train['num_words'].astype(float)
    train['chinese_word_count'] = train['内容'].apply(lambda comment: sum(1 for c in comment.split() if re.fullmatch(r'[\u4e00-\u9fff]+', c)))
    train['chi_word_rate'] = train.apply(lambda row: float(row['chinese_word_count']) / float(row['num_words']),axis=1)
    return train


if __name__ == '__main__':
    train = pd.read_csv("../inputs/train.tsv", sep='\t')
    test = pd.read_csv("../inputs/vali.tsv", sep='\t')
    train = add_features(train)
    test = add_features(test)
    train_features = train[['words_per_sent', 'chi_word_rate', 'words_vs_unique', 'mean_word_len']].fillna(0)
    test_features = test[['words_per_sent', 'chi_word_rate', 'words_vs_unique', 'mean_word_len']].fillna(0)
    ss = StandardScaler()
    ss.fit(np.vstack((train_features, test_features)))
    train_features = ss.transform(train_features)
    test_features = ss.transform(test_features)
    np.savez_compressed(features_npz, train=train_features,
                        test=test_features)
