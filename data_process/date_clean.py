import pandas as pd
import jieba

train_df = pd.read_json("../inputs/training.txt",
                        orient='records', lines=True)
train_df.drop_duplicates(subset=['内容'], inplace=True)
vali_df = pd.read_json("../inputs/validation.txt",
                       orient='records', lines=True)

clean_content = lambda text: " ".join(" ".join(jieba.cut(text)).split())
train_df['内容'] = train_df['内容'].apply(clean_content)
train_df['内容'] = train_df['内容'].apply(clean_content)
train_df.to_csv("../inputs/train.tsv", index=False, sep='\t')
vali_df.to_csv("../inputs/vali.tsv", index=False, sep='\t')
