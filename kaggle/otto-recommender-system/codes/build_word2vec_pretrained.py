import pandas as pd
import polars as pl
from gensim.models import Word2Vec

from fastparquet import ParquetFile

train = ParquetFile('data/train.parquet')
train = train.to_pandas()
test = ParquetFile('data/test.parquet')
test = test.to_pandas()

def map_column(df: pd.DataFrame, col_name: str):

    aid_sorted = sorted(list(df[col_name].unique()))
    mapping = {k: i + 2 for i, k in enumerate(aid_sorted)}
    inverse_mapping = {v: k for k, v in mapping.items()}

    df[col_name] = df[col_name].map(mapping)

    return df, mapping, inverse_mapping

print('did map column')

train, mapping, inverse_mapping = map_column(train, 'aid')
test['aid'] = test['aid'].map(mapping)

train.session = train.session.astype('int32')
train.ts = train.ts.astype('int32')
train.type =train.type.astype('uint8')

train = pl.DataFrame(train)
test = pl.DataFrame(test)

print('concat ready')

sentences_df = pl.concat([train, test]).groupby('session').agg(
    pl.col('aid').alias('sentence')
)

sentences = sentences_df['sentence'].to_list()

print('model ready')

w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4)
w2vec.save("/pretrained/word2vec.model")

print('end')