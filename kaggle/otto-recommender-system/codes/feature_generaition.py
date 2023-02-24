import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
# from google.cloud import storage
import polars as pl
import os
import gc
os.chdir('data/local_val')
#
# # multiprocessing
# # import psutil
# # N_CORES = psutil.cpu_count()     # Available CPU cores
# # print(f"N Cores : {N_CORES}")
# # from multiprocessing import Pool
df = pl.read_parquet('test.parquet')
print(datetime.fromtimestamp(df['ts'].min()))
print(datetime.fromtimestamp(df['ts'].max()))
print(df)

counts = df.groupby('session').count()
counts.columns = ['session', 'sess_cnt']

df = df.join(counts, on='session', how='left')
dts = pd.to_datetime(df['ts'].to_pandas(), unit='s')

df = df.with_column(pl.from_pandas(dts.dt.weekday.rename('day')))
df = df.with_column(pl.from_pandas(dts.dt.hour.rename('hour')))
df = df.with_column(pl.from_pandas((dts.dt.hour*100 + dts.dt.minute*100//60).rename('hm')))
## mean
hm_mean = df.groupby('session').mean().select(['session', 'hm'])
hm_mean.columns = ['session', 'hm_mean']
df = df.join(hm_mean, on='session', how='left')

## median
hm_med = df.groupby('session').median().select(['session', 'hm'])
hm_med.columns = ['session', 'hm_median']
df = df.join(hm_med, on='session', how='left')

temp = df.to_pandas()
temp['hm_std'] = temp.groupby('session')['hm'].transform('std')
temp['hm_std'] = temp['hm_std'].fillna(0)
df = df.with_column(pl.from_pandas(temp['hm_std']))

for i in range(7):
    df_temp_cnt = df.filter(pl.col('day') == i).groupby('session').count()
    df_temp_cnt.columns = ['session', f'day{i}cnt']
    df = df.join(df_temp_cnt, on='session', how='left')
    df = df.with_column(pl.col(f'day{i}cnt').fill_null(0))
    df = df.with_column(df[f'day{i}cnt']/df['sess_cnt'])
    del df_temp_cnt
    gc.collect()

for i in range(24):
    df_temp_cnt = df.filter(pl.col('hour') == i).groupby('session').count()
    df_temp_cnt.columns = ['session', f'hour{i}cnt']
    df = df.join(df_temp_cnt, on='session', how='left')
    df = df.with_column(pl.col(f'hour{i}cnt').fill_null(0))
    df = df.with_column(df[f'hour{i}cnt']/df['sess_cnt'])
    del df_temp_cnt
    gc.collect()

df = df.with_column(df['ts'].diff().rename('ts_diff'))
df_pd = df.to_pandas()
df_pd['index1'] = df_pd.index

temp = df_pd.groupby('session').first()
temp['flag'] = True
temp = temp[['flag', 'index1']]

df_pd = df_pd.merge(temp, how='left', left_on='index1', right_on='index1')
df_pd.drop(['index1'], inplace=True, axis=1)
df_pd.loc[df_pd['flag'] == True, 'ts_diff'] = -1
df_pd.drop(['flag'], inplace=True, axis=1)

df = pl.from_pandas(df_pd)
sess_cnt2 = df.filter(pl.col('ts_diff') > 1600).groupby('session').count()
sess_cnt2.columns = ['session', 'sess_cnt2']
sess_cnt2 = sess_cnt2.with_column(sess_cnt2['sess_cnt2'] + 1)
df = df.join(sess_cnt2, on='session', how='left')
df = df.with_column(pl.col('sess_cnt2').fill_null(1))

cl_cnt = df.filter(pl.col('type')==0).groupby('session').count()
cl_cnt.columns = ['session', 'cl_cnt']
ca_cnt = df.filter(pl.col('type')==1).groupby('session').count()
ca_cnt.columns = ['session', 'ca_cnt']
or_cnt = df.filter(pl.col('type')==2).groupby('session').count()
or_cnt.columns = ['session', 'or_cnt']

df = df.join(cl_cnt, on='session', how='left')
df = df.join(ca_cnt, on='session', how='left')
df = df.join(or_cnt, on='session', how='left')

df = df.with_column(pl.col('cl_cnt').fill_null(0))
df = df.with_column(pl.col('ca_cnt').fill_null(0))
df = df.with_column(pl.col('or_cnt').fill_null(0))

df = df.with_column((df['ca_cnt']/df['cl_cnt']).alias('ca_cl_ratio'))
df = df.with_column((df['or_cnt']/df['cl_cnt']).alias('or_cl_ratio'))
df = df.with_column((df['or_cnt']/df['ca_cnt']).alias('or_ca_ratio'))

df = df.with_column(pl.col('ca_cl_ratio').fill_nan(-1))
df = df.with_column(pl.col('or_cl_ratio').fill_nan(-1))
df = df.with_column(pl.col('or_ca_ratio').fill_nan(-1))

df['or_cl_ratio'].is_infinite().sum()

df = df.with_column(
    pl.when(pl.col(['ca_cl_ratio', 'or_cl_ratio', 'or_ca_ratio']).is_infinite())
    .then(0)
    .otherwise(pl.col(['ca_cl_ratio', 'or_cl_ratio','or_ca_ratio']))
    .keep_name()
)

df['or_cl_ratio'].is_infinite().sum()

ss_ts_max = df.groupby('session').max()[['session', 'ts']]
ss_ts_max.columns = ['session', 'ts_max']
ss_ts_min = df.groupby('session').min()[['session', 'ts']]
ss_ts_min.columns = ['session', 'ts_min']
ss_ts_mean = df.groupby('session').mean()[['session', 'ts']]
ss_ts_mean.columns = ['session', 'ts_mean']

df = df.join(ss_ts_max, on='session', how='left')
df = df.join(ss_ts_min, on='session', how='left')
df = df.join(ss_ts_mean, on='session', how='left')

def check_df_ok(df):
    for col in df.columns:
        print(col, end=' ')
        print(df[col].is_null().sum(), end=' ')
        try:
            print(df[col].is_nan().sum(), end=' ')
        except:
            print('exc', end=' ')
        try:
            print(df[col].is_infinite().sum(), end=' ')
        except:
            print('exc', end=' ')
        print('')
check_df_ok(df)

df = df.groupby('session').first().sort(pl.col('session'))
df = df.drop(['aid', 'ts', 'type', 'day', 'hour', 'hm', 'ts_diff'])
print(df.head())
df.write_parquet('sess_feature.parquet')


df = pl.concat([pl.read_parquet('train.parquet'),
                pl.read_parquet('test.parquet')])

counts = df.groupby('aid').count()
counts.columns = ['aid', 'aid_cnt']
df = df.join(counts, on='aid', how='left')

dts = pd.to_datetime(df['ts'].to_pandas(), unit='s')

df = df.with_column(pl.from_pandas(dts.dt.weekday.rename('day')))
df = df.with_column(pl.from_pandas(dts.dt.hour.rename('hour')))
df = df.with_column(pl.from_pandas((dts.dt.hour*100 + dts.dt.minute*100//60).rename('hm')))

hm_mean = df.groupby('aid').mean().select(['aid', 'hm'])
hm_mean.columns = ['aid', 'aid_hm_mean']
df = df.join(hm_mean, on='aid', how='left')
hm_med = df.groupby('aid').median().select(['aid', 'hm'])
hm_med.columns = ['aid', 'aid_hm_median']
df = df.join(hm_med, on='aid', how='left')
temp = df.to_pandas()
temp['aid_hm_std'] = temp.groupby('aid')['hm'].transform('std')
temp['aid_hm_std'] = temp['aid_hm_std'].fillna(0)
df = df.with_column(pl.from_pandas(temp['aid_hm_std']))

for i in range(7):
    df_temp_cnt = df.filter(pl.col('day') == i).groupby('aid').count()
    df_temp_cnt.columns = ['aid', f'aid_day{i}cnt']
    df = df.join(df_temp_cnt, on='aid', how='left')
    df = df.with_column(pl.col(f'aid_day{i}cnt').fill_null(0))
    df = df.with_column(df[f'aid_day{i}cnt']/df['aid_cnt'])
    del df_temp_cnt
    gc.collect()

for i in range(24):
    df_temp_cnt = df.filter(pl.col('hour') == i).groupby('aid').count()
    df_temp_cnt.columns = ['aid', f'aid_hour{i}cnt']
    df = df.join(df_temp_cnt, on='aid', how='left')
    df = df.with_column(pl.col(f'aid_hour{i}cnt').fill_null(0))
    df = df.with_column(df[f'aid_hour{i}cnt']/df['aid_cnt'])
    del df_temp_cnt
    gc.collect()

check_df_ok(df)
cl_cnt = df.filter(pl.col('type')==0).groupby('aid').count()
cl_cnt.columns = ['aid', 'aid_cl_cnt']
ca_cnt = df.filter(pl.col('type')==1).groupby('aid').count()
ca_cnt.columns = ['aid', 'aid_ca_cnt']
or_cnt = df.filter(pl.col('type')==2).groupby('aid').count()
or_cnt.columns = ['aid', 'aid_or_cnt']
df = df.join(cl_cnt, on='aid', how='left')
df = df.join(ca_cnt, on='aid', how='left')
df = df.join(or_cnt, on='aid', how='left')
df = df.with_column(pl.col('aid_cl_cnt').fill_null(0))
df = df.with_column(pl.col('aid_ca_cnt').fill_null(0))
df = df.with_column(pl.col('aid_or_cnt').fill_null(0))

df = df.with_column((df['aid_ca_cnt']/df['aid_cl_cnt']).alias('aid_ca_cl_ratio'))
df = df.with_column((df['aid_or_cnt']/df['aid_cl_cnt']).alias('aid_or_cl_ratio'))
df = df.with_column((df['aid_or_cnt']/df['aid_ca_cnt']).alias('aid_or_ca_ratio'))

df = df.with_column(pl.col('aid_ca_cl_ratio').fill_nan(-1))
df = df.with_column(pl.col('aid_or_cl_ratio').fill_nan(-1))
df = df.with_column(pl.col('aid_or_ca_ratio').fill_nan(-1))

df['aid_or_cl_ratio'].is_infinite().sum()

df = df.with_column(
    pl.when(pl.col(['aid_ca_cl_ratio',
                    'aid_or_cl_ratio',
                    'aid_or_ca_ratio']).is_infinite())
    .then(0)
    .otherwise(pl.col(['aid_ca_cl_ratio',
                       'aid_or_cl_ratio',
                       'aid_or_ca_ratio']))
    .keep_name()
)

df['aid_or_cl_ratio'].is_infinite().sum()

aid_ts_max = df.groupby('aid').max()[['aid', 'ts']]
aid_ts_max.columns = ['aid', 'aid_ts_max']
aid_ts_min = df.groupby('aid').min()[['aid', 'ts']]
aid_ts_min.columns = ['aid', 'aid_ts_min']
aid_ts_mean = df.groupby('aid').mean()[['aid', 'ts']]
aid_ts_mean.columns = ['aid', 'aid_ts_mean']

df = df.join(aid_ts_max, on='aid', how='left')
df = df.join(aid_ts_min, on='aid', how='left')
df = df.join(aid_ts_mean, on='aid', how='left')
df = df.drop(['session', 'ts', 'type', 'day', 'hour', 'hm'])
df = df.groupby('aid').first()
print(df.head())
df.write_parquet('aid_features.parquet')