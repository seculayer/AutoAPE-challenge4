from tqdm import tqdm
import numpy as np
import os
import sys
import builtins
import polars as pl
import pandas as pd
import time
import gc


train_df = pl.read_parquet('/kaggle/input/otto-train-and-test-data-for-local-validation/train.parquet')
test_df = pl.read_parquet('/kaggle/input/otto-train-and-test-data-for-local-validation/test.parquet')
labels_df = pl.read_parquet('/kaggle/input/otto-train-and-test-data-for-local-validation/test_labels.parquet')

def add_aid_pos_reverse(df):
    return df.select([
        pl.col('*'),
        pl.col('session').cumcount().reverse().over('session').alias('aid_pos_reverse')+1
    ])

def add_r_score(df):
    pos_alpha = 10
    r_score = pl.Series(1 / (2**((df['aid_pos_reverse']-1)/pos_alpha)))
    return df.with_columns(r_score.alias('r_score')).fill_nan(0)

def add_type_weighted_r_score(df):
    tw = {0:.2, 1:.3, 2:.4}
    tw_r_score = pl.Series(df['r_score'] * df['type'].apply(lambda x: tw[x]))
    df = df.with_columns(tw_r_score.alias('tw_r_score'))
    df = df.drop('r_score')
    return df

def apply(df, pipeline):
    for f in pipeline:
        df = f(df)
    return df

def transitions_shift(df):
    df = df.with_columns([
        pl.col(['aid']).shift(-1).over('session').prefix("next_"),
        pl.col(['type']).shift(-1).over('session').prefix("next_"),
        pl.col(['ts']).shift(-1).over('session').prefix("next_")
    ]).with_columns([
        pl.col("next_aid").fill_null(pl.col("aid")),
        pl.col("next_type").fill_null(pl.col("type")),
        pl.col("next_ts").fill_null(pl.col("ts")+1)
    ])
    
    time_diff = pl.Series(df['next_ts'] - df['ts']+1)

    # time_alpha = 60 sec * 60 min * 12 hours
    time_alpha = 60*60*12
    
    t_diff = 1/(2**((time_diff-1)/(time_alpha-1)))
    
    df = df.with_columns([
        t_diff.alias('time_diff'),
    ])    
    
    df = df.drop_nulls()
    
    return df

def eval_test_df(df):
    df = df.drop('ts')
    return df

pipeline = [
    add_aid_pos_reverse, 
    add_r_score,
    add_type_weighted_r_score
]




transitions_df = pl.concat([
    train_df,
    test_df
], how="vertical")

print("transitions_df")
transitions_df = apply(transitions_df, pipeline)

print("transitions_df shift")
transitions_df = transitions_shift(transitions_df)


print("test_df_transitions")
test_df_transitions = apply(eval_test_df(test_df), pipeline)

print("apply weights for aids in each session for initial states")
test_df_transitions = test_df_transitions.join(transitions_df, how='left', \
                               left_on=['session','aid','aid_pos_reverse'], \
                               right_on=['session','aid','aid_pos_reverse']).fill_null(0)
weight = pl.Series(test_df_transitions['tw_r_score']*test_df_transitions['time_diff'])
test_df_transitions = test_df_transitions.with_columns([weight.alias('weight')]) 

test_df_transitions = test_df_transitions[['session','aid','type','weight']]
transitions_df = transitions_df[['aid','type','next_aid','next_type']]


print("emission_probabilities")
emission_probabilities = transitions_df.groupby(['aid','type','next_type']).agg([
    pl.col('next_aid').count().alias('count')
])

emission_probabilities = emission_probabilities.select([
    pl.col('*'),
    pl.col('count').sum().over(['aid']).alias('total_count')
])

ep = emission_probabilities['count'] / emission_probabilities['total_count']
emission_probabilities = emission_probabilities.with_columns(pl.Series(ep).alias('ep')).fill_nan(0)
emission_probabilities = emission_probabilities.drop('count')
emission_probabilities = emission_probabilities.drop('total_count') 


print("transitions")
transitions = transitions_df.groupby(['aid','next_aid','type','next_type']).agg([
    pl.col('next_aid').count().alias('count')
])

transitions = transitions.select([
    pl.col('*'),
    pl.col('count').sum().over(['aid','type','next_type']).alias('total_count'),
])

tp = pl.Series((transitions['count'] / transitions['total_count']) )

transitions = transitions.with_columns(tp.alias('tp')).fill_nan(0)

transitions = transitions[['type','aid','next_type','next_aid','tp']]

## trimming for optimization
TOP_K = 100
transitions = transitions.sort('tp', reverse=True).groupby(['aid','type','next_type']).head(TOP_K)

def markov_decision_process(sessions):

    top_k = 20
    top_n = 50

    # forward (1st tour)
    ep = sessions.join(emission_probabilities, how='left', left_on=['aid','type'], right_on=['aid','type']).fill_null(0)
    tp = ep.join(transitions, how='left', left_on=['aid','type','next_type'],\
                       right_on=['aid','type','next_type']).fill_null(0)
    probabilities = tp.with_columns(pl.Series(tp['ep'] * tp['tp']).alias('prob_1'))
    
    candidates = probabilities[['session','next_aid','next_type','prob_1']] 
    candidates = candidates.rename({'next_aid':'aid','next_type':'type','prob_1':'score'})
    candidates = candidates.groupby(['session','aid','type']).agg([
            pl.col('score').sum().alias('score')
        ])
    candidates_first = candidates.sort('score', reverse=True).groupby(['session','type']).head(top_n)
    candidates_next = candidates.sort('score', reverse=True).groupby(['session','type']).head(top_k)

    # forward (2nd tour)
    ep = candidates_next.join(emission_probabilities, how='left', left_on=['aid','type'], right_on=['aid','type']).fill_null(0)
    tp = ep.join(transitions, how='left', left_on=['aid','type','next_type'],\
                       right_on=['aid','type','next_type']).fill_null(0)
    probabilities = tp.with_columns(pl.Series(tp['score'] * tp['ep'] * tp['tp']).alias('prob_2'))
    
    candidates_second = probabilities[['session','next_aid','next_type','prob_2']] 
    candidates_second = candidates_second.rename({'next_aid':'aid','next_type':'type','prob_2':'score'})
    candidates_second = candidates_second.groupby(['session','aid','type']).agg([
            pl.col('score').sum().alias('score')
        ])
    candidates_second = candidates_second.sort('score', reverse=True).groupby(['session','type']).head(top_n)

    # init recommends
    ses_init_aids = sessions.rename({'weight':'score'})[['session','type','aid','score']]
    
    return pl.concat([
        ses_init_aids,
        candidates_first,
        candidates_second
    ], how="vertical").groupby(['session','type','aid']).agg([
        pl.col('score').sum().alias('score')
    ]) 



VER = 1
CACHED = False
CHUNK_COUNT = 80

s = pl.Series(lag_test_df["session"])
smin = s.min()
smax = s.max()

if not CACHED:
    chunk_size = (smax-smin)/CHUNK_COUNT
    for c,x in enumerate(tqdm(range(smin, smax-1, round(chunk_size)))):
        ses_end = min(x+round(chunk_size)-1,smax)
        chunked_df = test_df_transitions.filter((pl.col("session")>=x)&(pl.col("session")<=ses_end))
        probs = markov_decision_process(chunked_df)
        probs.write_parquet(f"/kaggle/working/probs_tr_ver_{VER}_{c}.parquet")
    prob_set = pl.read_parquet(f"/kaggle/working/probs_tr_ver_{VER}_*")
else:
    prob_set = pl.read_parquet(f"/kaggle/working/probs_tr_ver_{VER}_*")



print("Weights for events")
score_clicks = pl.Series( prob_set['score'] + ( 1 * prob_set['type'] * prob_set['score'] ) )
score_carts = pl.Series( prob_set['score'] + ( 3 * prob_set['type'] * prob_set['score'] ) )
score_orders = pl.Series( prob_set['score'] + ( 6 * prob_set['type'] * prob_set['score'] ) )
 
print("Separate scores for each event")
prob_set = prob_set.with_columns([
    score_clicks.alias('score_clicks'),
    score_carts.alias('score_carts'),
    score_orders.alias('score_orders')
])

print("Merge and sum scores for to prevent duplicate aids")
prob_set = prob_set.groupby(['session','aid']).agg([
    pl.col('score_clicks').sum().alias('score_clicks'),
    pl.col('score_carts').sum().alias('score_carts'),
    pl.col('score_orders').sum().alias('score_orders')
])


click_predictions = prob_set.sort(['score_clicks'], reverse=True).groupby(['session']).agg([
    pl.col('aid').limit(20)
])
cart_predictions = prob_set.sort(['score_carts'], reverse=True).groupby(['session']).agg([
    pl.col('aid').limit(20)
])
order_predictions = prob_set.sort(['score_orders'], reverse=True).groupby(['session']).agg([
    pl.col('aid').limit(20)
])


#click preds
click_preds = click_predictions.to_pandas()
click_preds["session"] = click_preds.session.apply(lambda x: str(x)+"_clicks")
click_preds.columns = ["session_type", "labels"]
#cart preds
cart_preds = cart_predictions.to_pandas()
cart_preds["session"] = cart_preds.session.apply(lambda x: str(x)+"_carts")
cart_preds.columns = ["session_type", "labels"]
#order preds
order_preds = order_predictions.to_pandas()
order_preds["session"] = order_preds.session.apply(lambda x: str(x)+"_orders")
order_preds.columns = ["session_type", "labels"]

valid_set = pd.concat([click_preds,cart_preds,order_preds],ignore_index=True).sort_values("session_type")


print('- - -')
# Compute Recalls
score = 0
weights = [0.10, 0.30, 0.60]
labels = ['clicks','carts','orders']
tlab = labels_df.to_pandas()
for t in range(3):
    sub = valid_set.loc[valid_set.session_type.str.contains(labels[t])].copy()
    sub['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))    
    test_labels = tlab.loc[tlab['type']==labels[t]]
    test_labels = test_labels.loc[test_labels['session'].isin(sub['session'])]
    test_labels = test_labels.merge(sub, how='left', on=['session'])
    test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
    test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0,20)
    recall = test_labels['hits'].sum() /test_labels['gt_count'].sum()
    score += weights[t] * recall
    print(labels[t] + ' recall = ' + str(recall))
print('- - -')
print('Overall Recall ='+str(score))
print('- - -')

