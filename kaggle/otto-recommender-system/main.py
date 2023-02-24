import pandas as pd
import polars as pl
import os
import xgboost as xgb
from sklearn.model_selection import GroupKFold


os.chdir('data/local_val')

train = pl.read_parquet("train.parquet")
valid = pl.read_parquet("test.parquet")
valid_labels = pl.read_parquet("test_labels.parquet")

item_features = pl.read_parquet('aid_features.parquet')
user_features = pl.read_parquet('sess_feature.parquet')

need_features = []
for i in item_features.columns[1:]:
    need_features.append(i)

for i in user_features.columns[:]:
    need_features.append(i)


event_paths = {
    "clicks": "valid_click_candidates.parquet",
    "carts": "valid_carts_candidates.parquet",
    "orders": "valid_buys_candidates.parquet"
}

def train_ranker(event, df_cands, n_splits=5):

    skf = GroupKFold(n_splits=n_splits)
    FEATURES = need_features
        # ['session', 'item_item_count', 'item_user_count', 'item_buy_ratio', 'user_user_count', 'user_item_count', 'user_buy_ratio']
    TARGET = "target"
    for fold,(train_idx, valid_idx) in enumerate(skf.split(df_cands, df_cands['target'], groups=df_cands['session'])):

        X_train = df_cands.loc[train_idx, FEATURES]
        y_train = df_cands.loc[train_idx, TARGET]
        X_valid = df_cands.loc[valid_idx, FEATURES]
        y_valid = df_cands.loc[valid_idx, TARGET]

        X_train = X_train.sort_values("session").reset_index(drop=True)
        X_valid = X_valid.sort_values("session").reset_index(drop=True)

        train_group = X_train.groupby('session').session.agg('count').values
        valid_group = X_valid.groupby('session').session.agg('count').values

        X_train = X_train.drop(["session"], axis=1)
        X_valid = X_valid.drop(["session"], axis=1)

        dtrain = xgb.DMatrix(X_train, y_train, group=train_group) # [50] * (len(train_idx)//50)
        dvalid = xgb.DMatrix(X_valid, y_valid, group=valid_group) # [50] * (len(valid_idx)//50)
        xgb_parms = {
            'objective':'rank:pairwise',
            'tree_method':'hist',
            'random_state': 42,
            'learning_rate': 0.1,
            "colsample_bytree": 0.8,
            'eta': 0.05,
            'max_depth': 7,
            'subsample': 0.75,
            # n_estimators=110,
        }
        model = xgb.train(
            xgb_parms,
            dtrain=dtrain,
            evals=[(dtrain,'train'), (dvalid,'valid')],
            num_boost_round=100,
            verbose_eval=20
        )
        model.save_model(f'XGB_fold{fold}_{event}.xgb')

NEGATIVE_FRAC = 0.15


for event, path in event_paths.items():
    print(f"Started ranking model for: {event}")
    # Reading the candidates
    df_cands = pl.read_parquet(path)
    df_cands = df_cands.explode("candidates").with_columns([
        pl.col("session").cast(pl.Int32),
        pl.col("candidates").cast(pl.Int32).alias("aid")
    ]).drop("candidates").unique(subset=["session", "aid"])
    # Joining the item features
    df_cands = df_cands.join(item_features, on='aid', how='left').fill_nan(-1)
    # Joining the user features
    df_cands = df_cands.join(user_features, on='session', how='left').fill_nan(-1)
    cand_labels = valid_labels.filter(valid_labels["type"] == event).explode("ground_truth").with_columns([
        pl.col("session").cast(pl.Int32),
        pl.col("ground_truth").cast(pl.Int32)# .alias("aid")
    ]).rename({"ground_truth": "aid"})
    cand_labels = cand_labels.with_column(pl.lit(1).alias("target").cast(pl.Int8)).drop("type")
    # Joining the labels
    df_cands = df_cands.join(cand_labels, on=["session", "aid"], how="left").fill_null(0)
    # Negative sampling
    df_cands = pl.concat([
        df_cands.filter(df_cands["target"] == 0).sample(frac=NEGATIVE_FRAC, seed=42),
        df_cands.filter(df_cands["target"] == 1)
    ])
    print(df_cands.groupby("target").agg(pl.count()))
    df_cands = df_cands.to_pandas()
    df_cands = df_cands.sort_values("session").reset_index(drop=True)
    print(f"Event: {event} - started training...")

    train_ranker(event, df_cands)