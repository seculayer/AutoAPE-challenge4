import polars as pl
import os
import xgboost as xgb
import numpy as np

os.chdir('data/parquets')

item_features = pl.read_parquet('submit_aid_features.parquet')
user_features = pl.read_parquet('submit_sess_feature.parquet')

need_features = []
for i in item_features.columns[1:]:
    need_features.append(i)

for i in user_features.columns:
    need_features.append(i)

test_candidates = pl.read_parquet('test.parquet')
for_predictions = test_candidates[['session','aid']]
for_predictions = for_predictions.to_pandas()
test_candidates = test_candidates[:,:-2]
test_candidates = test_candidates.join(item_features, on='aid', how='left').fill_nan(-1)
test_candidates = test_candidates.join(user_features, on='session', how='left').fill_nan(-1)


test_candidates = test_candidates[:, need_features]
test_candidates = test_candidates.to_pandas()
test_candidates = test_candidates.sort_values("session").reset_index(drop=True)
test_candidates = test_candidates.drop(["session"], axis=1)

os.chdir('../XGB_models')


preds = np.zeros(len(test_candidates))
for fold in range(5):
    model = xgb.Booster()
    model.load_model(f'XGB_fold{fold}_clicks.xgb')
    model.set_param({'predictor': 'gpu_predictor'})
    dtest = xgb.DMatrix(data=test_candidates)
    preds += model.predict(dtest)/5
predictions = for_predictions.copy()
predictions['pred'] = preds

predictions = predictions.sort_values(['session','pred'], ascending=[True,False]).reset_index(drop=True)
print(predictions)
predictions['n'] = predictions.groupby('session').aid.cumcount().astype('int8')
print(predictions)
predictions = predictions.loc[predictions.n<20]
sub = predictions.groupby('session').aid.apply(list)
sub = sub.to_frame().reset_index()
sub.aid = sub.aid.apply(lambda x: " ".join(map(str,x)))
sub.columns = ['session_type','labels']
sub.session_type = sub.session_type.astype('str')+ '_clicks'
print(sub)


