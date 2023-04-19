import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

testing=False
filename = 'submission'
content = pd.read_csv('../input/promoted_content.csv')

chunksize = 50000 
train = pd.read_csv('../input/clicks_train.csv',iterator=True,chunksize=chunksize) #Load data
print( 'Training')
for chunk in train:
	chunk=pd.merge(chunk,content,how='left',on='ad_id')	
	predictors=[x for x in chunk.columns if x not in ['display_id','clicked']]
	chunk=chunk.fillna(0.0)
	alg = RandomForestClassifier(random_state=1, n_estimators=3, min_samples_split=4, min_samples_leaf=2, warm_start=True)
	alg.fit(chunk[predictors], chunk["clicked"])#Fit the Algorithm
	if testing:
		break

train=''
print('Testing')
test= pd.read_csv('../input/clicks_test.csv',iterator=True,chunksize=chunksize) #Load data
predY=[]
for chunk in test:
	init_chunk_size=len(chunk)
	chunk=pd.merge(chunk,content,how='left',on='ad_id')
	chunk=chunk.fillna(0.0)
	chunk_pred=list(alg.predict_proba(chunk[predictors]).astype(float)[:,1])
	predY += chunk_pred
	if testing:
		break
print('Done Testing')

print('Preparing for Submission')	
test=''
test= pd.read_csv('../input/clicks_test.csv')
results=pd.concat((test,pd.DataFrame(predY)) ,axis=1,ignore_index=True)
print(results.head(10))
results.columns = ['display_id','ad_id','clicked']
#results=results[results['clicked'] > 0.0]
results = results.sort_values(by=['display_id','clicked'], ascending=[True, False])
results = results.reset_index(drop=True)
results=results[['display_id','ad_id']].groupby('display_id')['ad_id'].agg(lambda col: ' '.join(map(str,col)))
results.columns=[['display_id','ad_id']]
results.to_csv(filename+'.csv')	