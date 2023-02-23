import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_y = pd.read_csv("data/open/train_df.csv")
print(train_y.head())

classList = train_y['class'].unique()

print('class 개수:', len(classList))

labelList = train_y['label'].unique()
print('label 개수:', len(labelList))

labelCount = train_y[['class', 'label']].groupby('label').count().rename(columns={'class': 'count'})

anomaly_dict = {}
for className in classList:
    df = pd.DataFrame(labelCount[labelCount.index.str.contains(className)]).sort_values(by='count', ascending=False)
    anomaly_dict[className] = df


fig, axs = plt.subplots(15, 1, figsize=(15, 15*5))
fig.subplots_adjust(hspace = .3)
axs = axs.ravel()

for i, (className, df) in enumerate(anomaly_dict.items()):
    colors = ['red' for i in range(len(df.index))]
    colors[0] = 'green'
    axs[i].bar(df.index, df.iloc[:, 0], color=colors, alpha=0.5)
    axs[i].set_title(className, fontsize=20)
    for j, value in enumerate(df.iloc[:, 0]):
        axs[i].text(j, 20, df.iloc[:, 0][j], ha='center', fontsize=20)