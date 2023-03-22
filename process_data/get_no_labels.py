import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
path = '/data/lxkui/Amazon/office/'

df = pd.read_csv(path+'embedding/three.csv', header=None)
# df = pd.read_csv(path + 'tfidf_three_20_3.csv', header=None)
df = pd.DataFrame(df.iloc[:, :3])
item_ids = df[1].unique()
nolabel_sample = 7
no_list = []
for reviewID, hist in df.groupby(0):
    pos_list = hist[1].tolist()
    candidate_set = list(set(item_ids) - set(pos_list))
    if not candidate_set:
        continue
    neg_list = np.random.choice(candidate_set, size=len(pos_list) * nolabel_sample, replace=True)
    for i in range(len(neg_list)):
        no_list.append((reviewID, neg_list[i]))
df_no = pd.DataFrame(no_list)
df_no = shuffle(df_no, random_state=4)
print(df_no.shape)
df_no.to_csv(path+'embedding/no_labels.csv', header=False, index=False)

