import pandas as pd
from tqdm import tqdm

path = '/data/lxkui/Amazon/office/'

df = pd.read_json(path + 'json2csv/Office_Products_5.json', orient='records', lines=True)
print(df.head(5))

df1 = pd.DataFrame(df, columns=['reviewerID', 'asin', 'overall', 'reviewText'])
print(df1.head(5))
df1.to_csv(path + 'json2csv/one.csv', header=False, index=False)

# oldID->newID: user
dict_user = {}
user_num = 0
users_id = list(df1['reviewerID'])
for i in range(len(users_id)):
    if users_id[i] not in dict_user:
        dict_user[users_id[i]] = user_num
        user_num += 1
df_user1 = pd.DataFrame.from_dict(dict_user, orient='index', columns=['new_id'])
df_user1 = df_user1.reset_index().rename(columns={'index': "old_id"})
print(df_user1.head(5))
df_user1.to_csv(path + "json2csv/user_old2new_id.csv", header=False, index=False)

# oldID->newID: item
dict_item = {}
item_num = 0
item_id = list(df1['asin'])
for i in range(len(item_id)):
    if item_id[i] not in dict_item:
        dict_item[item_id[i]] = item_num
        item_num += 1
df_item1 = pd.DataFrame.from_dict(dict_item, orient='index', columns=['new_id'])
df_item1 = df_item1.reset_index().rename(columns={'index': "old_id"})
print(df_item1.head(5))
df_item1.to_csv(path + "json2csv/item_old2new_id.csv", header=False, index=False)

user_list = [0] * len(users_id)
for i in range(len(users_id)):
    user_list[i] = dict_user[users_id[i]]
print(user_list)

item_list = [0] * len(item_id)
for i in range(len(item_id)):
    item_list[i] = dict_item[item_id[i]]
print(item_list)

# concat id and reviews
user_df = pd.DataFrame(user_list)
item_df = pd.DataFrame(item_list)
df = pd.read_csv(path + "json2csv/one.csv", header=None)
df.drop(0, axis=1, inplace=True)
df.drop(1, axis=1, inplace=True)
print(df.head(5))
df_two = pd.concat([user_df, item_df, df], axis=1, ignore_index=True)
print(df_two.head(5))
df_two = df_two.dropna(axis=0, how='any') 
df_two.to_csv(path + "json2csv/two.csv", header=False, index=False)

# get all reviews for each user
user_text = {}
for id, text in tqdm(df_two.groupby(0)):
    s = ''
    list_text = list(text[3])
    for i in list_text:
        s += str(i) + '*'
    user_text[id] = s
df_user_text = pd.DataFrame.from_dict(user_text, orient='index', columns=['text'])
df_user_text = df_user_text.reset_index().rename(columns={'index': "user_id"})
print(df_user_text.head(5))
df_user_text.to_csv(path + "json2csv/user_text.csv", header=False, index=False)

# get all reviews for each item
item_text = {}
for id, text in tqdm(df_two.groupby(1)):
    s = ''
    list_text = list(text[3])
    for i in list_text:
        s += str(i) + '*'
    item_text[id] = s
df_item_text = pd.DataFrame.from_dict(item_text, orient='index', columns=['text'])
df_item_text = df_item_text.reset_index().rename(columns={'index': "item_id"})
print(df_item_text.head(5))
df_item_text.to_csv(path + "json2csv/item_text.csv", header=False, index=False)
