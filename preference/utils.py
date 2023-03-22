import torch
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from collections import defaultdict

def train_co_iter(model, optimizer, data_loder, criterion, device, min_val, max_val, dic_review, dic_ui_review, dic_average_review,
                  is_user=True):
    model.train()
    total_loss = 0
    total_len = 0

    for index, (x_u, x_i, y) in enumerate(data_loder):


        if is_user:
            text_list = x_u.tolist()
        else:
            text_list = x_i.tolist()
        for i, x in enumerate(text_list):
            text_list[i] = dic_review[x]
        text_list = torch.FloatTensor(text_list)


        user_list, item_list = x_u.tolist(), x_i.tolist()
        ui_review_list = []
        
        flag = 0 if is_user else 1
        for tmp in zip(user_list, item_list):
            if tmp not in dic_ui_review:
                dic_ui_review[tmp] = sum(dic_average_review[tmp[flag]], [])
            ui_review_list.append(dic_ui_review[tmp])
        ui_review_list = torch.FloatTensor(ui_review_list)

        x_u, x_i, y, review, ui_review = x_u.to(device), x_i.to(device), y.to(device), text_list.to(
            device), ui_review_list.to(device)
        y = (y - min_val) / (max_val - min_val) + 0.01
        y_pre = model(x_u, x_i, review, ui_review)

        loss = criterion(y.view(-1, 1), y_pre)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_pre)
        total_len += len(y_pre)

    loss = total_loss / total_len
    return loss


def val_co_iter(model, data_loader, device, min_val, max_val, dic_review, dic_ui_review, dic_average_review,is_user=True):
    model.eval()
    labels, predicts = list(), list()

    with torch.no_grad():
        for x_u, x_i, y in data_loader:

            if is_user:
                text_list = x_u.tolist()
            else:
                text_list = x_i.tolist()
            for i, x in enumerate(text_list):
                text_list[i] = dic_review[x]
            text_list = torch.FloatTensor(text_list)

            user_list, item_list = x_u.tolist(), x_i.tolist()
            ui_review_list = []
            flag = 0 if is_user else 1
            for tmp in zip(user_list, item_list):
                if tmp not in dic_ui_review:
                    dic_ui_review[tmp] = dic_average_review[tmp[flag]]
                ui_review_list.append(dic_ui_review[tmp])
            ui_review_list = torch.FloatTensor(ui_review_list)

            x_u, x_i, y, review, ui_review = x_u.to(device), x_i.to(device), y.to(device), text_list.to(
                device), ui_review_list.to(device)
            y_pre = model(x_u, x_i, review, ui_review)
            y_pre = min_val + (y_pre - 0.01) * (max_val - min_val)
            y_pre = torch.where(y_pre > 5.0, torch.full_like(y_pre, 5.0), y_pre)
            y_pre = torch.where(y_pre < 1.0, torch.full_like(y_pre, 1.0), y_pre)
            labels.extend(y.tolist())
            predicts.extend(y_pre.tolist())
    mse = mean_squared_error(np.array(labels), np.array(predicts))

    return mse, predicts, labels


def predict(model, data_loader, device, min_val, max_val, dic_review, dic_ui_review, is_user=True):
    model.eval()
    predicts = list()

    with torch.no_grad():
        for x_u, x_i in data_loader:

            
            if is_user:
                text_list = x_u.tolist()
                text_ui_list = x_u.tolist()
            else:
                text_list = x_i.tolist()
                text_ui_list = x_i.tolist()
            for i, x in enumerate(text_list):
                text_list[i] = dic_review[x]
                text_ui_list[i] = dic_ui_review[x]
            text_list = torch.FloatTensor(text_list)
            text_ui_list = torch.FloatTensor(text_ui_list).squeeze(1)


            x_u, x_i, review, ui_review = x_u.to(device), x_i.to(device), text_list.to(
                device), text_ui_list.to(device)
            y_pre = model(x_u, x_i, review, ui_review)
            y_pre = min_val + (y_pre - 0.01) * (max_val - min_val)
            y_pre = torch.where(y_pre > 5.0, torch.full_like(y_pre, 5.0), y_pre)
            y_pre = torch.where(y_pre < 1.0, torch.full_like(y_pre, 1.0), y_pre)
            predicts.extend(y_pre.tolist())

    return predicts



def get_review_dic(df_review, review_len, u_or_i=True):
    
    review_count = int(max(df_review[1].groupby(df_review[0]).count()) * review_len)

    
    tmp = 0 if u_or_i else 1
    dic = defaultdict(list)
    for id, text_vec in df_review.groupby(tmp):
        text_vec = np.array(text_vec)
        for i in text_vec:
            if len(dic[id]) >= review_count:
                continue
            dic[id].append(list(i[3:]))

    
    for k, v in dic.items():
        if len(v) < review_count:
            for i in range(review_count - len(v)):
                dic[k].append([0] * 50)

    return dic


def get_ui_review_dic(df):
    dic_ui_review = defaultdict(list)
    df = np.array(df)
    for line in df:
        dic_ui_review[(int(line[0]), int(line[1]))] = list(line[3:])
    return dic_ui_review


def get_dic(df):
    dic_review = defaultdict(list)
    df = np.array(df)
    for line in df:
        dic_review[int(line[0])] = list(line[1:])
    return dic_review


def get_average_dic(df, u_or_i='user'):
    df = pd.DataFrame(df)
    tmp = 0 if u_or_i == 'user' else 1
    dic_average = defaultdict(list)
    for reviewID, hist in df.groupby(tmp):
        hist.reset_index(inplace=True, drop=True)
        hist = pd.DataFrame(hist.iloc[:, 3:])
        average_list = np.array([0.0] * 50).reshape((1, -1))
        length = len(hist)
        for k, v in hist.iterrows():
            v = np.array(v).reshape((1, -1))
            average_list += v
        average_list = (average_list / length).tolist()
        dic_average[reviewID] = average_list
    return dic_average


def linearRegression(data):
    X = np.array(data[:, 0:-1], dtype=np.float64)
    y = np.array(data[:, -1], dtype=np.float64)

    model = LinearRegression()
    model.fit(X, y)

    a, b = model.coef_
    return a, b, model.intercept_


def join_two_res(res1, res2, a=0.5, b=0.5, bias=0.):
    a = a
    b = b
    c = [0] * len(res1)
    for i in range(len(res1)):
        c[i] = a * float(res1[i]) + b * float(res2[i]) + bias
    return c
