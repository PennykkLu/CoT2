
import numpy as np
import pandas as pd
import random
import torch
import time
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from preference.utils import get_ui_review_dic, get_dic, get_average_dic, val_co_iter, join_two_res, predict
from preference.fm_model import FmDataset, FmDatasetNoLabels, FmReviewUI
from preference.train_fm import TrainFm
from hyperopt import STATUS_OK


seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


org_path = '/data/lxkui/Amazon/'
path = org_path + 'Arts_Crafts_and_Sewing/'
# path = org_path + 'instrument/'
# path = org_path + 'office/'
# path = org_path + 'videogame/'
model_path = '../save_model/CoFM_Agreement_torch/'
result_path = 'result/art_Agreement_82.csv'
# learning_rate = 5e-3
# weight_decay = 1e-4
epochs = 40
batch_size = 512
# mini_batch_size = 100
# tao = 0.8
epoch_k = 10
min_val, max_val = 1.0, 5.0
device = torch.device('cuda:2')
train_ratio = 0.8


df = pd.read_csv(path + 'embedding/three.csv', header=None)
u_max_id, i_max_id = max(df[0]) + 1, max(df[1]) + 1
print(df.shape, max(df[0]), max(df[1]))
df = shuffle(df, random_state=3)
df_train_val, df_test = pd.DataFrame(df[:int(df.shape[0] * train_ratio)]), \
                                       pd.DataFrame(df[int(df.shape[0] * train_ratio):int(df.shape[0] * 1)])
print(df_train_val.shape[0],df_test.shape[0])


df_user_review = pd.read_csv(path + 'embedding/user_embedding_vec.csv', header=None)
dic_user_review = get_dic(df_user_review)
df_item_review = pd.read_csv(path + 'embedding/item_embedding_vec.csv', header=None)
dic_item_review = get_dic(df_item_review)
dic_ui_review = get_ui_review_dic(df)
dic_user_average_review = get_average_dic(df, 'user')
dic_item_average_review = get_average_dic(df, 'item')

df_no_labels = pd.read_csv(path + 'embedding/no_labels.csv', header=None)


def cot2(para):
    # learning_rate, weight_decay, tao, mini_batch_size = para['learning_rate'], para['weight_decay'], para['tao'], para[
    #     'mini_batch_size']
    learning_rate=0.01
    weight_decay=0.001
    tao=0.8
    mini_batch_size =para

    x, y = df_train_val.iloc[:, :2], df_train_val.iloc[:, 2]
    x_train_user, x_val_user, y_train_user, y_val_user = train_test_split(x, y, test_size=0.2, random_state=4) 
    x_train_item, x_val_item, y_train_item, y_val_item = train_test_split(x, y, test_size=0.2, random_state=3) 
    user_train_data, user_val_data = pd.concat([x_train_user, y_train_user], axis=1), pd.concat(
        [x_val_user, y_val_user], axis=1)
    item_train_data, itm_val_data = pd.concat([x_train_item, y_train_item], axis=1), pd.concat([x_val_item, y_val_item],
                                                                                               axis=1)
    print(x_train_user.shape, x_val_user.shape, y_train_user.shape, y_val_user.shape)
    x_train_user_u, x_train_user_i, x_train_user_rating = np.array(x_train_user[0]), np.array(x_train_user[1]), \
                                                          np.array(y_train_user).astype(np.float32)
    x_val_user_u, x_val_user_i, x_val_user_rating = np.array(x_val_user[0]), np.array(x_val_user[1]), np.array(
        y_val_user).astype(np.float32)
    x_train_item_u, x_train_item_i, x_train_item_rating = np.array(x_train_item[0]), np.array(
        x_train_item[1]), np.array(y_train_item).astype(np.float32)
    x_val_item_u, x_val_item_i, x_val_item_rating = np.array(x_val_item[0]), np.array(x_val_item[1]), np.array(
        y_val_item).astype(np.float32)
    test_u, test_i, test_rating = np.array(df_test[0]), np.array(df_test[1]), np.array(df_test[2]).astype(np.float32)


    train_user_loader = DataLoader(FmDataset(x_train_user_u, x_train_user_i, x_train_user_rating),
                                   batch_size=batch_size, shuffle=True)
    val_user_loader = DataLoader(FmDataset(x_val_user_u, x_val_user_i, x_val_user_rating), batch_size=batch_size)
    train_item_loader = DataLoader(FmDataset(x_train_item_u, x_train_item_i, x_train_item_rating),
                                   batch_size=batch_size, shuffle=True)
    val_item_loader = DataLoader(FmDataset(x_val_item_u, x_val_item_i, x_val_item_rating), batch_size=batch_size)
    test_loader = DataLoader(FmDataset(test_u, test_i, test_rating), batch_size=batch_size)

    # initialize model
    model_user = FmReviewUI(u_id_len=u_max_id, i_id_len=i_max_id, id_embedding_dim=256).to(device)
    optimizer_user = torch.optim.Adam(model_user.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func_user = torch.nn.MSELoss().to(device)

    model_item = FmReviewUI(u_id_len=u_max_id, i_id_len=i_max_id, id_embedding_dim=256).to(device)
    optimizer_item = torch.optim.Adam(model_item.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func_item = torch.nn.MSELoss().to(device)

    # training model
    tmp = 0 # the position of unlabelled data
    mse_plt_list_user, mse_plt_list_item = [], []
    test_mse_plt_list_user, test_mse_plt_list_item = [], []
    regression_mse_plt_list = []
    best_regression_mse, best_regression_epoch = 10, 0
    user_train_data_len,item_train_data_len = [],[]

    fm = TrainFm(u_max_id, i_max_id, learning_rate, weight_decay)
    for epoch in range(epochs):
        if epoch == 0:
            model_user, mse_user, _ = fm.train(train_user_loader, val_user_loader, dic_user_review, dic_ui_review,
                                               dic_user_average_review,device)
            model_item, mse_item, _ = fm.train(train_item_loader, val_item_loader, dic_item_review, dic_ui_review,
                                               dic_item_average_review,device,is_user=False)

            torch.save(model_user, model_path + 'model_user_{}'.format(0))
            torch.save(model_item, model_path + 'model_item_{}'.format(0))

            print(
                "epoch:{}, mse_user:{:.5} ***  mse_item:{:.5}".format(epoch, mse_user, mse_item))
            mse_plt_list_user.append(mse_user)
            mse_plt_list_item.append(mse_item)

            test_mse_user, predicts_user, labels = val_co_iter(model_user, test_loader, device, min_val, max_val, dic_user_review,
                                              dic_ui_review,
                                              dic_user_average_review)
            test_mse_item, predicts_item, labels = val_co_iter(model_item, test_loader, device, min_val, max_val, dic_item_review,
                                              dic_ui_review,
                                              dic_item_average_review,
                                              is_user=False)
            print("user test mse is {:.5} *** item test mse is {:.5}".format(test_mse_user, test_mse_item))
            test_mse_plt_list_user.append(test_mse_user)
            test_mse_plt_list_item.append(test_mse_item)

            a, b, bias = 0.5,0.5,0
            # 2 getting testing data
            list_pre = join_two_res(np.array(predicts_user), np.array(predicts_item), a, b, bias)
            mse = mean_squared_error(np.array(labels), np.array(list_pre))
            print("linear regression mse is: {:.5}".format(mse))
            regression_mse_plt_list.append(mse)
            if best_regression_mse > mse:
                best_regression_epoch = epoch
                best_regression_mse = mse
            user_train_data_len.append(len(user_train_data))
            item_train_data_len.append(len(item_train_data))

        else:
            # generate a mini-set
            print(len(user_train_data), len(item_train_data))
            if tmp + mini_batch_size >= len(df_no_labels):
                break
            mini_batch = pd.DataFrame(np.array(df_no_labels.iloc[tmp:tmp + mini_batch_size, :]))
            tmp += mini_batch_size
            # generate pseudo-label
            mini_batch_loader = DataLoader(
                FmDatasetNoLabels(np.array(mini_batch[0]), np.array(mini_batch[1])), batch_size=64)
            user_predict = predict(model_user, mini_batch_loader, device, min_val, max_val, dic_user_review,
                                   dic_user_average_review)
            item_predict = predict(model_item, mini_batch_loader, device, min_val, max_val, dic_item_review,
                                   dic_item_average_review, is_user=False)

            # select reliable instances
            sub_predict = np.abs(np.array(user_predict) - np.array(item_predict))
            Rt = min(epoch/epoch_k*tao,tao)
            update_index = np.argsort(sub_predict[:,0])[:int(mini_batch_size*Rt)]

            # updata labelled set
            user_pseu = pd.concat([mini_batch, pd.DataFrame(user_predict)], axis=1,ignore_index=True)
            item_pseu = pd.concat([mini_batch, pd.DataFrame(item_predict)], axis=1, ignore_index=True)
            user_train_data = pd.concat([user_train_data, item_pseu.iloc[update_index,:]], axis=0)
            item_train_data = pd.concat([item_train_data, user_pseu.iloc[update_index,:]], axis=0)

            train_user_loader = DataLoader(
                FmDataset(np.array(user_train_data[0]), np.array(user_train_data[1]), np.array(user_train_data[2],dtype='float32')),
                batch_size=batch_size, shuffle=True)
            train_item_loader = DataLoader(
                FmDataset(np.array(item_train_data[0]), np.array(item_train_data[1]), np.array(item_train_data[2],dtype='float32')),
                batch_size=batch_size, shuffle=True)

            model_user, mse_user, _ = fm.train(train_user_loader, val_user_loader, dic_user_review, dic_ui_review,
                                               dic_user_average_review,device)
            model_item, mse_item, _ = fm.train(train_item_loader, val_item_loader, dic_item_review, dic_ui_review,
                                               dic_item_average_review,device,is_user=False)
            torch.save(model_user, model_path + 'model_user_{}'.format(epoch))
            torch.save(model_item, model_path + 'model_item_{}'.format(epoch))
            print(
                "epoch:{}, mse_user:{:.5} ***  mse_item:{:.5}".format(epoch, mse_user, mse_item))
            mse_plt_list_user.append(mse_user)
            mse_plt_list_item.append(mse_item)

            test_mse_user, predicts_user, labels = val_co_iter(model_user, test_loader, device, min_val, max_val, dic_user_review,
                                              dic_ui_review,
                                              dic_user_average_review)
            test_mse_item, predicts_item, labels = val_co_iter(model_item, test_loader, device, min_val, max_val, dic_item_review,
                                              dic_ui_review,
                                              dic_item_average_review,
                                              is_user=False)
            print("user test mse is {:.5} *** item test mse is {:.5}".format(test_mse_user, test_mse_item))
            test_mse_plt_list_user.append(test_mse_user)
            test_mse_plt_list_item.append(test_mse_item)
            a, b, bias = 0.5, 0.5, 0
            list_pre = join_two_res(np.array(predicts_user), np.array(predicts_item), a, b, bias)
            mse = mean_squared_error(np.array(labels), np.array(list_pre))
            print("linear regression mse is{:.5}".format(mse))
            regression_mse_plt_list.append(mse)
            if best_regression_mse > mse:
                best_regression_epoch = epoch
                best_regression_mse = mse
            user_train_data_len.append(len(user_train_data))
            item_train_data_len.append(len(item_train_data))
        print('best regression epoch:', best_regression_epoch, '  mse:', best_regression_mse)

    print('test mse: ', best_regression_mse)
    print("=====================================")
    return{'loss':best_regression_mse,'status':STATUS_OK}


# main
start_time = time.time()

# for para in [100,500,1000,1500,2000]:
#     cot2(para)
#     print("Up  para:", para, "===============================")


# fspace={
#     'learning_rate':hp.choice('learning_rate', [0.01]),
#     # 'learning_rate':hp.choice('learning_rate', [0.001,0.01]),
#     # 'weight_decay':hp.choice('weight_decay', [0.001,0.005,0.01,0.05,0.1]),
#     'weight_decay':hp.choice('weight_decay', [0.001]),
#     # 'tao':hp.choice('tao', [0.5, 0.6,0.7,0.8,0.9]),
#     'tao':hp.choice('tao', [0.8]),
#     # 'mini_batch_size':hp.choice('mini_batch_size', [200,400,600,800,1000]),
#     'mini_batch_size':hp.choice('mini_batch_size', [100,300,500,700,900,1100,1300,1500]),
# }
# trials = Trials()
# best = fmin(fn=cot2, space=fspace, algo=tpe.suggest, max_evals=15,trials=trials)
# print(best)
cot2(100)
end_time = time.time()
print("The program runs in {} minutes".format(str((end_time - start_time) // 60)))


