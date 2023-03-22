import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from preference.utils import get_ui_review_dic, get_dic, get_average_dic, train_co_iter, val_co_iter, linearRegression, \
    join_two_res, predict
from preference.fm_model import FmDataset, FmDatasetNoLabels, FmReviewUI
from sklearn.model_selection import train_test_split
from tqdm import tqdm
device = torch.device('cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
path = '/data/lxkui/Amazon/instrument/'
lr = 5e-3
wd = 1e-4
bs = 1024
epochs = 100
class TrainFm():
    def __init__(self, u_max_id, i_max_id, lr, weight_decay, id_embedding_dim=256):
        self.max_val = 5.0
        self.min_val = 1.0
        self.u_max_id = u_max_id
        self.i_max_id = i_max_id
        self.id_embedding_dim = id_embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay

    def train(self, train_data_loader, val_data_loader, dic_review, dic_ui_review, dic_average_review,device,
              is_user=True):
        model = FmReviewUI(u_id_len=self.u_max_id, i_id_len=self.i_max_id, id_embedding_dim=self.id_embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_func = torch.nn.MSELoss().to(device)

        epoch = 0
        mse_best = 0
        # for i in tqdm(range(epochs)):
        for i in range(epochs):
            loss = train_co_iter(model, optimizer, train_data_loader, loss_func, device, self.min_val,
                                 self.max_val, dic_review, dic_ui_review, dic_average_review, is_user=is_user)
            mse, _, _ = val_co_iter(model, val_data_loader, device, self.min_val, self.max_val, dic_review,
                                    dic_ui_review, dic_average_review, is_user=is_user)
            # print(loss, mse)
            if i == 0:
                mse_best = mse
                torch.save(model, path + 'best_model/cot_best_model2')
            else:
                if mse_best > mse:
                    mse_best = mse
                    epoch = i
                    torch.save(model, path+'best_model/cot_best_model2')

        # print('best mse is {}, epoch is {}'.format(mse_best, epoch))
        model = torch.load(path+'best_model/cot_best_model2')
        return model, mse_best, epoch


def main():
    df = pd.read_csv(path + 'embedding/three.csv', header=None)
    u_max_id, i_max_id = max(df[0]) + 1, max(df[1]) + 1
    fm = TrainFm(u_max_id, i_max_id, lr, wd)
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=2020)
    train_u, train_i, train_rating = np.array(train_x[0]), np.array(train_x[1]), np.array(train_y).astype(np.float32)
    val_u, val_i, val_rating = np.array(val_x[0]), np.array(val_x[1]), np.array(val_y).astype(np.float32)
    train_loader = DataLoader(FmDataset(train_u, train_i, train_rating), batch_size=bs, shuffle=True)
    val_loader = DataLoader(FmDataset(val_u, val_i, val_rating), batch_size=bs)

    df_user_review = pd.read_csv(path+'embedding/user_embedding_vec.csv', header=None)
    dic_user_review = get_dic(df_user_review)
    dic_ui_review = get_ui_review_dic(df)
    dic_user_average_review = get_average_dic(df, 'user')

    model, mse, epoch = fm.train(train_loader, val_loader, dic_user_review, dic_ui_review, dic_user_average_review,device=device)
    print(mse, epoch)

if __name__ == '__main__':
    main()
