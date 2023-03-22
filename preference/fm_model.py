import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np


class FactorizationMachine(nn.Module):

    def __init__(self, p, k, d=False):
        super(FactorizationMachine, self).__init__()

        self.p, self.k = p, k
        self.linear = nn.Linear(self.p, 1, bias=True)
        self.v = nn.Parameter(torch.zeros(self.p, self.k))
        self.drop = nn.Dropout(0.2)
        self.d = d

    def fm_layer(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        pair_interactions = torch.sum(torch.sub(torch.pow(inter_part1, 2),
                                                inter_part2), dim=1)
        self.drop(pair_interactions)
        # Binarization
        if self.d:
            linear_out = linear_part.detach()  
            linear_out = torch.sign(linear_out)
            # linear_out[linear_out < 0] = 0
            linear_out = linear_out > 0

            pair_interactions_out = pair_interactions.detach()
            pair_interactions_out = torch.sign(pair_interactions_out)
            # pair_interactions_out[pair_interactions_out < 0] = 0
            pair_interactions_out = pair_interactions_out > 0
        else:
            linear_out = linear_part
            pair_interactions_out = pair_interactions

        output = linear_out.transpose(1, 0) + 0.5 * pair_interactions_out

        return output

    def forward(self, x):
        output = self.fm_layer(x)
        return output.view(-1, 1)


class FM(nn.Module):

    def __init__(self, u_id_len, i_id_len, id_embedding_dim):
        super(FM, self).__init__()
        
        self.user_id_vec = nn.Embedding(u_id_len, id_embedding_dim)
        self.item_id_vec = nn.Embedding(i_id_len, id_embedding_dim)

        self.user_bias = nn.Embedding(u_id_len, 1)
        self.item_bias = nn.Embedding(i_id_len, 1)
        self.bias = nn.Parameter(torch.tensor([0.0]))

        self.fm = FactorizationMachine(id_embedding_dim * 2, 10)

        # self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.user_id_vec.weight, std=0.01)
        nn.init.normal_(self.item_id_vec.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)

    def forward(self, u_id, i_id):
        u_vec = self.user_id_vec(u_id)
        i_vec = self.item_id_vec(i_id)

        x = torch.cat((u_vec, i_vec), dim=1)
        rate = self.fm(x)
        return rate



class FmReviewUI(nn.Module):

    def __init__(self, u_id_len, i_id_len, id_embedding_dim,discrete=False):
        super(FmReviewUI, self).__init__()
        
        self.user_id_vec = nn.Embedding(u_id_len, id_embedding_dim)
        self.item_id_vec = nn.Embedding(i_id_len, id_embedding_dim)
        self.fm = FactorizationMachine(id_embedding_dim * 2 + 50 * 2, 10,discrete)

    def forward(self, u_id, i_id, review, ui_review):
        u_vec = self.user_id_vec(u_id)
        i_vec = self.item_id_vec(i_id)
        x = torch.cat((u_vec, i_vec, review, ui_review), dim=1)
        rate = self.fm(x)
        return rate


class FmDataset(Dataset):

    def __init__(self, uid, iid, rating):
        self.uid = uid
        self.iid = iid
        self.rating = rating

    def __getitem__(self, index):
        return self.uid[index], self.iid[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


class FmDatasetNoLabels(Dataset):
    def __init__(self, uid, iid):
        self.uid = uid
        self.iid = iid

    def __getitem__(self, index):
        return self.uid[index], self.iid[index]

    def __len__(self):
        return len(self.uid)



