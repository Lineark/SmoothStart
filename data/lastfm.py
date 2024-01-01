
import pandas as pd
import json
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


# from .process import get_item_counts, support_query_split, support_query_split_by_time, kto_bin
from .process import *


def split_interactions(interactions, by='uidi', n_query=20, n_support_w=20, random_seed=42):
    interactions = interactions.sample(frac=1, random_state=random_seed)

    grouped = interactions.groupby(by)

    train_query_list = []
    train_support_w_list = []
    train_support_list = []

    for name, group in grouped:
        if len(group) < n_query + n_support_w:
            continue  
        else:
            train_query = group.iloc[:n_query]
            train_query_list.append(train_query)

            group = group.iloc[n_query:]

            train_support_w = group.iloc[:n_support_w]
            train_support_w_list.append(train_support_w)

            train_support = group.iloc[n_support_w:]
            train_support_list.append(train_support)

    train_query = pd.concat(train_query_list)
    train_support_w = pd.concat(train_support_w_list)
    train_support = pd.concat(train_support_list)

    return train_query, train_support_w, train_support



class LastFMDataset(torch.utils.data.Dataset):

    def __init__(self, ratings, users):
        data = ratings.to_numpy()[:, :3]
        self.data = data[:, :2].astype(int)
        self.targets = data[:, 2].astype(float)
        self.field_dims = np.max(self.data, axis=0) + 1

        try:
            user_feat_cols = config.user_feat_cols
        except:
            user_feat_cols = ['genderi', 'agei', 'countryi', 'monthi']

        n_user_max = users[config.uid_name].max()
        self.n_items_bin = np.zeros(n_user_max+1, dtype=int)
        self.user_features = np.full((n_user_max+1, len(user_feat_cols)), 0, dtype=int)

        for line in users[[config.uid_name]+user_feat_cols+['ni_bin']].to_numpy().astype(int):
          self.user_features[line[0]] = line[1:-1]
          self.n_items_bin[line[0]] = line[-1]

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        uid_iid = self.data[index]
        uid, iid = uid_iid
        return [uid_iid, self.user_features[uid], self.n_items_bin[uid]], self.targets[index]


class LastFMData():
    """
    """

    def __init__(self, config):

        self.plays = pd.read_csv(config.lastfm_path + 'fm_plays.csv')
        self.users = pd.read_csv(config.lastfm_path + 'fm_users_processed.csv')

        try:
            user_feat_cols = config.user_feat_cols
        except:
            user_feat_cols = ['genderi', 'agei', 'countryi', 'monthi']

        config.ui_dims = self.plays[[config.uid_name, config.iid_name]].max(axis=0) + 1
        config.user_dims = self.users[user_feat_cols].max(axis=0) + 1
        config.item_dims = []

        user_counts = self.plays.groupby(config.uid_name).count()[config.iid_name].rename("ni") - (config.n_query + config.n_gate)
        users_use = set(user_counts[user_counts>=0].index)
        self.plays = self.plays[self.plays[config.uid_name].isin(users_use)]

        self.users = self.users.merge(user_counts[user_counts>=0], on=config.uid_name, how='left')
        self.users['ni'].fillna(0, inplace=True)
        self.users['ni_bin'] = self.users['ni'].apply(lambda x:kto_bin(x, config.bounds))


        # dataset split
        train_users, test_users = train_test_split(self.users[self.users[config.uid_name].isin(users_use)], test_size=0.2, random_state=42)
        train_df = self.plays[self.plays.uidi.isin(train_users.uidi)]
        test_df = self.plays[self.plays.uidi.isin(test_users.uidi)]

        print("num null in ni_bin users:", self.users['ni_bin'].isnull().sum())
        print("number of test users:", len(test_users))

        train_query, train_support_w, train_support = split_interactions(train_df[[config.uid_name, config.iid_name, 'label']], by=config.uid_name, n_query=config.n_query, n_support_w=config.n_gate)
        test_query, test_support_w, test_support = split_interactions(test_df[[config.uid_name, config.iid_name, 'label']], by=config.uid_name, n_query=config.n_query, n_support_w=config.n_gate)
        test_query = pd.concat([test_query, test_support_w])

        self.counts_tr_train = get_item_counts(train_support)
        self.counts_tr_test = get_item_counts(test_support)
        self.counts_tr = pd.concat([self.counts_tr_train, self.counts_tr_test])
        config.counts_tr = self.counts_tr 

        cold_dataset = LastFMDataset
        train_support_dataset_g = cold_dataset(train_support, self.users)
        train_query_dataset_g = cold_dataset(train_query, self.users)
        train_support_dataset_w = cold_dataset(train_support.iloc[:10] if config.n_gate==0 else train_support_w, self.users)
        self.train_data_loader_g = DataLoader(train_support_dataset_g, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        self.valid_data_loader_g = DataLoader(train_query_dataset_g, batch_size=config.batch_size, num_workers=config.num_workers)
        self.train_data_loader_w = DataLoader(train_support_dataset_w, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        config.train_data_loader_w = self.train_data_loader_w

        test_support_dataset = cold_dataset(test_support, self.users)
        test_query_dataset = cold_dataset(test_query, self.users)
        self.test_support_loader = DataLoader(test_support_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        self.test_query_loader = DataLoader(test_query_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        self.train_support_dict = get_user_items(pd.concat([train_support, test_support]))
        self.item_user_dict = get_item_users(train_support)
        config.train_support_dict = self.train_support_dict
        config.item_user_dict = self.item_user_dict

    def get_loaders(self):
        return self.train_data_loader_g, self.train_data_loader_w, self.valid_data_loader_g

    def get_test_loaders(self):
        return self.test_support_loader, self.test_query_loader

