import numpy as np
import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tqdm
from collections import defaultdict


def get_user_items(train_support):
    train_support_dict = defaultdict(list) 
    for user, item in zip(train_support[config.uid_name], train_support[config.iid_name]):
        train_support_dict[user].append(item)
    return train_support_dict

def get_item_users(train_support):
    train_support_dict = defaultdict(list) 
    for user, item in zip(train_support[config.uid_name], train_support[config.iid_name]):
        train_support_dict[item].append(user)
    return train_support_dict

def load_dataframes(config):
    plays = pd.read_csv(config.lastfm_path + 'fm_plays.csv')
    users = pd.read_csv(config.lastfm_path + 'fm_users_processed.csv')
    return plays, users

def filter_interactions(plays, users, config, user_counts):
    users_use = set(user_counts[user_counts>=0].index)
    plays = plays[plays[config.uid_name].isin(users_use)]
    return plays, user_counts

def split_data_by_users(plays, users, config, user_counts = None, test_size=0.2):
    if user_counts is not None:
        users_use = set(user_counts[user_counts >= 0].index)
    else:
        users_use = set(users[config.uid_name])
    train_users, test_users = train_test_split(users[users[config.uid_name].isin(users_use)], test_size=test_size, random_state=42)
    train_df = plays[plays[config.uid_name].isin(train_users[config.uid_name])]
    test_df = plays[plays[config.uid_name].isin(test_users[config.uid_name])]
    return train_df, test_df


def process_users_dataframe(users, user_counts, config):
    users = users.merge(user_counts[user_counts >= 0], on=config.uid_name, how='left')
    users['ni'].fillna(0, inplace=True)
    users['ni_bin'] = users['ni'].apply(lambda x: kto_bin(x, config.bounds))
    return users

def create_data_loader(df, users, items, cold_dataset, shuffle=True):
    dataset = cold_dataset(df, users, items, config.item_feat_cols)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers)
    return data_loader

def set_config_attributes(config, plays, users, items=None, default_user_feat_cols=['genderi', 'agei', 'countryi', 'monthi']):
    try:
        user_feat_cols = config.user_feat_cols
    except AttributeError:
        user_feat_cols = default_user_feat_cols

    config.ui_dims = plays[[config.uid_name, config.iid_name]].max(axis=0) + 1
    config.user_dims = users[user_feat_cols].max(axis=0) + 1
    config.item_dims = items[config.item_feat_cols].max(axis=0) + 1 if (items is not None and config.item_feat_cols is not None) else []

def get_id_map(df_col, ignore_threshold=2):
    """
    return map that maps ids in df_col to 1, 2... df_col.nunique(),
        map infrequent ids to 0 (null)
    params:
        df_col(Series): column of a dataframe with discrete ids

    returns:
        id_map(Dict)
    """
    col_counts = df_col.value_counts()
    col_sort = col_counts[col_counts > ignore_threshold].index
    id_map = {old_id:i+1 for i, old_id in enumerate(col_sort)}

    to_fill = col_counts[col_counts <= ignore_threshold]
    id_map.update({x:0 for x in to_fill.index})
    return id_map   # map to 1 - n     0:NA

def map_ids(df, ids):
  for col in ids:
    mapping = get_id_map(df[col])
    df[col] = df[col].fillna(-100)
    mapping[-100] = 0
    df[col + 'i'] = df[col].apply(lambda x:mapping[x])


def kto_bin(x, bounds=[2, 5, 10, 25, 40, 60, 100]):
    try:
      bounds = config.bounds
    except:
      None
    for i, bound in enumerate(bounds):
      if x < bound:
        return i
    return len(bounds)




def calculate_percentiles(series):
    percentiles = np.arange(5, 105, 5)
    values = [np.nanpercentile(series, p) for p in percentiles]
    return pd.Series(values, index=percentiles)



def get_movie_counts(ratings):
  return ratings.groupby(config.uid_name).count()[config.iid_name]

def get_item_counts(trades):
  return trades.groupby(config.uid_name).count()[config.iid_name] 

def support_query_split(trades, all_ids, col_name, n_query=10, add_to_support=False, add_to_query=False):
  all_supports = []
  all_queries = []
  # all_ids = trades[col_name]   # I think this is correct, and less complex
  for id_t in tqdm.tqdm(all_ids):
    samples = trades[trades[col_name] == id_t]
    if len(samples) < n_query:
        if add_to_support:
            all_supports.append(samples)
        elif add_to_query:
            all_queries.append(samples)
        continue
    query = samples.sample(n_query)
    support = samples[~samples.index.isin(query.index)]
    all_queries.append(query)
    all_supports.append(support)
  return pd.concat(all_supports, axis=0), pd.concat(all_queries, axis=0)


def support_query_split_by_time(trades, all_ids, col_name, n_query=10, add_to_support=False, add_to_query=False, t_col='date'):
  all_supports = []
  all_queries = []
  # all_ids = trades[col_name]   
  for id_t in tqdm.tqdm(all_ids):
    samples = trades[trades[col_name] == id_t]
    if len(samples) < n_query:
        if add_to_support:
            all_supports.append(samples)
        elif add_to_query:
            all_queries.append(samples)
        continue
    samples_sort = samples.sort_values(t_col)
    query = samples_sort.iloc[-n_query:]
    support = samples_sort[:len(samples_sort) - n_query]
    all_queries.append(query)
    all_supports.append(support)
  return pd.concat(all_supports, axis=0), pd.concat(all_queries, axis=0)




def split_interactions(interactions, by='uidi', n_query=20, n_support_w=20, random_seed=42):
    interactions = interactions.sample(frac=1, random_state=random_seed)
    print('interactions.head():')
    print(interactions.head())

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


class ColdDatasetBase(torch.utils.data.Dataset):

    def __init__(self, ratings, users, items=None, item_feat_cols=None, mode='pretrain'):
        self.mode = mode 
        data = ratings.to_numpy()[:, :3]
        self.data = data[:, :2].astype(int)
        self.targets = data[:, 2].astype(float)
        self.field_dims = np.max(self.data, axis=0) + 1

        user_feat_cols = getattr(config, 'user_feat_cols', ['genderi', 'agei', 'countryi', 'monthi'])
        n_user_max = users[config.uid_name].max()
        self.n_items_bin = np.zeros(n_user_max + 1, dtype=int)
        self.user_features = np.full((n_user_max + 1, len(user_feat_cols)), 0, dtype=int)

        for line in users[[config.uid_name] + user_feat_cols + ['ni_bin']].to_numpy().astype(int):
            self.user_features[line[0]] = line[1:-1]
            self.n_items_bin[line[0]] = line[-1]

        if items is not None:
            n_item_max = items[config.iid_name].max()
            self.n_users_bin = np.zeros(n_item_max+1, dtype=int)
            for line in items[[config.iid_name, 'nu_bin']].to_numpy().astype(int):
                self.n_users_bin[line[0]] = line[-1]

            if item_feat_cols is not None:
                self.item_features = np.full((n_item_max + 1, len(item_feat_cols)), 0, dtype=int)
                for line in items[[config.iid_name] + item_feat_cols].to_numpy().astype(int):
                    self.item_features[line[0]] = line[1:]
            else:
                self.item_features = None
        else:
            self.n_users_bin = None
            self.item_features = None

        self.user_iids = defaultdict(list)
        for (uid, iid), label in zip(self.data, self.targets):
            self.user_iids[uid].append([iid, label])
        for key in self.user_iids.keys():
          self.user_iids[key] = np.array(self.user_iids[key])
        self.users = list(set(self.data[:, 0]))
        self.n_users = len(self.users)

    def shuffle_dictionary_lists(self, dataset):
        for key in dataset.keys():
            random.shuffle(dataset[key])

    def shuffle_user_iids(self):
        self.shuffle_dictionary_lists(self.user_iids)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        uid_iid = self.data[index]
        uid, iid = uid_iid
        item_features = [self.item_features[iid]] if self.item_features is not None else []
        users_bin  = [self.n_users_bin[iid]] if self.n_users_bin is not None else []
        return [uid_iid, self.user_features[uid]] + item_features +  [self.n_items_bin[uid]] + users_bin, self.targets[index]




import pickle
class BaseDataset():

    def __init__(self, config, attributes_to_load, cold_dataset_class=ColdDatasetBase):
        self.cold_dataset_class = cold_dataset_class
        for attr in attributes_to_load:
            with open(f'{attr}.pkl', 'rb') as f:
                setattr(self, attr, pickle.load(f))
        self.merge_data()
        self.create_loaders()
    
    def merge_data(self):
        pass
    
    def create_loaders(self):
        self.train_data_loader = create_data_loader(self.train_support, self.cold_dataset_class)
        self.train_query_data_loader = create_data_loader(self.train_query, self.cold_dataset_class, shuffle=False)
        self.test_data_loader = create_data_loader(self.test_support, self.cold_dataset_class)
        self.test_query_data_loader = create_data_loader(self.test_query, self.cold_dataset_class, shuffle=False)

    

    def create_data_loader(df, dataset_class, shuffle=True):
        dataset = dataset_class(df, self.users, self.items, config.item_feat_cols)
        data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)
        return data_loader
