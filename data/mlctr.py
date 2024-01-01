import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .process import *
from sentence_transformers import SentenceTransformer


def get_bin_year(year):
  if year < 1989:
    return int((year - 1919) / 5)
  else:
    return int(year - 1975)

class MovieLensGatedDataset(torch.utils.data.Dataset):

    def __init__(self, ratings, users, items):
        data = ratings.to_numpy()[:, :3]
        self.data = data[:, :2].astype(int)
        self.targets = data[:, 2].astype(np.float32)
        self.field_dims = np.max(self.data, axis=0) + 1

        n_user_max, n_item_max = users[config.uid_name].max(), items[config.iid_name].max()
        self.n_movies_bin = np.zeros(n_user_max+1, dtype=int)

        self.user_features = np.full((n_user_max+1, 3), 0, dtype=int)
        for line in users[[config.uid_name, 'Occupation', 'age', 'gender', 'ni_bin']].to_numpy().astype(int):
          self.user_features[line[0]] = line[1:-1]
          self.n_movies_bin[line[0]] = line[-1]

        #% item side
        self.n_users_bin = np.zeros(n_item_max+1, dtype=int)

        for line in items[[config.iid_name, 'nu_bin']].to_numpy().astype(int):
          self.n_users_bin[line[0]] = line[-1]


    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        """
        returns:
          self.data[index]: [uid, iid]
        """
        uid_iid = self.data[index]
        uid, iid = uid_iid
        return [uid_iid, self.user_features[uid], self.n_movies_bin[uid], self.n_users_bin[iid]], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target



class ColdDataset(torch.utils.data.Dataset):

    def __init__(self, ratings, users, items=None, item_feat_cols=None):
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

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        uid_iid = self.data[index]
        uid, iid = uid_iid
        item_features = [self.item_features[iid]] if self.item_features is not None else []
        users_bin  = [self.n_users_bin[iid]] if self.n_users_bin is not None else []
        return [uid_iid, self.user_features[uid]] + item_features +  [self.n_items_bin[uid]] + users_bin, self.targets[index]



class MovieLensCTRDataset():

    def __init__(self, config):
        self.interactions = pd.read_csv(config.ml_path+"/ml_ratings_freq.csv")
        
        self.users = pd.read_csv(config.ml_path+"/users.dat", sep = "::",\
                            names = [config.uid_name, 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python')

        self.items = pd.read_csv(config.ml_path+"/movies.dat", sep = "::",\
                            names = [config.iid_name, 'Title', 'Genres'], encoding='latin-1', engine='python')

        self.process_movies()
        self.process_users()
        set_config_attributes(config, self.interactions, self.users)

        user_counts = self.interactions.groupby(config.uid_name).count()[config.iid_name].rename("ni") - (config.n_query + config.n_gate)
        train_df, test_df = split_data_by_users(self.interactions, self.users, config, user_counts)
        self.users = process_users_dataframe(self.users, user_counts, config)

        train_query, train_support_w, train_support = split_interactions(train_df[[config.uid_name, config.iid_name, 'label']], by=config.uid_name, n_query=config.n_query, n_support_w=config.n_gate)
        test_query, test_support_w, test_support = split_interactions(test_df[[config.uid_name, config.iid_name, 'label']], by=config.uid_name, n_query=config.n_query, n_support_w=config.n_gate)
        test_query = pd.concat([test_query, test_support_w])

        item_counts = train_support.groupby(config.iid_name).count()[config.uid_name]\
                                            .apply(lambda x:kto_bin(x, [5, 15, 30, 60, 120, 200])).rename("nu_bin")
        self.items = self.items.merge(item_counts, on=config.iid_name, how='left')
        self.items['nu_bin'].fillna(0, inplace=True)

        self.train_data_loader_g = create_data_loader(train_support, self.users, self.items, ColdDataset)
        self.valid_data_loader_g = create_data_loader(train_query, self.users, self.items, ColdDataset, shuffle=False)
        self.train_data_loader_w = create_data_loader(train_support_w, self.users, self.items, ColdDataset)
        self.test_support_loader = create_data_loader(test_support, self.users, self.items, ColdDataset)
        self.test_query_loader = create_data_loader(test_query, self.users, self.items, ColdDataset, shuffle=False)

        config.train_data_loader_w = self.train_data_loader_w
        self.counts_tr_train = get_item_counts(train_support)
        self.counts_tr_test = get_item_counts(test_support)
        config.counts_tr = pd.concat([self.counts_tr_train, self.counts_tr_test])

        self.train_support_dict = get_user_items(pd.concat([train_support, test_support]))
        self.item_user_dict = get_item_users(train_support)
        config.train_support_dict = self.train_support_dict
        config.item_user_dict = self.item_user_dict


    def get_loaders(self):
        return self.train_data_loader_g, self.train_data_loader_w, self.valid_data_loader_g

    def get_test_loaders(self):
        return self.test_support_loader, self.test_query_loader

    def process_movies(self):
        self.items['name'] = self.items.Title.apply(lambda x:x[:-7])
        self.items['year'] = self.items.Title.apply(lambda x:x[-5:-1])

        self.items['types'] = self.items.Genres.apply(lambda x:x.split('|'))
        movie_types = list(set([tp for lst in self.items.types for tp in lst]))
        self.items['tps'] = self.items['types'].apply(lambda x:[movie_types.index(t) for t in x])
        self.items['tps2'] = self.items.tps.apply(lambda x:x[:2]+[x[0] for _ in range(2 - len(x))])


        model_sent = SentenceTransformer('all-MiniLM-L6-v2')

        self.items['year_bin'] = self.items.year.astype(int).apply(get_bin_year)
        movie_name_embeds = model_sent.encode(self.items.name)


        # get movie feature tensors  [config.iid_name]
        nmov = self.items[config.iid_name].max() + 1
        movie_year_np = np.zeros((nmov, 1))
        movie_genres_np = np.zeros((nmov, 2))
        movie_name_np = np.zeros((nmov, 384))

        for mid, year, genre, name in zip(self.items[config.iid_name], self.items.year_bin, self.items.tps2, movie_name_embeds):
            movie_year_np[mid] = year
            movie_genres_np[mid] = genre
            movie_name_np[mid] = name

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.movie_year = torch.tensor(movie_year_np, dtype=torch.long).to(device)
        self.movie_genres = torch.tensor(movie_genres_np, dtype=torch.long).to(device)
        self.movie_name = torch.tensor(movie_name_np).to(device)

        self.movie_year.requires_grad_(False)
        self.movie_genres.requires_grad_(False)
        self.movie_name.requires_grad_(False)


    def process_users(self):
        labelencoder = LabelEncoder()
        self.users['age'] = labelencoder.fit_transform(self.users['Age'])
        self.users['gender'] = labelencoder.fit_transform(self.users['Gender'])









