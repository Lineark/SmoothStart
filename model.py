import torch
from torch import nn
from torch.nn import functional as F

from torchfm.layer import MultiLayerPerceptron
from torchfm.layer import FeaturesEmbedding
from torchfm.layer import CrossNetwork, MultiLayerPerceptron

import math
import numpy as np

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=int)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias
      
class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=int)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i



class SmoothStart(torch.nn.Module):
    def __init__(self, field_dims, user_field_dims, gate_dims, embed_dim,\
                     item_field_dims=[1943, 22315, 15195], mlp_dims=(64, 64),\
                       dropout=0, weight_fields=False,\
                       dataset = None, dataset_name = 'ml'):
        super().__init__()
        self.embed_dim = embed_dim
        user_dim, item_dim = field_dims
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.user_embedding = torch.nn.Embedding(user_dim, embed_dim)
        self.item_embedding = torch.nn.Embedding(item_dim, embed_dim)

        self.linear = FeaturesLinear(field_dims)


        torch.nn.init.xavier_uniform_(self.user_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight.data)

        try:
          self.use_item_embeds = config.use_item_embeds
        except:
          self.use_item_embeds = False

        self.n_fields = len(field_dims) + len(user_field_dims) + len(item_field_dims)
        if self.use_item_embeds:
          self.n_fields += 1

        if 'use_user_item' in config.settings:
          self.n_fields += 1
        if 'use_item_user' in config.settings:
          self.n_fields += 1

        self.embed_output_dim = self.n_fields * embed_dim
        self.feat_embed = FeaturesEmbedding(user_field_dims, embed_dim)
        self.feat_embed_item = FeaturesEmbedding(item_field_dims, embed_dim)

        self.year_embed = FeaturesEmbedding([26], embed_dim)
        self.genre_embed = FeaturesEmbedding([18, 18], embed_dim)
        self.name_fc = nn.Linear(384, embed_dim)

        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout) #mlp_dims=(16, 16), dropout=0.2)
        self.cn_v2 = CrossNetV2(self.embed_output_dim, 2)

        self.mlp_item = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                        nn.GELU(),
                                        nn.Linear(embed_dim, embed_dim)
                                        )
        self.cn = CrossNetwork(self.embed_output_dim, 2)
        self.cn_out = torch.nn.Linear(self.embed_output_dim, 1)

        self.model_use = 'svd'


        self.criterion = F.binary_cross_entropy_with_logits
        self.softmax = nn.Softmax(dim=1)

        self.epoch = 1

        self.user_item_embeds = torch.nn.Embedding(user_dim, embed_dim)
        self.user_item_embeds.weight.requires_grad_(False)#.requires_grad = False

        self.fc_out = nn.Linear(embed_dim, 1)


        self.weights = nn.Parameter(torch.zeros(1, 3))
        self.weightsK = nn.Parameter(torch.zeros(gate_dims[0], 3))


        self.weight_fields = weight_fields if isinstance(weight_fields, list) else [weight_fields, False]  # tuple of 2 booleans, (weight_u, weight_i)
        self.field_weights_u = nn.Parameter(torch.ones(gate_dims[0], self.n_fields))
        self.field_weights_i = nn.Parameter(torch.ones(gate_dims[1], self.n_fields))


        self.count_bias = nn.Parameter(torch.zeros(gate_dims[0]))


        self.dataset_name = dataset_name
        if dataset is not None:
          self.movie_year = dataset.movie_year      # (bs, 1)
          self.movie_genres = dataset.movie_genres  # (bs, 2)
          self.movie_name = dataset.movie_name   # (bs, 384)


        self.train_support_dict = config.train_support_dict
        try:
          self.item_user_dict = config.item_user_dict
        except:
          print("no item user embeds in config...")

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.user_item_embeds = torch.nn.Embedding(user_dim, embed_dim)
        self.user_item_embeds.weight.requires_grad_(False)

        self.item_user_embeds = torch.nn.Embedding(item_dim, embed_dim)
        self.item_user_embeds.weight.requires_grad_(False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def get_avg_item_embedding(self, uids):
        # This method should return the average item embedding for each user
        return self.user_item_embeds(uids)

    def get_user_avg_embd(self, u_id):
        item_ids = torch.tensor(self.train_support_dict[u_id]).to(self.device)  # (n_itmes, )
        embeds = self.item_embedding(item_ids)  # (n_itmes, embd_size)
        return torch.mean(embeds, axis=0)

    def get_user_item_embeds(self):
        for i in range(self.user_dim):
          if i in self.train_support_dict and len(self.train_support_dict[i]) > 0:
            embed = self.get_user_avg_embd(i)
            self.user_item_embeds.weight.data[i] = embed

    def get_avg_user_embedding(self, iids):
        # This method should return the average user embedding for each item
        return self.item_user_embeds(iids)

    def get_item_avg_embd(self, i_id):
        user_ids = torch.tensor(self.item_user_dict[i_id]).to(self.device)  # (n_users, )
        embeds = self.user_embedding(user_ids)  # (n_users, embd_size)
        return torch.mean(embeds, axis=0)

    def get_item_user_embeds(self):
        for i in range(self.item_dim):
            if i in self.item_user_dict and len(self.item_user_dict[i]) > 0:
                embed = self.get_item_avg_embd(i)
                self.item_user_embeds.weight.data[i] = embed

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad_(False)

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad_(True)

    def get_item_feats(self, iids):
        """
        params:
          iids (bs): item ids of this batch

        """
        year = self.movie_year[iids]       # (bs, 1)
        genres = self.movie_genres[iids]  # (bs, 2)
        name = self.movie_name[iids]    # (bs, 384)
        year_embed = self.year_embed(year)    # (bs, 1, embed_dim)
        genres_embed = self.genre_embed(genres).mean(axis=1).unsqueeze(1)  # (bs, 2, embed_dim) -> (bs, 1, embed_dim)
        name_embed = self.name_fc(name.float()).unsqueeze(1) # (bs, 384) -> (bs, 1, embed_dim)
        return torch.cat([year_embed, genres_embed, name_embed], dim=1)  # (bs, 3, embed_dim)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, inp, target=None):

        if self.dataset_name == 'lastfm':
          x, feats, gin = inp
          item_feats = None 
        else:   
          x, feats, gin, gin_item = inp
          x = x.long()
          iids = x[:, 1]
          item_feats = self.get_item_feats(iids)

        bs, n_field = x.shape  # x: (uid, iid)
        bs1, n_feats = feats.shape

        user_embed = self.user_embedding(x[:, [0]])

        embeds_ui = torch.cat([user_embed, self.item_embedding(x[:, [1]])], axis=1)  # (bs, 1) -> (bs, 1, n_embd) -> (bs, 2, n_embd)

        try:
          feat_embeds = self.feat_embed(feats)
          if item_feats is None:
            embed_x = torch.cat([embeds_ui, feat_embeds], dim=1)  # (bs, n_fields, embed_dim)
          else:
            embed_x = torch.cat([embeds_ui, feat_embeds, item_feats], dim=1)  # (bs, n_fields, embed_dim)
        except:
          if item_feats is None:
            embed_x = embeds_ui
          else:    
            embed_x = torch.cat([embeds_ui, item_feats], dim=1)

        if self.use_item_embeds:
          item_embeds = self.user_item_embeds(x[:, 0])  # of shape (bs, n_embd)
          user_embed_i = self.mlp_item(item_embeds).unsqueeze(1)  # (bs, n_embd) -> (bs, 1, n_embd)
          embed_x = torch.cat([embed_x, user_embed_i], dim=1)

        if 'use_user_item' in config.settings:
          avg_item_embed = self.get_avg_item_embedding(x[:, 0])
          embed_x = torch.cat([embed_x, avg_item_embed.unsqueeze(1)], dim=1)

        if 'use_item_user' in config.settings:
          avg_user_embed = self.get_avg_user_embedding(x[:, 1])
          embed_x = torch.cat([embed_x, avg_user_embed.unsqueeze(1)], dim=1)


        if self.weight_fields[0]:
          embed_x = embed_x * self.field_weights_u[gin.long()].unsqueeze(-1)    # （bs, n_fields, 1)
        if self.weight_fields[1]:
          # assert gin_item is not None
          embed_x = embed_x * self.field_weights_i[gin_item.long()].unsqueeze(-1)    # （bs, n_fields, 1)

        if self.model_use == 'wd':
          out = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim)) # out: (bs, 1) 
        elif self.model_use == 'dcn_v2':
          out = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim)) + self.cn_out(self.cn_v2(embed_x.view(-1, self.embed_output_dim))) 
        elif self.model_use == 'dcn':
          out = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))  + self.cn_out(self.cn(embed_x.view(-1, self.embed_output_dim)))
        else:
          print(f"error: model name illegal: {self.model_use}")
        
        if 'use_count_bias' in config.settings:
          out += self.count_bias[gin.long()].unsqueeze(-1)  # (bs, 1)

        loss = None
        if target is not None:
          loss = self.criterion(out.squeeze(1), target)

        return out.squeeze(1), loss # shape (bs,)