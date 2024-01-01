import os
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import wandb
wandb.login()

import torch
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from collections import defaultdict
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
import random
import numpy as np
import pickle as pkl
import copy
import math
import pprint

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
from torchfm.layer import FeaturesEmbedding, FeaturesLinear, AttentionalFactorizationMachine
from torchfm.layer import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron

from sentence_transformers import SentenceTransformer

from config import Config

bounds = [5, 15, 30, 50, 80, 130, 200, 300, 550]
config = Config(dataset_name='mlctr', criterion_name='bce', cap=2.0, diff_max=0.4, dic={'path':'./results/ml_exp_ui_no_mono'},\
                bounds = bounds, user_feat_cols = ['Occupation', 'age', 'gender'], feat_cols=[2, 3, 4],\
                item_feat_cols = None, n_query = 15, n_gate = 25, fixed_cols = [1, 5, 6, 7])
config.metrics.append('auc')

# import my own modules, and pass config_main as their config, to control the overall data flow
import model
from model import *

import trainer
from trainer import *

import data.process
from data.process import *

if config.dataset_name == 'lastfm':
  import data.lastfm
  from data.lastfm import *
  data.lastfm.config = config
elif config.dataset_name == 'mlctr':
  import data.mlctr
  from data.mlctr import *
  data.mlctr.config = config

model.config = config
trainer.config = config
data.process.config = config

for pth in [config.save_dir, config.save_dir_res, config.cold_path]:
  if not os.path.exists(pth):
    os.mkdir(pth)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if config.seed:
  seed_everything(config.seed)

def reset_score():
  config.mae_best, config.mse_best, config.ndcg_best, config.auc_best = 1e7, 1e7, 0, 0
  for k in config.ndcgs_best:
    config.ndcgs_best[k] = 0

def get_config(config_add={}):
    bounds = config.bounds
    bounds_i = [5, 15, 30, 60, 120, 200]

    config_wb = {
            'lr':1e-3,
            'wd':5e-5,
            'dropout':0,
            'lr_f':None,
            'lr_f_factor':1,
            'every':1,
            'wd_f':0,
            'model':'wd',
            'weight_fields':[True, False],
            'weight_fields_u':True,
            'nbins':kto_bin(2300, bounds) + 1,
            'nbins_i':kto_bin(2300, bounds_i) + 1,
            'seperate_decay':False,
            'log_train_bin':False,
            'warmup_epoch':None,
            'period_epoch':None,
            'log_freq':None,
            'dataset':config.dataset_name,
            'criterion':config.criterion_name,
            'use_item_embeds':False,
            'settings':list(config.settings),
            'seperate_weighting':False,
            'transplant':False,
            'gate_repeat':1,
            'gate_positive':True,
            'gate_mono':False,
            'gate_mono_feat':False,
            'use_count_bias':True,
            'use_user_item':True,
            'use_item_user':True,
            'gate_ui_limit':False,
    }

    config_wb.update(config_add)
    return config_wb

def run_main(config_wb=None):

  wandb.init(project="cold-ml-ctr", config=config_wb, name=config_wb.get('name'))
  config_t = wandb.config
  config.seperate_weighting = config_t.seperate_weighting
  reset_score()


  config.is_test = False
  try:
    config.gate_repeat = config_t.gate_repeat
    print(f"gate repeat: {config.gate_repeat}")
  except:
    config.gate_repeat = 1

  if 'gate_positive' in config_t.keys():
    if config_t['gate_positive']:
      config.settings.add('gate_positive')
    else:
      config.settings.discard('gate_positive')

  if 'gate_mono' in config_t.keys():
    if config_t['gate_mono']:
      config.settings.add('gate_mono')
      print('using mono gate...')
    else:
      config.settings.discard('gate_mono')

  for setting in ['gate_mono_feat', 'use_user_item', 'use_item_user', 'gate_ui_limit']:
    if setting in config_t.keys():
      if config_t[setting]:
        config.settings.add(setting)
        print(f'using {setting}...')
      else:
        config.settings.discard(setting)

  for setting in ['use_count_bias']:
    if setting in config_t.keys():
      if config_t[setting] and config_t.weight_fields_u:
        config.settings.add(setting)
        print(f'using {setting}...')
      else:
        config.settings.discard(setting)

  dropout = 0
  if 'dropout' in config_t.keys():
    dropout = config_t['dropout']

  config.transplant_gate = config_t.transplant
  config.use_item_embeds = config_t.use_item_embeds
  weight_fields = config_t.weight_fields if config_t.weight_fields is not None else [config_t.weight_fields_u, config_t.weight_fields_i]
  if config.dataset_name == 'lastfm':
    svdg = SmoothStart(list(config.ui_dims), list(config.user_dims), [config_t.nbins, config_t.nbins_i],\
                            item_field_dims = list(config.item_dims),\
                           embed_dim=config.mf_embed_dim, weight_fields=weight_fields,\
                           dataset=None, dataset_name=config_t.dataset).to(device)

  else: # default to movielens
    svdg = SmoothStart([6041, 3953], [21,  7,  2], [config_t.nbins, config_t.nbins_i],\
                           embed_dim=config.mf_embed_dim, weight_fields=weight_fields, mlp_dims=[64, 64],\
                           dataset=cold_dataset, dataset_name=config_t.dataset, dropout=dropout).to(device)

  if config_t.log_freq is not None:
    wandb.watch(svdg, log="all", log_freq=config_t.log_freq)

  base_params, field_params, field_params_i = [], [], []
  user_embed_params = []
  for name, param in svdg.named_parameters():
    if name in ['field_weights_u']:
      field_params.append(param)
    elif name in ['field_weights_i']:
      field_params_i.append(param)
    else:
      base_params.append(param)

    if 'user_embedding' in name:
        print("add to user_embed_params: ", name)
        user_embed_params.append(param)

  print(len(field_params), len(base_params), len(user_embed_params))

  if config_t.seperate_decay:
    lr_of = config_t.lr_f if config_t.lr_f is not None else config_t.lr_f_factor*config_t.lr
    optimizer_b, optimizer_f = configure_optimizers(svdg, config_t.wd, config_t.lr, lr_of, betas_f=[0.85, 0.995], wd_f=config_t.wd_f)
  else:
    optimizer_b = torch.optim.AdamW(params=base_params, lr=config_t.lr, weight_decay=config_t.wd)
    lr_of = config_t.lr_f if config_t.lr_f is not None else config_t.lr_f_factor*config_t.lr
    optimizer_f = torch.optim.AdamW(params=field_params, betas=[0.7, 0.99], lr=lr_of, weight_decay=config_t.wd_f)
    optimizer_fi = torch.optim.AdamW(params=field_params_i, betas=[0.7, 0.99], lr=lr_of, weight_decay=config_t.wd_f)

  svdg.model_use = config_t.model
  for name, param in svdg.named_parameters():
    print(name, param.requires_grad)
  print(svdg)

  print("field_weights_u:\n", svdg.field_weights_u, "\nfield_weights_i:\n", svdg.field_weights_i)

  config.every = config_t.every

  if config_t.warmup_epoch is not None:
    config.warmup_steps = len(train_data_loader_g) * config_t.warmup_epoch
    config.period = len(train_data_loader_g) * config_t.period_epoch
    scheduler_f = torch.optim.lr_scheduler.LambdaLR(optimizer_f, OneCyclePolicy())
  else:
    config.period = None
    scheduler_f = torch.optim.lr_scheduler.LambdaLR(optimizer_f, constant_lr)

  config.scheduler_f = scheduler_f
  config.optimizer_f = optimizer_f
  config.optimizer_fi = optimizer_fi

  n_epoch, stop_rounds = None, None
  if config_t.dataset == 'mlctr':
    n_epoch, stop_rounds = 40, 3
  else:
    n_epoch, stop_rounds = 20, 2

  model_mf1 = main('None',
            'None',
            'None',
            n_epoch,
            0.001,
            3000,
            1e-6,
            'cuda:0' if torch.cuda.is_available() else 'cpu', model=svdg, stop_rounds=stop_rounds, optimizer=optimizer_b, log_train_bin=False,
            train_data_loader=train_data_loader_g, valid_data_loader=valid_data_loader_g, dataset=train_support_dataset_g,
            criterion = config.criterion, score_func=mean_squared_error, higher_better=config.higher_better, print_bin=True, print_bin_train=False)

  if config.test_users:
      reset_score()

      config.is_test = True
      config.every = 1e7  # never train gate

      svdg.load_model(config.best_model_path)
      # initialize an optimizer with only user embeddings as trainable parameters
      optimizer_test = torch.optim.AdamW(params=user_embed_params, lr=config_t.lr, weight_decay=config_t.wd)

      scheduler_test = None
      if config.warmup_epoch_test is not None:
        print(f"warmup for {config.warmup_epoch_test} epochs")
        config.warmup_steps = len(test_data_loader) * config.warmup_epoch_test
        scheduler_test = torch.optim.lr_scheduler.LambdaLR(optimizer_f, lambda x: min(x / (config.warmup_steps), 1.0))

      print("\n\nentering test mode....\n\n\n")
      model_mf_test = main('None',
            'None',
            'None',
            10,
            0.001,
            3000,
            1e-6,
            'cuda:0' if torch.cuda.is_available() else 'cpu', model=svdg, stop_rounds=stop_rounds, optimizer=optimizer_test, log_train_bin=False,
            train_data_loader=test_data_loader, valid_data_loader=test_query_loader, dataset=train_support_dataset_g, scheduler=scheduler_test,
            criterion = config.criterion, score_func=mean_squared_error, higher_better=config.higher_better, print_bin=True, print_bin_train=False, test_last=True)


  wandb.finish()



# get data   (dataloader of particular dataset, and necessary data structures)
cold_dataset = None

if config.dataset_name == 'lastfm':
  cold_dataset = LastFMData(config)
elif config.dataset_name == 'mlctr':
  cold_dataset = MovieLensCTRDataset(config)
else:
  cold_dataset = MovieLens1MDataset(config)

train_data_loader_g, train_data_loader_w, valid_data_loader_g = cold_dataset.get_loaders()
config.train_data_loader_w = train_data_loader_w


try:
  train_support_dataset_g = cold_dataset.train_support_dataset_g    # passed to main in run_main
except:
  train_support_dataset_g = None
test_data_loader, test_query_loader = cold_dataset.get_test_loaders()


# main experiment with ui, wide&deep as base model
bounds = config.bounds
bounds_i = [5, 15, 30, 60, 120, 200]
config.dic['path'] = './results/ml_exp_ui_wd'

config_wb = {
        'name':'exp_ui_wd',
        'lr':1e-3,
        'wd':5e-5,
        'dropout':0,
        'lr_f':None,
        'lr_f_factor':1,
        'every':1,
        'wd_f':0,
        'model':'wd',
        'weight_fields':[True, False],
        'weight_fields_u':True,
        'nbins':kto_bin(2300, bounds) + 1,
        'nbins_i':kto_bin(2300, bounds_i) + 1,
        'seperate_decay':False,
        'log_train_bin':False,
        'warmup_epoch':None,
        'period_epoch':None,
        'log_freq':None,
        'dataset':config.dataset_name,
        'criterion':config.criterion_name,
        'use_item_embeds':False,
        'settings':list(config.settings),
        'seperate_weighting':False,
        'transplant':False,
        'gate_repeat':1,
        'gate_positive':True,
        'gate_mono':False,
        'gate_mono_feat':False,
        'use_count_bias':True,
        'use_user_item':True,
        'use_item_user':True,
        'gate_ui_limit':False,
}

run_main(config_wb)


bounds = config.bounds
bounds_i = [5, 15, 30, 60, 120, 200]
config.dic['path'] = './results/ml_control_wd'


config_wb = {
        'name':'control_wd',
        'lr':1e-3,
        'wd':5e-5,
        'dropout':0,
        'lr_f':None,
        'lr_f_factor':1,
        'every':1,
        'wd_f':0,
        'model':'wd',
        'weight_fields':[False, False],
        'weight_fields_u':False,
        'nbins':kto_bin(2300, bounds) + 1,
        'nbins_i':kto_bin(2300, bounds_i) + 1,
        'seperate_decay':False,
        'log_train_bin':False,
        'warmup_epoch':None,
        'period_epoch':None,
        'log_freq':None,
        'dataset':config.dataset_name,
        'criterion':config.criterion_name,
        'use_item_embeds':False,
        'settings':list(config.settings),
        'seperate_weighting':False,
        'transplant':False,
        'gate_repeat':1,
        'gate_positive':True,
        'gate_mono':False,
        'gate_mono_feat':False,
        'use_count_bias':True,
        'use_user_item':False,
        'use_item_user':False,
        'gate_ui_limit':False,
}

run_main(config_wb)


# main experiment with ui, dcn-v2 as base model
bounds = config.bounds
bounds_i = [5, 15, 30, 60, 120, 200]
config.dic['path'] = './results/ml_exp_dcn_v2'

config_wb = {
        'name':'exp_dcn_v2',
        'lr':1e-3,
        'wd':5e-5,
        'dropout':0,
        'lr_f':None,
        'lr_f_factor':1,
        'every':1,
        'wd_f':0,
        'model':'dcn_v2',
        'weight_fields':[True, False],
        'weight_fields_u':True,
        'nbins':kto_bin(2300, bounds) + 1,
        'nbins_i':kto_bin(2300, bounds_i) + 1,
        'seperate_decay':False,
        'log_train_bin':False,
        'warmup_epoch':None,
        'period_epoch':None,
        'log_freq':None,
        'dataset':config.dataset_name,
        'criterion':config.criterion_name,
        'use_item_embeds':False,
        'settings':list(config.settings),
        'seperate_weighting':False,
        'transplant':False,
        'gate_repeat':1,
        'gate_positive':True,
        'gate_mono':False,
        'gate_mono_feat':False,
        'use_count_bias':True,
        'use_user_item':True,
        'use_item_user':True,
        'gate_ui_limit':False,
}

run_main(config_wb)
