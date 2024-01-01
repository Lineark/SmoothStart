import pandas as pd
import numpy as np
import torch
import random
from torch.nn import functional as F


class Config:
  save_dir = './models'  
  save_dir_res = './results'    
  ml_path = './ml-1m'
  cold_path = './cold/'
  lastfm_path = './Lastfm/'

  transplant_gate = False

  params = {}
  dic = {}
  settings = {'gate_positive'}
  dataset_name = 'lastfm'  
  split_order = ['user'] 
  add_to_support = False
  seed = 42
  num = 0
  info = ''
  mf_embed_dim = 32

  seperate_weighting = False

  n_query = 30   
  n_gate = 40   

  batch_size = 3000
  num_workers = 2

  grad_norm_clip = 1.8 
  grad_norm_clip_gate = 1.8

  every = 5
  warmup_steps = None
  period = None

  uid_name = 'uidi'
  iid_name = 'iidi'

  dataset = None
  kto_bin = None  
  mae_best = 1e5
  mse_best = 1e7
  ndcg_best = 0
  auc_best = 0
  mae_t, ndcg_t, mse_t = None, None, None
  scheduler_f, optimizer_f, optimizer_fi = None, None, None
  messager = {}

  metrics = ['mse', 'mae', 'ndcg']

  gate = torch.tensor([[0.0664, 0.3071, 0.7955, 1.5197, 1.6781, 0.6791, 0.8044, 0.7104],
        [0.1341, 0.2937, 1.1471, 1.6705, 1.5629, 0.6357, 0.8705, 0.6605],
        [0.2735, 0.3194, 0.6400, 1.6848, 1.6015, 0.4829, 0.8120, 0.7300],
        [0.2640, 0.2797, 0.8053, 1.6682, 1.5930, 0.6754, 0.8637, 0.7430],
        [0.3301, 0.3503, 0.9499, 1.6549, 1.5973, 0.6721, 1.1296, 0.6178],
        [0.3636, 0.3451, 0.7032, 1.6939, 1.5710, 0.8200, 1.1806, 0.5327],
        [0.5922, 0.4219, 0.8397, 1.4939, 1.4872, 1.0452, 1.2247, 0.6959]],
       device='cuda:0' if torch.cuda.is_available() else 'cpu', requires_grad=False)
  fixed_cols = [1]
  gate_repeat = 1
  gate_feature = 'bin'

  higher_better = True

  criterion_name = 'bce'
  train_support_dict = None
  use_item_embeds = True


  test_users = True
  is_test = False
  n_query_test = n_query + n_gate

  random_int = random.randint(0, 1000)
  best_model_path = cold_path + f'best_model_{random_int}.pt'


  warmup_epoch_test = 5
  warmup_steps_test = None

  ndcg_at = [3, 5, 7, 10]
  ndcgs_best = {f'ndcg@{i}':0 for i in range(1, 20)}


  bounds = [2, 5, 10, 20, 35, 60, 90, 150]  
  feat_cols = []
  user_feat_cols = None
  item_feat_cols = None


  def __init__(self, dataset_name='lastfm', **kwargs):
    self.dataset_name = dataset_name
    self.init_dataset_config(dataset_name)
    for k, v in kwargs.items():
      setattr(self, k, v)

    self.criterion = F.binary_cross_entropy_with_logits if self.criterion_name == 'bce' else torch.nn.MSELoss()
    

  def init_dataset_config(self, dataset_name):
    if dataset_name == 'lastfm':
        self.bounds = [2, 4, 7, 10, 15, 20, 25, 35, 40, 50, 60, 70, 80]
        self.user_feat_cols = ['genderi', 'agei', 'countryi', 'monthi']
        self.item_feat_cols = None
        self.target_name = 'label'
        self.metrics.append('auc')

        self.criterion_name = 'bce'

        self.ui_dims = [267383, 145973]
        self.user_dims = [3, 110, 232, 86]
        self.item_dims = []
        self.feat_cols = [2, 3, 4, 5]
    elif dataset_name == 'mlctr':
        self.criterion_name='bce'
        self.cap=2.0
        self.diff_max=0.4
        self.bounds = [5, 15, 30, 50, 80, 130, 200, 300, 550]
        self.user_feat_cols = ['Occupation', 'age', 'gender']
        self.feat_cols=[2, 3, 4]
        self.item_feat_cols = None
        self.n_query = 15
        self.n_gate = 25
        self.fixed_cols = [1, 5, 6, 7]


