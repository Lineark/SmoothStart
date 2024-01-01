import pandas as pd
import numpy as np 
from collections import defaultdict

import torch 
import math 
from torch.optim.lr_scheduler import LambdaLR

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import warnings
from typing import Union, Iterable
import wandb
from data.process import kto_bin

from torch.utils.data import DataLoader
import tqdm
import pickle



def get_dcg_one(scores, k=10):
  pair = np.array(scores)
  topk=pair[np.argsort(pair[:,1])[::-1]][:k][:, 0]
  denominator = np.log2([i+2 for i in range(len(topk))])
  return np.sum(topk / denominator)

def get_idcg_one(scores, k=10):
  targets = np.array(scores)[:, 0]
  topk = np.sort(targets)[::-1][:k]
  denominator = np.log2([i+2 for i in range(len(topk))])
  return np.sum(topk / denominator)


def get_dcg(u_dict, func=get_idcg_one, k=10):
  dcgs = {}
  for uid, pairs in u_dict.items():
    dcgs[uid] = func(pairs, k=k)
  return dcgs

def get_ndcg(res, k=10):

    uids, targets, predicts = res

    user_dict = defaultdict(list)

    for uid, target, pred in zip(uids, targets, predicts):
      user_dict[uid].append((target, pred))  # list of tuple

    idcgs = get_dcg(user_dict, func=get_idcg_one, k=k)
    dcgs = get_dcg(user_dict, func=get_dcg_one, k=k)
    ndcgs = {uid:dcgs[uid] / idcg for uid, idcg in idcgs.items()}
    return ndcgs

def configure_optimizers(model, weight_decay, learning_rate, lr_of, betas_f=[0.85, 0.995], wd_f=0):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    param_f = set()

    for pn, p in model.named_parameters():
        if pn == 'field_weights':
          param_f.add(pn)
        elif pn.endswith('bias') or pn in ['linear.fc.weight']:
          no_decay.add(pn)
        else:
          decay.add(pn)

    print("decay:\n", decay, "\nno_decay:\n", no_decay, "\nparam_f:\n", param_f)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay | param_f
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    optimizer_f = torch.optim.AdamW([param_dict[pn] for pn in sorted(list(param_f))], lr=lr_of, betas=betas_f, weight_decay=wd_f)

    return optimizer, optimizer_f





class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            # torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class OneCyclePolicy(object):
    def __init__(self):
        pass
    def __call__(self, t):
      step = t % config.period
      if t < config.warmup_steps:
        return t / config.warmup_steps
      elif t < config.period:
        factor = 0.05 + 0.95 * .5 * (1 + math.cos(math.pi * (step - config.warmup_steps) / (config.period - config.warmup_steps)))
        return factor
      return 0.05

class CyclicPolicy(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, t):
      config = self.config
      step = t % config.period
      if t < config.warmup_steps:
        return step / config.warmup_steps
      if step < config.warmup_steps:
        factor = 0.1 + 0.9 * step / config.warmup_steps
      else:
        factor = 0.1 + 0.9 * .5 * (1 + math.cos(math.pi * (step - config.warmup_steps) / (config.period - config.warmup_steps)))
      return factor

class WarmupPolicy(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, t):
      config = self.config
      if t < config.warmup_steps:
        return t / config.warmup_steps
      return 1

def constant_lr(t):
  return 1


def get_eval_df(res, counts, bin_func=kto_bin, eval_ndcg=True):
    uids, targets, predicts = res
    uids = np.array(uids)

    eval_df = pd.DataFrame({config.uid_name:np.array(uids), 'target':targets, 'predict':predicts})
    eval_df = eval_df.merge(counts, 'left', config.uid_name)
    eval_df = eval_df.rename({counts.name:'nMovies'}, axis=1)

    eval_df['nMovies'].fillna(0, inplace=True)
    eval_df['bin'] = eval_df['nMovies'].apply(bin_func)

    if 'auc' in config.metrics:
      auc = roc_auc_score(eval_df.target, eval_df.predict)
      eval_df['auc'] = auc
      wandb.log({'auc':auc})

    if eval_ndcg:
      ndcg_dict = get_ndcg(res)
      ndcg_arr = np.array([(uid, ndcg) for uid, ndcg in ndcg_dict.items()])
      ndcg_df = pd.DataFrame.from_dict({config.uid_name:ndcg_arr[:, 0], 'ndcg':ndcg_arr[:, 1]})
      eval_df = eval_df.merge(ndcg_df, on=config.uid_name, how='left')
      for i in config.ndcg_at:
        ndcg_dict_t = get_ndcg(res, k=i)
        score = np.mean(list(ndcg_dict_t.values()))
        name = f'ndcg@{i}'
        print(f'ndcg@{i}: {score}')
        config.ndcgs_best[name] = max(config.ndcgs_best[name], score)
        wandb.log({f'ndcg@{i}':score, f'ndcg@{i}_best':config.ndcgs_best[name]})
    return eval_df



def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

def sigmoid_np(x):
  return list(1 / (1 + np.exp(-np.array(x))))

class TrainerConfig:
  return_uids = False

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
cft = TrainerConfig(return_uids=True)


all_grads = []
def train(model, optimizer, data_loader, criterion, device, log_interval=10, epoch=None, scheduler=None):
    scheduler_f = config.scheduler_f
    optimizer_f = config.optimizer_f
    optimizer_fi = config.optimizer_fi


    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)

    train_data_loader_w = config.train_data_loader_w
    dataloader_iterator = iter(train_data_loader_w)
    for i, (fields, target) in enumerate(tk0):

        target = target.to(device)
        if isinstance(fields, (list, tuple)):
          fields = (x.to(device) for x in fields)
        else:
          fields = fields.to(device)


        y, loss = model(fields) 
        if loss is None:
          loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        grad_norm = None
        if config.grad_norm_clip is not None:
          grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
        all_grads.append(grad_norm)

        if scheduler is not None:
          scheduler.step()
          lr_t = scheduler.get_lr()[0]
          try:
            scheduler_f.step()
          except:
            None

          wandb.log({"epoch":epoch, "lr_factor":lr_t})
        optimizer.step()
        total_loss += loss.item()

        wandb.log({"loss": loss, "epoch":epoch, "grad_norm_l2":grad_norm})

        if config.transplant_gate and config.params['update_gate']:
          model.field_weights_u.data = model.field_weights_u.data + config.gate_diff_per_step

        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

        if not config.transplant_gate and (i + 1) % config.every == config.every - 1 and not config.is_test:
            try:
              _ = config.gate_repeat 
            except:
              config.gate_repeat = 1
            for _ in range(config.gate_repeat):
                try:
                    fields, target = next(dataloader_iterator)
                except:
                    dataloader_iterator = iter(train_data_loader_w)
                    fields, target = next(dataloader_iterator)

                target = target.to(device)
                if isinstance(fields, (list, tuple)):
                    fields = (x.to(device) for x in fields)
                else:
                    fields = fields.to(device)


                y, loss = model(fields, target.float())
                if loss is None:
                    loss = criterion(y, target.float())
                model.zero_grad()
                loss.backward()

                grad_norm = None
                if config.grad_norm_clip_gate is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip_gate)

                optimizer_f.step()
                optimizer_fi.step()
                wandb.log({"loss_f": loss, "epoch":epoch,  "grad_norm_f_l2":grad_norm})

                if config.seperate_weighting:
                # Reset certain columns of field_weights_u to 1
                    with torch.no_grad():
                        model.field_weights_u[:, config.fixed_cols] = 1

                if 'gate_positive' in config.settings:
                  with torch.no_grad():
                      model.field_weights_u[model.field_weights_u < 0] = 0

                if 'gate_mono' in config.settings:
                  with torch.no_grad():
                      try:
                        cap = config.cap
                      except:
                        cap = 2.0
                        print("no max gate value specified in config, using default 2.0")
                        config.cap = cap
                      model.field_weights_u[model.field_weights_u > cap] = cap

                      try:
                        diff_max = config.diff_max
                      except:
                        diff_max = 0.3
                        print("no max gate diff value specified in config, using default 0.3")
                        config.diff_max = diff_max

                      for i in range(model.field_weights_u.shape[0]-1, 0, -1):
                          if model.field_weights_u[i, 0] < 0.06*i:
                              model.field_weights_u[i, 0] = 0.06*i
                          if model.field_weights_u[i, 0] < model.field_weights_u[i-1, 0]:
                              mean = (model.field_weights_u[i-1, 0] + model.field_weights_u[i, 0]) / 2
                              model.field_weights_u[i-1, 0] = mean
                              model.field_weights_u[i, 0] = mean
                          elif model.field_weights_u[i, 0] > model.field_weights_u[i-1, 0]+0.3:
                              mean = (model.field_weights_u[i-1, 0] + model.field_weights_u[i, 0]) / 2
                              model.field_weights_u[i-1, 0] = mean - 0.15
                              model.field_weights_u[i, 0] = mean + 0.15

                if 'gate_mono_feat' in config.settings:
                  with torch.no_grad():
                      try:
                        f_cols = config.feat_cols 
                        if len(f_cols) == 0:
                          f_cols = list(range(2, model.field_weights_u.shape[1]))
                      except:
                        f_cols = list(range(2, model.field_weights_u.shape[1]))
                      for col in f_cols:
                          for i in range(model.field_weights_u.shape[0]-1, 0, -1):
                              if model.field_weights_u[i, col] > model.field_weights_u[i-1, col]:
                                  mean = (model.field_weights_u[i-1, col] + model.field_weights_u[i, col]) / 2
                                  model.field_weights_u[i-1, col] = mean
                                  model.field_weights_u[i, col] = mean
                              elif model.field_weights_u[i, col] < model.field_weights_u[i-1, col]-0.2:
                                  mean = (model.field_weights_u[i-1, col] + model.field_weights_u[i, col]) / 2
                                  model.field_weights_u[i-1, col] = mean + 0.1
                                  model.field_weights_u[i, col] = mean - 0.1
                                
                if 'gate_ui_limit' in config.settings:
                  with torch.no_grad():
                      model.field_weights_u[:, 6:][model.field_weights_u[:, 6:] > 0.8] = 0.8
                      

def test(model, data_loader, device, score_func=roc_auc_score, config_t=None, epoch=None, mode='valid'):
    if config_t is None:
      config_t = TrainerConfig()

    model.eval()
    targets, predicts = list(), list()
    uids = list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            target = target.to(device)
            if isinstance(fields, (list, tuple)):
              uids.extend(list(fields[0][:, 0].detach().numpy()))
              fields = (x.to(device) for x in fields)
            else:
              fields = fields.to(device)
            out = model(fields)
            y = out[0] if isinstance(out, (tuple, list)) else out
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
        if config.criterion_name == 'bce':
          predicts = sigmoid_np(predicts)
        if mode == 'valid':
          if 'auc' in config.metrics:
            config.auc_t = roc_auc_score(targets, predicts)
            config.auc_best = max(config.auc_best, config.auc_t)
            wandb.log({"val_auc":config.auc_t, "val_auc_best":config.auc_best})

    if config_t.return_uids:
      return uids, targets, predicts   # all has shape (n_samples,)
    return score_func(targets, predicts)




def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         train_data_loader=None, valid_data_loader=None, test_data_loader=None,
         dataset=None, stop_rounds=2, log_train_bin = False,
         criterion = torch.nn.BCELoss(), score_func=roc_auc_score, rev_metrics=False, higher_better = None,
         model = None, optimizer=None, optimizer_f=None, scheduler=None,
         add_epoch=False, print_bin=True, print_bin_train=False,
         save_dir=None, additional_info=None, test_last=False):
    save_dir = config.save_dir if save_dir is None else save_dir
    if 'auc' in config.metrics:
      higher_better = True
    rev_metrics = not higher_better if higher_better is not None else rev_metrics
    higher_better = not rev_metrics if (higher_better is None and rev_metrics is not None) else higher_better

    device = torch.device(device)
    if train_data_loader is None:
      if dataset is None:
        dataset = get_dataset(dataset_name, dataset_path)
      train_length = int(len(dataset) * 0.8)
      valid_length = int(len(dataset) * 0.1)
      test_length = len(dataset) - train_length - valid_length
      train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
          dataset, (train_length, valid_length, test_length))
      train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
      valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=2)
      test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    if model is None:
      model = get_model(model_name, dataset).to(device)   # not implemented yet, model cannot be none now

    sample_test, _ = next(iter(valid_data_loader))
    if isinstance(sample_test, (list, tuple)):
      sample_test = tuple([[x.to(device) for x in sample_test]])

    if optimizer is None:
      optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    if scheduler is None:
      if config.period is not None:
          scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, OneCyclePolicy())
      else:
          scheduler = None 

    early_stopper = EarlyStopper(num_trials=stop_rounds, save_path=f'{save_dir}/{model_name}_{dataset_name}_{epoch}_{config.seed}_{config.num}.pt')


    mse_tr, mae_tr, ndcg_tr, mse_val, mae_val = None, None, None, None, None
    mse_tr_bin = None

    best_score = 0 if higher_better else 1e7

    if config.transplant_gate:
      model.field_weights_u.requires_grad_(False)

    counts_tr = config.counts_tr

    if config.transplant_gate:
      gate_diff = config.gate.data - model.field_weights_u.data

      n_steps = len(train_data_loader) * 3
      config.gate_diff_per_step = gate_diff / n_steps
      config.params['update_gate'] = True

    for epoch_i in range(epoch):
        if epoch_i >= 3:
          config.params['update_gate'] = False

        if 'use_user_item' in config.settings and not config.is_test:
          model.get_user_item_embeds()
        if 'use_item_user' in config.settings and not config.is_test:
          model.get_item_user_embeds()
        train(model, optimizer, train_data_loader, criterion, device, epoch=epoch_i, scheduler=scheduler)
        
        if model.use_item_embeds:
          model.get_user_item_embeds()
        if log_train_bin:
          resf = test(model, train_data_loader, device, score_func=mean_squared_error, config_t=cft, epoch=epoch_i, mode='train')
          eval_df_f = get_eval_df(resf, counts_tr)
          ndcg_tr = eval_df_f.ndcg.mean()

          res_f = eval_df_f.groupby('bin').mean()

          wandb.log({"epoch":epoch_i, "train_res_bin": res_f, "train_ndcg":ndcg_tr})
          if print_bin_train:
            print("Train error by bin:")
            print(res_f)

        if test_last and not epoch_i == epoch-1:
          continue
        resf = test(model, valid_data_loader, device, score_func=mean_squared_error, config_t=cft, epoch=epoch_i)
        eval_df_f = get_eval_df(resf, counts_tr)
        try:
          if test_last and 'path' in config.dic.keys():
            with open(config.dic['path'], 'wb') as f:
              pickle.dump(resf, f)
        except:
          None 

        ndcg_t = eval_df_f.ndcg.mean()

        config.ndcg_best = max(config.ndcg_best, ndcg_t)

        res_f = eval_df_f.groupby('bin').mean()


        score_val = eval_df_f['auc'].mean()
        print('epoch:', epoch_i, 'val score:', score_val, 'val ndcg:', ndcg_t)
        print("Valid error by bin:")
        print(res_f.drop(config.uid_name, axis=1))
        print("field_weights:\n", model.field_weights_u, "\n", model.field_weights_i)

        wandb.log({"epoch":epoch_i, "field_weights_u":model.field_weights_u.detach().cpu().numpy(), "field_weights_i":model.field_weights_i.detach().cpu().numpy()})
        prefix = "val" if not config.is_test else "test"

        wandb.log({"epoch":epoch_i, "val_res_bin": res_f.drop(config.uid_name, axis=1),\
                    f"{prefix}_ndcg_best":config.ndcg_best, f"{prefix}_ndcg":ndcg_t})
        wandb.log({f"{prefix}_ndcg@{i}_best":config.ndcgs_best[f'ndcg@{i}'] for i in config.ndcg_at})

        score = None
        if 'auc' in config.metrics:
          score_val = eval_df_f['auc'].mean() 
          higher_better = True

        if (higher_better and score_val > best_score)\
             or (not higher_better and score_val < best_score):
          best_score = score_val
          if not config.is_test:
            wandb.log({"epoch":epoch_i, "best_score_val":best_score, "ndcg_t_best":ndcg_t})
          else:
            wandb.log({"epoch":epoch_i, "best_score_test":best_score, "ndcg_t_best_test":ndcg_t})

          model.save_model(config.best_model_path)



        score = 10000-score_val if rev_metrics else score_val
        if not early_stopper.is_continuable(model, score):
            print(f'validation: best score_val: {early_stopper.best_accuracy}')
            break


    if test_data_loader:
      score_val = test(model, test_data_loader, device, score_func=score_func)
      print(f'test score_val: {score_val}')




