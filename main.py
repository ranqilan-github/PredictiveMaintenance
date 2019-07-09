# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:50:38 2019

@author: makraus
"""

from sklearn.metrics import mean_absolute_error
import numpy as np

import pyro
import torch
from pyro.distributions import Normal, Gumbel, LogNormal, Gamma
from pyro.infer import SVI
from pyro.optim import Adam

from helper import read_data, get_train_test_seqs
import model
 
sequence_length = 50
engines_eval = [1, 2]
cuda = True
ftype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
ltype = torch.cuda.LongTensor if cuda else torch.LongTensor

sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['cycle', 'setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

train_df, test_df = read_data()
trainX, trainY, valX, valY, testX, testY = get_train_test_seqs(train_df, test_df, sequence_length)

engines = []    
for engine_id in engines_eval:
    engines.append([]) 
    train_one_eng = train_df[train_df.id == engine_id]
    for i in range(train_one_eng.shape[0]):
        engines[-1].append(train_one_eng[sequence_cols].values[:i])
        
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['cycle', 'setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

trainX = np.vstack([trainX, valX])
trainY = np.vstack([trainY, valY])

optim = Adam({'lr': 0.005})
svi = SVI(model.model, model.guide, optim, loss='ELBO', num_particles=1)

y_data = trainY.squeeze(-1)
x_data, y_data = torch.tensor(trainX).type(ftype), torch.tensor(y_data).type(ftype)

y_test = testY.squeeze(-1)
x_test, y_test = torch.tensor(testX).type(ftype), torch.tensor(y_test).type(ftype)


def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


def run():
    pyro.clear_param_store()
    best_loss = np.inf
    best_MAE = np.inf
    early_stopping_counter = 0
    for j in range(100000):
        epoch_loss = 0.0
        perm = torch.randperm(trainX.shape[0]).type(ltype)
        x_data = x_data[perm]
        y_data = y_data[perm]
        # get indices of each batch
        all_batches = get_batch_indices(trainX.shape[0], 512)
        for ix, batch_start in enumerate(all_batches[:-1]):
            batch_end = all_batches[ix + 1]
            x_batch_data = x_data[batch_start: batch_end]
            y_batch_data = y_data[batch_start: batch_end]
            epoch_loss += svi.step(x_batch_data, y_batch_data)
        
        loss = epoch_loss / float(x_data.shape[0])
        print("{}: epoch avg loss {}".format(j, loss))
        if loss < best_loss:
            best_loss = loss
            early_stopping_counter = 0
        
        if j % 1 == 0:
            pred = []
            for _ in range(100):
                regression_sample, lstm_sample = model.guide(None, None)
                y_reg_pred = regression_sample(x_test).cpu().data.numpy()[:, -1, 0]
                y_lstm_pred = lstm_sample(torch.transpose(x_test, 0, 1)).cpu().data.numpy().reshape(-1,)
                y_pred = np.sum([y_reg_pred, y_lstm_pred], axis=0)
                pred.append(y_pred)
            pred = np.mean(pred, axis=0)
            MAE = mean_absolute_error(pred, testY)
            if MAE < best_MAE:
                best_MAE = MAE
                print('MAE: {}'.format(MAE))
        
        if early_stopping_counter > 1000:
            break
                  
        early_stopping_counter += 1
        
        
if __name__ == '__main__':
    run()
    