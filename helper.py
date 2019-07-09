#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:41:10 2017

@author: mkraus
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
    

def read_data():
    train_df = pd.read_csv('./data/PHM08/train.txt', sep=" ", header=None)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']

    train_df = train_df.sort_values(['id', 'cycle'])

    test_df = pd.read_csv('./data/PHM08/test.txt', sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']

    truth_df = pd.read_csv('./data/PHM08/truth.txt', sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)

    train_df['cycle_norm'] = train_df['cycle']
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=train_df.index)
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns = train_df.columns)


    test_df['cycle_norm'] = test_df['cycle']
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)

    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop('more', axis=1, inplace=True)

    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)
    
    return train_df, test_df


def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


def get_seqs_by_list(df_l, seq_len):
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols = ['cycle', 'setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)

    ret_seqs = []
    ret_label = []
    for df in df_l:
        seq_gen = (list(gen_sequence(df[df['id'] == id], seq_len, sequence_cols)) 
                        for id in df['id'].unique())
        seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
        npad = ((0, 0), (50-seq_len, 0), (0, 0))
        seq_padded = np.pad(seq_array, pad_width=npad, mode='constant', constant_values=0)

        label_gen = [gen_labels(df[df['id']==id], seq_len, ['RUL']) 
                     for id in df['id'].unique()]
        label_array = np.concatenate(label_gen).astype(np.float32)

        ret_seqs.append(seq_padded)
        ret_label.append(label_array)
    return ret_seqs, ret_label

def get_train_test_seqs(train_df, test_df, seq_len):
    sensor_cols = ['s' + str(i) for i in range(1,22)]
    sequence_cols = ['cycle', 'setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)

    train_seq_array_l = []
    train_label_array_l = []

    seq_gen = (list(gen_sequence(train_df[train_df['id']==id], seq_len, sequence_cols)) 
               for id in train_df['id'].unique() if id <= 90)
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    npad = ((0, 0), (50-seq_len, 0), (0, 0))
    train_seq_array_l.append(np.pad(seq_array, pad_width=npad, mode='constant', constant_values=0))

    label_gen = [gen_labels(train_df[train_df['id']==id], seq_len, ['RUL']) 
                 for id in train_df['id'].unique() if id <= 90]
    label_array = np.concatenate(label_gen).astype(np.float32)
    train_label_array_l.append(label_array)

    train_seq_array = np.vstack(train_seq_array_l)
    train_label_array = np.vstack(train_label_array_l)

    val_seq_array_l = []
    val_label_array_l = []

    seq_gen = (list(gen_sequence(train_df[train_df['id']==id], seq_len, sequence_cols)) 
               for id in train_df['id'].unique() if id > 90)
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    npad = ((0, 0), (50-seq_len, 0), (0, 0))
    val_seq_array_l.append(np.pad(seq_array, pad_width=npad, mode='constant', constant_values=0))

    label_gen = [gen_labels(train_df[train_df['id']==id], seq_len, ['RUL']) 
                 for id in train_df['id'].unique() if id > 90]
    label_array = np.concatenate(label_gen).astype(np.float32)
    val_label_array_l.append(label_array)

    val_seq_array = np.vstack(val_seq_array_l)
    val_label_array = np.vstack(val_label_array_l)

    test_seq_array = [test_df[test_df['id']==id][sequence_cols].values[-seq_len:] 
                           for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= seq_len]
    test_seq_array = np.asarray(test_seq_array).astype(np.float32)

    y_mask = [len(test_df[test_df['id']==id]) >= seq_len for id in test_df['id'].unique()]
    test_label_array = test_df.groupby('id')['RUL'].nth(-1)[y_mask].values
    test_label_array = test_label_array.reshape(test_label_array.shape[0],1).astype(np.float32)
    
    return train_seq_array, train_label_array, val_seq_array, val_label_array, test_seq_array, test_label_array
