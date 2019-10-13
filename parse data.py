# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:11:29 2017

@author: jtay
"""

import pandas as pd
import numpy as np
import os
import gzip

# JRD Notes:
# https://www.openml.org/d/31 (Credit risk)
#

# fashionmnist

path = '../fashionmnist/'
kind = 'train'

labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

with gzip.open(labels_path, 'rb') as lbpath:
	labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

with gzip.open(images_path, 'rb') as imgpath:
	images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

print(images.shape)
print(labels.shape)

images_df = pd.DataFrame(data=images).astype(float)
labels_df = pd.DataFrame(data=labels)
labels_df.columns = ['Class']
fmnist = pd.concat([images_df, labels_df],1)

print("Images head: ")
print(images_df.head())
print("Labels head: ")
print(labels_df.head())
print("Dataframe head: ")
print(fmnist.head())

# TODO: make dataframe of this and dump it as hdf
#fmnist.to_hdf('datasets_full.hdf','fmnist',complib='blosc',complevel=9)
#fmnist.to_csv('fmnist.csv', index=False)
samples = np.random.rand(len(fmnist.index)) < 0.2
fmnist_test = fmnist.loc[samples]
fmnist_train = fmnist.loc[np.invert(samples)]
print(fmnist_test.shape)
print(fmnist_train.shape)
fmnist_test.to_csv('fmnist_test.csv', index=False, header=False)
fmnist_train.to_csv('fmnist_train.csv', index=False, header=False)

# TODO: make dataframe of this and dump it as hdf

# chess

path = '../chess/'
kind = 'kr-vs-kp'

# All in one
data_labels_path = os.path.join(path, '%s.data' % kind)

chess = pd.read_csv(data_labels_path,header=None)

chess.columns = ['bkblk','bknwy','bkon8','bkona','bkspr','bkxbq','bkxcr','bkxwp','blxwp','bxqsq','cntxt','dsopp','dwipd',
                 'hdchk','katri','mulch','qxmsq','r2ar8','reskd','reskr','rimmx','rkxwp','rxmsq','simpl','skach','skewr',
                 'skrxp','spcop','stlmt','thrsk','wkcti','wkna8','wknck','wkovl','wkpos','wtoeg', 'win']

chess = pd.get_dummies(chess)
chess['win'] = chess['win_won']
chess = chess.drop(['win_won', 'win_nowin'], axis=1)

print("Chess head: ")
print(chess.head())
chess.to_csv('chess.csv', index=False)

# Need to split into training and testing data
samples = np.random.rand(len(chess.index)) < 0.2
chess_test = chess.loc[samples]
chess_train = chess.loc[np.invert(samples)]
print(chess_test.shape)
print(chess_train.shape)
chess_test.to_csv('chess_test.csv', index=False, header=False)
chess_train.to_csv('chess_train.csv', index=False, header=False)

# TODO: make dataframe of this and dump it as hdf
#chess.to_hdf('datasets_full.hdf','chess',complib='blosc',complevel=9)
