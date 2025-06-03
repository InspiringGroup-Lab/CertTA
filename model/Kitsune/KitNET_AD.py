import torch
import dA as AE
import numpy as np
import corClust as CC
import torch.nn as nn

# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# For licensing information, see the end of this document

class KitNET_AD(nn.Module):
    #n: the number of features in your input dataset (i.e., x \in R^n)
    #m: the maximum size of any autoencoder in the ensemble layer
    #AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    #FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period
    #learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    #hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    #feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
    #           where the i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
    #           For example, [[2,5,3],[4,0,1],[6,7]]
    def __init__(self,feat_num,labels_num,max_autoencoder_size=10,hidden_ratio=0.75,feature_map=None, device='cpu'):
        super(KitNET_AD, self).__init__()
        # Parameters:
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.hr = hidden_ratio
        self.n = feat_num
        self.labels_num = labels_num

        # Variables
        self.device = device
        self.FM = CC.corClust(self.n) #incremental feature cluatering for the feature mapping process
        self.ensembleLayer = []
        self.v = feature_map
        self.train_distances = None
        if self.v is None:
            print("Feature-Mapper: train-mode, Anomaly-Detector: off-mode")
        else:
            self.__createAD__()
            print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        

    #x: a numpy array of length n
    #Note: KitNET automatically performs 0-1 normalization on all attributes.
    def train_FM(self, x):
        assert self.v is None
        #update the incremetnal correlation matrix
        self.FM.update(x)
    
    def build_FM(self):
        #If the feature mapping should be instantiated
        self.v = self.FM.cluster(self.m)
        self.__createAD__()
        print("The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders.")
        print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")

    def forward(self,x):
        # Ensemble Layer
        hiddens = []
        loss_rec = torch.zeros([x.shape[0]]).to(self.device)
        for a in range(len(self.ensembleLayer)):
            # make sub inst
            indices = torch.tensor(self.v[a]).to(self.device)
            xi = x.index_select(-1, indices)
            hidden_i, xi_rec = self.ensembleLayer[a].forward(xi)
            hiddens.append(hidden_i)
            loss_rec_i = torch.sqrt(torch.pow(xi - xi_rec, 2).mean(dim=-1) + 1e-6)
            loss_rec += loss_rec_i
        
        return loss_rec

    def __createAD__(self):
        # construct ensemble layer
        for map in self.v:
            params = AE.dA_params(n_visible=len(map), n_hidden=0, hiddenRatio=self.hr)
            ae = AE.dA(params, self.device).to(self.device)
            self.ensembleLayer.append(ae)


# Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.