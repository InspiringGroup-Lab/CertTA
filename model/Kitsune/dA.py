# Copyright (c) 2017 Yusuke Sugomori
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

# Portions of this code have been adapted from Yusuke Sugomori's code on GitHub: https://github.com/yusugomori/DeepLearning

import numpy
from utils import *
import torch.nn as nn


class dA_params:
    def __init__(self,n_visible = 5, n_hidden = 3, hiddenRatio=None):
        self.n_visible = n_visible# num of units in visible (input) layer
        self.n_hidden = n_hidden# num of units in hidden layer
        self.hiddenRatio = hiddenRatio

class dA(nn.Module):
    def __init__(self, params, device='cpu'):
        super(dA, self).__init__()
        self.params = params

        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(numpy.ceil(self.params.n_visible*self.params.hiddenRatio))

        self.device = device
        self.enc = nn.Sequential(
            nn.Linear(self.params.n_visible, self.params.n_hidden),
            nn.Sigmoid()
        ).to(device)
        
        self.dec = nn.Sequential(
            nn.Linear(self.params.n_hidden, self.params.n_visible),
            nn.Sigmoid()
        ).to(device)

    # Encode
    def get_hidden_values(self, input):
        return self.enc(input)

    # Decode
    def get_reconstructed_input(self, hidden):
        return self.dec(hidden)

    def forward(self, x):
        hidden = self.get_hidden_values(x)
        x_rec = self.get_reconstructed_input(hidden)
        return hidden, x_rec