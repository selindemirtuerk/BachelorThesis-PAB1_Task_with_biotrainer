# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

AAS = "ILVAGMFYWEDQNHCRKSTP"

def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:

    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out

def string_to_feature(string):
    seq_list = []
    seq_list.append(string)
    seq_np = np.array(
        [string_to_one_hot(seq, AAS) for seq in seq_list]
    )
    one_hots = torch.from_numpy(seq_np)
    one_hots = one_hots.to(torch.float32)
    return one_hots

def predict(model,inputs):
    one_hots_0= string_to_one_hot(inputs, AAS)
    one_hots = torch.from_numpy(one_hots_0)
    one_hots = one_hots.unsqueeze(0)
    one_hots = one_hots.to(torch.float32)
    with torch.no_grad():
        inputs = one_hots
        inputs = inputs.permute(0, 2, 1)
        outputs = model(inputs)
        outputs = outputs.squeeze()
    return outputs

class CNN(nn.Module):
    """predictor network module"""

    def __init__(
        self,
        seq_len,
        alphabet_len,
    ):
        super(CNN, self).__init__()
        self.board_width = seq_len
        self.board_height = alphabet_len
        # conv layers
        self.conv1 = nn.Conv1d(20, 32, kernel_size=3) #
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1) #, padding=0
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1) # , padding=0
        self.maxpool1 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=1, stride=1)

        self.val_fc1 = nn.Linear(7520, 100)

        self.val_fc2 = nn.Linear(100, 100)  # * alphabet_len
        self.dropout = nn.Dropout(p=0.25)
        self.val_fc3 = nn.Linear(100, 1)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.maxpool1(x))

        x = F.relu(self.conv3(x))
        x_act = F.relu(self.maxpool2(x))
        x_score_1 = x_act.view(x_act.shape[0], -1)
        
        x_score_2 = F.relu(self.val_fc1(x_score_1))
        x_score_2 = F.relu(self.val_fc2(x_score_2))
        x_score_2 = self.dropout(x_score_2)
        x_score_3 = self.val_fc3(x_score_2)
        return x_score_3

    
class CNN2(nn.Module):
    """predictor network module"""

    def __init__(
        self,
        #seq_len,
        #alphabet_len,
    ):
        super(CNN2, self).__init__()
        #self.board_width = seq_len
        #self.board_height = alphabet_len
        # conv layers
        
        self.conv1 = nn.Conv1d(1024, 32, kernel_size=3) #
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1) #, padding=0
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1) # , padding=0

        self.maxpool1 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=1, stride=1)

        self.val_fc1 = nn.Linear(13536, 100)# 2336 #* alphabet_len
        self.val_fc2 = nn.Linear(100, 100)  # * alphabet_len
        self.dropout = nn.Dropout(p=0.25)
        self.val_fc3 = nn.Linear(100, 1)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.maxpool1(x))

        x = F.relu(self.conv3(x))
        x_act = F.relu(self.maxpool2(x))

        x_score_1 = x_act.view(x_act.shape[0], -1)

        x_score_2 = F.relu(self.val_fc1(x_score_1))
        x_score_2 = F.relu(self.val_fc2(x_score_2))
        x_score_2 = self.dropout(x_score_2)
        x_score_3 = self.val_fc3(x_score_2)
        return x_score_3    


class CNN3(nn.Module):
    """predictor network module"""

    def __init__(
        self,
    ):
        super(CNN3, self).__init__()
        # conv layers
        self.linear_project = nn.Linear(1024, 64)
        self.conv1 = nn.Conv1d(64, 32, kernel_size=7, padding=3) #
        self.conv2 = nn.Conv1d(32, 32, kernel_size=7, padding=3) #, padding=0
        self.conv3 = nn.Conv1d(32, 32, kernel_size=7, padding=3) # , padding=0

        self.maxpool = torch.nn.MaxPool2d((32, 1), stride=(1, 1))

        self.val_fc1 = nn.Linear(1, 64)# 2336 #* alphabet_len
        self.val_fc2 = nn.Linear(64, 32)  # * alphabet_len
        self.dropout = nn.Dropout(p=0.1)
        self.val_fc3 = nn.Linear(32, 1)

    def forward(self, input):
        x = F.relu(self.linear_project(input))
        x = x.permute(0,2,1)
        x = self.dropout(x)
        x = F.relu(self.conv1(x))
        
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        
        x = self.dropout(x)
        x = F.relu(self.conv3(x))

        x_act = self.maxpool(x)
        x_score_1 = x_act.view(x_act.shape[0], -1)

        x_score_2 = F.relu(self.val_fc1(x_score_1))
        x_score_2 = F.relu(self.val_fc2(x_score_2))
        x_score_3 = self.val_fc3(x_score_2)
        return x_score_3   

    
class FNN( nn.Module ):
    
    def __init__( self ):
        super(FNN, self).__init__()
        # Linear layer, taking embedding dimension 1024 to make predictions:

        self.layer = nn.Sequential(
                        nn.Linear( 1024, 32),
                        nn.Dropout( 0.25 ),
                        nn.ReLU(),
                        )
        
        self.activitiy_head = nn.Linear( 32, 1)


    def forward( self, x):
        # Inference
        rep   = self.layer( x ) # swap feature dimension to second axis
        score = self.activitiy_head(rep)
        return score