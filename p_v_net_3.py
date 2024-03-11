# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import os
from pathlib import Path
import sys
cwd = os.getcwd()
biotrainer = os.path.join(cwd, 'biotrainer')  
sys.path.append(biotrainer)
from biotrainer.protocols import Protocol
import shutil



from typing import List, Union
AAS = "ILVAGMFYWEDQNHCRKSTP"


def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.

    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing
            a sequence.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        Sequence string representation of `one_hot`.

    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def single_embed_for_policy_net(policy_net, seq):

    protocol = Protocol.residue_to_class
    embedding_list = policy_net.embedding_service.compute_embeddings_from_list([seq], protocol)
    return embedding_list[0]

class Net_emb(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net_emb, self).__init__()

        self.board_width = board_width # sequence length
        self.board_height = board_height # alphabet length (20)
        # common layers
        self.linear_project = nn.Linear(1024, 32)
        self.conv1 = nn.Conv1d(32, 32, kernel_size=7, padding=3) #32
        self.conv2 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        # action policy layers
        self.act_conv1 = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        self.act_fc1 = nn.Linear(board_width,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        self.val_fc1 = nn.Linear(board_width, 128) 
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.linear_project(state_input))
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, self.board_width)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, self.board_width)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val
    
class Net_one_hot(nn.Module):
    """policy-value network module"""
    #print("Net2_one_hot is used!")
    def __init__(self, board_width, board_height):
        super(Net_one_hot, self).__init__()

        self.board_width = board_width # sequence length
        self.board_height = board_height # alphabet length (20)
        # common layers
        #self.linear_project = nn.Linear(1024, 32)
        self.conv1 = nn.Conv1d(20, 32, kernel_size=7, padding=3) #32
        self.conv2 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        # action policy layers
        self.act_conv1 = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        self.act_fc1 = nn.Linear(board_width,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        self.val_fc1 = nn.Linear(board_width, 128) 
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, self.board_width)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, self.board_width)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, embedding_service, one_hot_switch,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width # sequence length
        self.board_height = board_height # alphabet length (20)
        self.embedding_service = embedding_service
        self.one_hot_switch = one_hot_switch
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            if self.one_hot_switch:
                self.policy_value_net = Net_one_hot(board_width, board_height).cuda()
            else:
                self.policy_value_net = Net_emb(board_width, board_height).cuda()
        else:
            if self.one_hot_switch:
                self.policy_value_net = Net_one_hot(board_width, board_height)
            else:
                self.policy_value_net = Net_emb(board_width, board_height)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)


    def state2emb(self,state_batch):
        embs = list()
        for state in state_batch:
            one_hot = torch.from_numpy(state).permute(1, 0)
            seq = one_hot_to_string(one_hot, AAS)
            emb = single_embed_for_policy_net(self, seq)
            if self.use_gpu:
                emb = torch.from_numpy(emb).to(torch.float32).unsqueeze(dim=0).cuda()
            else:
                emb = torch.from_numpy(emb).to(torch.float32).unsqueeze(dim=0)
            embs.append(emb)
        return torch.cat(embs,dim=0)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            if self.one_hot_switch:
                state_batch = Variable(torch.FloatTensor(torch.from_numpy(np.asarray(state_batch))).cuda()) # added torch.from_numpy() here
            else:
                state_batch = self.state2emb(state_batch)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            if self.one_hot_switch:
                state_batch = Variable(torch.FloatTensor(torch.from_numpy(np.asarray(state_batch)))) # added torch.from_numpy() here
            else:
                state_batch = self.state2emb(state_batch)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables

        current_state_0 = np.expand_dims(board.current_state(), axis = 0)
        current_state = np.ascontiguousarray(current_state_0)  ##

        if self.use_gpu:
            if self.one_hot_switch:
                log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            else:
                log_act_probs, value = self.policy_value_net(
                    self.state2emb(current_state).float())
                
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            if self.one_hot_switch:
                log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            else:
                log_act_probs, value = self.policy_value_net(
                    self.state2emb(current_state).float())
            
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            if self.one_hot_switch:
                state_batch = Variable(torch.FloatTensor(torch.from_numpy(np.asarray(state_batch))).cuda())
            else:
                state_batch = self.state2emb(state_batch)
            mcts_probs = Variable(torch.FloatTensor(torch.from_numpy(np.asarray(mcts_probs)).float()).cuda())
            winner_batch = Variable(torch.FloatTensor(torch.from_numpy(np.asarray(winner_batch)).float()).cuda())
        else:
            if self.one_hot_switch:
                state_batch = Variable(torch.FloatTensor(torch.from_numpy(np.asarray(state_batch))))
            else:
                state_batch = self.state2emb(state_batch)
            mcts_probs = Variable(torch.FloatTensor(torch.from_numpy(np.asarray(mcts_probs)).float()))
            winner_batch = Variable(torch.FloatTensor(torch.from_numpy(np.asarray(winner_batch))))

       
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)

        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
