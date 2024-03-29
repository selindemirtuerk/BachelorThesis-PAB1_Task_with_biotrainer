from __future__ import print_function
import os.path
import random
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from sequence_env_m_p import Seq_env, Mutate
from mcts_alphaZero_mutate_expand_m_p_gfp import MCTSMutater
from p_v_net_torch import PolicyValueNet  # Pytorch
from p_v_net_3 import PolicyValueNet
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Union
import sys
import os
cwd = os.getcwd()
biotrainer = os.path.join(cwd, 'biotrainer') 
sys.path.append(biotrainer)
from biotrainer.utilities import cli
from biotrainer.inference import Inferencer
from biotrainer.protocols import Protocol
from biotrainer.embedders import EmbeddingService, get_embedding_service
from fasta_converter import FastaConverter
from utils import parse_arguments, extract_starting_seq
import datetime
from pathlib import Path
import copy
import shutil
from tqdm import tqdm
from data_viz import DataVisualization

AAS = "ILVAGMFYWEDQNHCRKSTP"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_via_biotrainer(config_file_path):
    # Define the output file path
    output_file_path = Path(os.path.join(cwd, "oracle_training/output/out.yml"))

    # Check if the output file does NOT exist
    if not os.path.exists(output_file_path):
        
        cli.headless_main(config_file_path)
    # At this point, the output file should exist, either from previous execution or the training above
        
    # Create the Inferencer from the output file
    inferencer, _ = Inferencer.create_from_out_file(Path(output_file_path))
    # You can now call the inference like this:
    # inferencer.from_embeddings(embeddings)
    return inferencer


class TrainPipeline():
    def __init__(self, start_seq, alphabet, model, embedding_service, trust_radius, one_hot_switch, init_model=None): #init_model=None
        self.seq_len = len(start_seq)
        self.vocab_size = len(alphabet)
        self.n_in_row = 4
        self.embedding_service = embedding_service
        self.seq_env = Seq_env(
            self.seq_len,
            alphabet,
            model,
            embedding_service,
            start_seq,
            trust_radius)  #n_in_row=self.n_in_row
        self.mutate = Mutate(self.seq_env)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 200  # num of simulations for each move 400   1600
        self.c_puct = 10 #0.5  # 10
        self.buffer_size = 10000
        self.batch_size = 4 #128  # mini-batch size for training  512 # was 32 before
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        #self_added
        self.buffer_no_extend = False
        #self_added
        #playout
        self.generated_seqs = []
        self.fit_list = []
        self.p_dict = {}
        self.m_p_dict = {}
        self.retrain_flag = False
        self.part = 2
        #playout
        use_gpu = False if device=="cpu" else True

        print(f"Using GPU: {use_gpu}")

        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.seq_len,
                                                   self.vocab_size,
                                                   embedding_service, #embedder
                                                   model_file=init_model,
                                                   use_gpu=use_gpu,
                                                   one_hot_switch=one_hot_switch
                                                   )
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.seq_len,
                                                   self.vocab_size,
                                                   embedding_service, #embedder
                                                   use_gpu=use_gpu,
                                                   one_hot_switch=one_hot_switch
                                                   )
        print("Number of free parameters (PolicyValueNet): " +
              f"{sum([p.numel() for p in self.policy_value_net.policy_value_net.parameters() if p.requires_grad])}")

        self.mcts_player = MCTSMutater(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        print("Created MCTSMutater player")

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        counts = len(self.generated_seqs)
        self.buffer_no_extend = False
        for i in tqdm(range(n_games), desc="Playing"):
            play_data, seq_and_fit = self.mutate.start_mutating(self.mcts_player,
                                                          temp=self.temp)    #winner,
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            
            
            if self.episode_len == 1: #Modified to 1, because we do not want unsuccessful runs in there, never 0 ?
                self.buffer_no_extend = True
            else:
                self.data_buffer.extend(play_data)
                for seq, fit in seq_and_fit:  #alphafold_d
                    if seq not in self.generated_seqs:
                        self.generated_seqs.append(seq)
                        self.fit_list.append(fit)
                        
                        if seq not in self.seq_env.playout_dict.keys(): #xxrs: adds one seq when it fails to improve during a play (mutation end)
                            self.seq_env.playout_dict[seq] = fit
                       

    def policy_update(self):
        """update the policy-value net"""
        
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        
        state_batch = [data[0] for data in mini_batch]
        
        mcts_probs_batch = [data[1] for data in mini_batch]
        
        winner_batch = [data[2] for data in mini_batch]
       
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch, #added [0] here 
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy


    def run(self, output_dir, num_of_sequences):
        """run the training pipeline"""
        starttime = datetime.datetime.now() 
        #part = 2
        if True:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))

                if len(self.seq_env.playout_dict.keys()) >= num_of_sequences:
                    m_p_fitness = np.array(list(self.seq_env.playout_dict.values()))
                    m_p_seqs = np.array(list(self.seq_env.playout_dict.keys()))
                    df_m_p = pd.DataFrame(
                        {"sequence": m_p_seqs, "pred_fit": m_p_fitness})
                    df_m_p.to_csv( output_dir / "generated_sequences_and_scores.csv", index=False)
                    print(f"File successfully saved to {output_dir}")
                    endtime = datetime.datetime.now() 
                    print('time cost：',(endtime-starttime).seconds)
                    break
                    #sys.exit(0)

                if len(self.data_buffer) > self.batch_size and self.buffer_no_extend == False:
                    if True:
                        loss, entropy = self.policy_update()
                    
        
               
if __name__ == '__main__':
    
    starttime = datetime.datetime.now()
    seed = 144
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # argument parsing for the dataset, num of desired generated sequences and the output directory
    dataset_file_path, num_of_sequences, output_dir_name = parse_arguments()
    #extract a starting sequence from the given dataset
    starting_sequence = extract_starting_seq(dataset_file_path)

    #create output directory
    output_dir = os.path.join(cwd, output_dir_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = Path(output_dir)

    #biotrainer requires the fasta format so sequences file is first converted into a fasta file
    sequences_file = os.path.join(cwd, "oracle_training/sequences.fasta")
    converter = FastaConverter(input_file=dataset_file_path, output_file=sequences_file)
    converter.convert_to_fasta()

    #get config file for oracle training
    config_file = Path(os.path.join(cwd, "oracle_training/config.yml"))
    model = train_via_biotrainer(config_file)

    #get emebedding service for creating embeddings for the oracle and the policy value net
    embedding_service: EmbeddingService = get_embedding_service(embedder_name="Rostlab/prot_t5_xl_uniref50", embeddings_file_path=None,
                                                            use_half_precision=False, device=device)
        
    # flag for using one-hots in policy net training 
    one_hot_switch = True

    training_pipeline = TrainPipeline(
        starting_sequence, 
        AAS,
        model,
        embedding_service,
        trust_radius=100,
        one_hot_switch = one_hot_switch
    )
    training_pipeline.run(output_dir, num_of_sequences)
   
    #add the evolution of scores, the distribution of scores and the top 10 sequences of the run to output directory
    output_dir = Path(os.path.join(cwd, output_dir_name))
    results_file_path = output_dir / "generated_sequences_and_scores.csv"
    data_visualiser = DataVisualization(results_file_path, output_dir)
    data_visualiser.create_data_visualisations(num_of_sequences, dataset_file_path, results_file_path)
   
    


   
    
    
    
    