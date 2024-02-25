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
sys.path.append('/mnt/project/demirtuerks/PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/biotrainer/')
from biotrainer.utilities import cli
from biotrainer.inference import Inferencer
from biotrainer.protocols import Protocol
from biotrainer.embedders import EmbeddingService, get_embedding_service
from fasta_converter import FastaConverter
import datetime
from pathlib import Path
import copy
import shutil
from tqdm import tqdm

root_data = Path("/mnt/project/demirtuerks/PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/data/")
root_code = Path("/mnt/project/demirtuerks/PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/trial/")

data_dir = root_data / 'PAB1.txt' #'/data/PAB1_GFP_data/PAB1.txt'

pab1_wt_sequence = (
        "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP"
        #"LTAPSIKSGTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPINQVKEGNQGDKSMSNWYWLYQPTSYQIGNRYLGTEQEFKEMCAAAEEYGIKVIVDAVINHTTSDYAAISNEVKSIPNWTHGNTPIKNWSDRWDVTQNSLSGLYDWNTQNTQVQSYLKRFLDRALNDGADGFRFDAAKHIELPDDGSYGSQFWPNITNTSAEFQYGEILQDSVSRDAAYANYMDVTASNYGHSIRSALKNRNLGVSNISHYAVDVSADKLVTWVESHDTYANDDEESTWMSDDDIRLGWAVIASRSGSTPLFFSRPEGGGNGVRFPGKSQIGDRGSALFEDQAITAVNRFHNVMAGQPEELSNPNGNNQIFMNQRGSHGVVLANAGSSSVSINTATKLPDGRYDNKAGAGSFQVNDGKLTGTINARSVAVLYPD"
    )
starts = {
        "start_seq": "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALKGMLLNGQEIYFAP",
        #"LTAPSIKSGTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPINQVKEGNQGDKSMSNWYWLYQPTSYQIGNRYLGTEQEFKEMCAAAEEYGIKVIVDAVINHTTSDYAAISNEVKSIPNWTHGNTPIKNWSDRWDVTQNSLSGLYDWNTQNTQVQSYLKRFLDRALNDGADGFRFDAAKHIELPDDGSYGSQFWPNITNTSAEFQYGEILQDSVSRDAAYANYMDVTASNYGHSIRSALKNRNLGVSNISHYAVDVSADKLVTWVESHDTYANDDEESTWMSDDDIRLGWAVIASRSGSTPLFFSRPEGGGNGVRFPGKSQIGDRGSALFEDQAITAVNRFHNVMAGQPEELSNPNGNNQIFMNQRGSHGVVLANAGSSSVSINTATKLPDGRYDNKAGAGSFQVNDGKLTGTINARSVAVLYPD",
        "start_mut_good": "ATAPSIKSMTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPIMQVKEGNQGDKSMSNWYWLYQPTSYHIGNRYLGTEQEFKEMCAAAEEYGIKVIVDAVLNHTTSDYAAISNEVKSIPNWTHGNTPIKNWSDRWDVTQNSLLGLYDWNTQNTQVQSYLKRFLDRALNDGADGFRFDAAKHIELPDDGSYGSQFWPNITNTSAEFQYGEILQDSVSRDAAYANYMDITASNYGHSIRSALKNRNLGVSNISHYAIDVSADKLVTWVESHDTYANDDEESTWMSDDDIRLGWAVIASRSGSTPLFFSRPEGGGNGVRFPGKSQIGDRGSALFEDQAITAVNRFHNVMAGQPEELSNPNGNNQIFMNQRGSHGVVLANAGSSSVSINTATKLPDGRYDNKAGAGSFQVNDGKLTGTINARSVAVLYAD", # noqa: E501
        "start_mut_bad" : "LTAPSIKSGTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPINQVKEGNQGDKSMSNWYWLYQPTSYQIGNRYLGTEQEFKEMCAAAEEYGIKVIVDAVINHTTSDYAAISAEVKSIPNWTHGNTPIKNWSDRWDVTQNSLSGLYDWNTQNTQVQSYLKRFLDRALNDGADGFRFDAAKHIELPDDGSYGSQFWPNITNTSAEFQYGEILQDSVSRDAAYANYMDITASNYGHSIRSALKNRNLGVSNISHYAVDVSADKLVTWVESHDTYANDDEESTWMSDDDIRLGWAVIASRSGSTPLFFSRPEGGGNGVRFPGKSQIGDRGSALFEDQAITAVNRFHNVMAGQPEELSNPNGNNQIFMNQRGSHGVVLANAGSSSVSINTATKLPDGRYDNKAGAGSFQVNDGKLTGTINARSVAVLYPD",
        "start_mut_superBad" : "LTAPSIKSGTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPINQVKEGNQGDKSMSNWYWLYQPTSYQIGNRYLGTEQEFKEMCAAAEEYGIKVIVDAVGNHTTSDYAAISNEVKSIPNWTHGNTPIKNWSDRWDVTQNSLSGLYDWNTQNTQVQSYVKRFLDRALNDGADGFRFDAAKHIELPDDGSYGSQFWPNITNTSAEFQYGEILQDSVSRDAAYANYMDVTASNYGHSIRSALKNRNLGVSNISHYAVDVSADKLVTWVESHDTYANDDEESTWMSDDDIRLGWAVIASRSGSTPLFFSRPEGGGNGVRFPGKSQIGDRGSALFEDQAITAVNRFHNVMAGQPEELSNPNGNNQIFMNQRGSHGVVLANAGSSSVSINTATKLPDGRYDNKAGAGSFQVNDGKLTGTINARSVAVLYPD"
    }
AAS = "ILVAGMFYWEDQNHCRKSTP"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_via_biotrainer(config_file_path):
    # Define the output file path
    output_file_path = Path(os.path.join(cwd, "PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/oracle_training/output/out.yml"))

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
        print("collect self-play data for training")
        counts = len(self.generated_seqs)
        self.buffer_no_extend = False
        for i in tqdm(range(n_games), desc="Playing"):
        #for i in n_games:
            play_data, seq_and_fit, p_dict = self.mutate.start_mutating(self.mcts_player,
                                                          temp=self.temp)    #winner,
            print("after start mutating")
            play_data = list(play_data)[:]

            self.episode_len = len(play_data)
            
            self.p_dict = p_dict
            self.m_p_dict.update(self.p_dict)

            if self.episode_len == 1: #Modified to 1, because we do not want unsuccessful runs in there, never 0 ?
                self.buffer_no_extend = True
            else:
                self.data_buffer.extend(play_data)
                for seq, fit in seq_and_fit:  #alphafold_d
                    if seq not in self.generated_seqs:
                        self.generated_seqs.append(seq)
                        self.fit_list.append(fit)
                        if seq not in self.m_p_dict.keys():
                            self.m_p_dict[seq] = fit
                        if len(self.generated_seqs)%10==0 and len(self.generated_seqs)>counts and self.part<=10:
                            self.retrain_flag=True
                       

    def policy_update(self):
        """update the policy-value net"""
        print("update the policy-value net")
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        print("mini_batch length:{}".format(len(mini_batch)))
        state_batch = [data[0] for data in mini_batch]
        print("state_batch length:{}".format(len(state_batch)))
        mcts_probs_batch = [data[1] for data in mini_batch]
        #####
        print("mcts_probs_batch length:{}".format(len(mcts_probs_batch)))
        #print("mcts_probs_batch:")
        #print(mcts_probs_batch)
        print("mcts_probs_batch[0]:")
        print(mcts_probs_batch[0])
        print(len(mcts_probs_batch))
        #####
        winner_batch = [data[2] for data in mini_batch]
        print("winner_batch length:{}".format(len(winner_batch)))
        #print("winner_batch:")
        #print(winner_batch)
        print("winner_batch[0]:")
        print(winner_batch[0])
        print(len(winner_batch))
        #####
        
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


    def run(self):
        """run the training pipeline"""
        starttime = datetime.datetime.now() 
        #part = 2
        try:
            if True:
                for i in range(self.game_batch_num):
                    self.collect_selfplay_data(self.play_batch_size)
                    print("batch i:{}, episode_len:{}".format(
                            i+1, self.episode_len))

                    if len(self.m_p_dict.keys()) >= 8000:
                        m_p_fitness = np.array(list(self.m_p_dict.values()))
                        m_p_seqs = np.array(list(self.m_p_dict.keys()))
                        df_m_p = pd.DataFrame(
                            {"sequence": m_p_seqs, "pred_fit": m_p_fitness})
                        df_m_p.to_csv( root_code / "trial_emb_modified_8000.csv",index=False)
                        endtime = datetime.datetime.now() 
                        print('time cost：',(endtime-starttime).seconds)
                        sys.exit(0)

                    print("data buffer: ", len(self.data_buffer))
                    print("batch_size: ", self.batch_size)
                    print("buffer_no_extend: ", self.buffer_no_extend)

                    if len(self.data_buffer) > self.batch_size and self.buffer_no_extend == False:
                        #try:
                        if True:
                            loss, entropy = self.policy_update()
                        #except TypeError as e:
                        #    print(e)
                        #    continue

        except KeyboardInterrupt:
            try:
                output_directory = Path(os.path.join(cwd, "PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/single_emb/"))
                folder_to_delete = output_directory / "sequence_to_value"
                shutil.rmtree(folder_to_delete)
                print("sequence_to_value folder removed")
                folder_to_delete = output_directory / "residue_to_class"
                shutil.rmtree(folder_to_delete)
                print("residue_to_class folder removed")
            except Exception as error:
                pass
        
                        
def main(filePath):
    
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

    cwd = os.getcwd()
    sequences_file = os.path.join(cwd, "oracle_training/sequences.fasta")
    input_file = filePath
    converter = FastaConverter(input_file=input_file, output_file=sequences_file)
    converter.convert_to_fasta()
    print("converted to fasta")

    # embedding service for sequence_to_value predictions : the oracle
    ###

    config_file = Path(os.path.join(cwd, "oracle_training/config.yml"))
    model = train_via_biotrainer(config_file)

    # embedding service for the policy neural net
    ###

    protocol = Protocol.residue_to_class
    embedding_service: EmbeddingService = get_embedding_service(embedder_name="Rostlab/prot_t5_xl_uniref50", embeddings_file_path=None,
                                                            use_half_precision=True, device=device)
    
    # flag for using only emb in policy net training 
    one_hot_switch = False

    #print(f"Number of free parameters (oracle): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    training_pipeline = TrainPipeline(
        starts["start_seq"], 
        AAS,
        model,
        embedding_service,
        trust_radius=100,
        one_hot_switch = one_hot_switch
    )
    training_pipeline.run()
    
    

                 
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

    cwd = os.getcwd()
    sequences_file = os.path.join(cwd, "PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/oracle_training/sequences.fasta")
    input_file = os.path.join(cwd, "PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/data/PAB1.txt")
    converter = FastaConverter(input_file=input_file, output_file=sequences_file)
    converter.convert_to_fasta()

    # embedding service for sequence_to_value predictions : the oracle
    ###

    config_file = Path(os.path.join(cwd, "PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/oracle_training/config.yml"))
    model = train_via_biotrainer(config_file)

    # embedding service for the policy neural net
    ###

    embedding_service: EmbeddingService = get_embedding_service(embedder_name=model.embedder_name, embeddings_file_path=None,
                                                            device=device)
    inputs = Path(os.path.join(cwd, "PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/oracle_training/output/sequence_to_value/prot_t5_xl_uniref50/reduced_embeddings_file_prot_t5_xl_uniref50.h5"))
    embeddings = embedding_service.load_embeddings(inputs)
    predictions = model.from_embeddings(embeddings)["mapped_predictions"]
    count = 0
    for sequence, score in predictions.items():
        if count < 5:
            print(f'Sequence: {sequence}, Score: {score}')
            count += 1
        else:
            break  # Exit the loop after 5 iterations

    embedding_service: EmbeddingService = get_embedding_service(embedder_name="Rostlab/prot_t5_xl_uniref50", embeddings_file_path=None,
                                                            use_half_precision=False, device=device)
    
    # flag for using only emb in policy net training 
    one_hot_switch = True

    #print(f"Number of free parameters (oracle): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    training_pipeline = TrainPipeline(
        starts["start_seq"], 
        AAS,
        model,
        embedding_service,
        trust_radius=100,
        one_hot_switch = one_hot_switch
    )
    training_pipeline.run()
    
    