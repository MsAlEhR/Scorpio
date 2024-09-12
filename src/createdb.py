"""
Author: Saleh Refahi
Email: sr3622@drexel.edu
"""

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from pathlib import Path
import time
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd
import faiss
import argparse
from TripletModel import AveragePool1dAlongAxis,Tuner
import re
import itertools 
import os
from collections import defaultdict, Counter
from itertools import product
from confidence_score import confidence_score
import yaml
from multiprocessing import Pool, Manager, cpu_count
from joblib import Parallel, delayed
from KmerTokenizer import KmerTokenizer

def read_fasta(file_path, buffer_size=4194304):
    x_header = []
    x_seq = []
    current_sequence = []
    current_header = None

    try:
        with open(file_path, "r", buffering=buffer_size) as f:
            pattern = re.compile(r'seqid\|(\d+)')
            indx = 0

            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    indx += 1
                    # Save the previous sequence if it exists
                    if current_sequence:
                        x_seq.append("".join(current_sequence))
                        current_sequence = []

                    # Extract the index string using the compiled regex pattern
                    current_header_match = pattern.search(line)
                    if current_header_match is not None:
                        index_string = current_header_match.group(1)
                        current_header = int(index_string)
                    else:
                        current_header = indx
                    x_header.append(current_header)
                else:
                    current_sequence.append(line)

            # Add the last sequence after exiting the loop
            if current_sequence:
                x_seq.append("".join(current_sequence))
                
    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return x_seq, x_header





def read_fastq(file_path):
    x_header = []
    x_seq = []
    x_qual = []

    try:
        with open(file_path, "r") as f:
            while True:
                header = f.readline().strip()
                if not header:  # End of file
                    break
                if not header.startswith("@"):
                    raise ValueError(f"Expected '@' at the beginning of the header, but found: {header}")

                # Extract the first part of the header
                header_main = header.split(" ")[0][1:]  # Remove the '@' and get the first part
                x_header.append(header_main)

                # Read the sequence
                sequence = f.readline().strip()
                x_seq.append(sequence)

                # Read the plus line
                plus_line = f.readline().strip()
                if not plus_line.startswith("+"):
                    raise ValueError("Expected '+' line, but found something else.")

                # Read the quality scores
                quality_scores = f.readline().strip()
                if len(quality_scores) != len(sequence):
                    raise ValueError("Quality scores length does not match sequence length.")
                x_qual.append(quality_scores)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the FASTQ file: {e}")

    return x_seq, x_header





def tokenize_sequence(sequence, tokenizer):
    return tokenizer.kmer_tokenize(sequence)

def kmer_tokenize(seqlist, n_jobs=12):
    tokenizer = KmerTokenizer(kmerlen=6, overlapping=True, maxlen=4096)
    tokenized_sequences = Parallel(n_jobs=n_jobs)(
        delayed(tokenize_sequence)(sequence, tokenizer) for sequence in tqdm(seqlist)
    )
    return np.array(tokenized_sequences)



def read_fasta_or_fastq(file_path):
    # Decide whether the file is FASTA or FASTQ based on its extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() in [".fq", ".fastq"]:  # Common extensions for FASTQ
        return read_fastq(file_path)
    elif ext.lower() in [".fa", ".fasta"]:  # Common extensions for FASTA
        return read_fasta(file_path)
    else:
        raise ValueError("Unrecognized file extension. Please provide a FASTA or FASTQ file.")



def calculate_single_sequence_kmer_frequencies(sequence, possible_kmers, k, sequence_len, index):
    sequence = sequence.upper()
    sequence_kmer_counts = Counter(sequence[j:j+k] for j in range(sequence_len - k + 1))
    frequencies = np.array([sequence_kmer_counts.get(kmer, 0) / sequence_len for kmer in possible_kmers], dtype=np.float16)
    return index, frequencies

def calculate_kmer_frequencies(sequences, k=6, n_jobs=16):
    possible_kmers = [''.join(p) for p in product('ATCG', repeat=k)]  

    # Parallel processing with index tracking
    frequencies_with_indices = Parallel(n_jobs=n_jobs)(
        delayed(calculate_single_sequence_kmer_frequencies)(
            sequence, possible_kmers, k, len(sequence), idx
        ) for idx, sequence in enumerate(tqdm(sequences))
    )

    # Sort by the original indices to maintain the order
    frequencies_with_indices.sort(key=lambda x: x[0])

    # Extract the k-mer frequencies, ignoring the indices
    frequencies = np.array([freq for idx, freq in frequencies_with_indices], dtype=np.float16)

    return frequencies



def load_data(max_len,db_fasta,test_fasta,cal_kmer_freq):
    print("Loading data ....")
    start = time.time()
    data_test, test_indices = read_fasta_or_fastq(test_fasta)
    data_train, train_indices = read_fasta_or_fastq(db_fasta)
    end = time.time()
    print("Loading data time:", end - start)
    print("Establishing initial encodings ....")
    if cal_kmer_freq :
        data_train=calculate_kmer_frequencies(data_train)
        data_test=calculate_kmer_frequencies(data_test)
    else:
        data_train=kmer_tokenize(data_train) 
        data_test=kmer_tokenize(data_test)

    return data_train,data_test,train_indices,test_indices





def load_model(weights_p,from_embedding,embedding_size):
    print("Loading Model ....")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # state = torch.load(weights_p, map_location=torch.device(device))['model_state_dict']
    # model = torch.load(weights_p, map_location=torch.device(device))["Tuner"]
    
    # Load the state_dict
    state = torch.load(weights_p, map_location=device)
    # Initialize your model
    model = Tuner(pretrained_model="MsAlEhR/MetaBERTa-bigbird-gene", embedding_size=1024, from_embedding=False)
    # model.from_embedding=from_embedding
    # model.embedding_size=embedding_size
    model.load_state_dict(state, strict=False)
    model = model.eval()
    print(".....")
    return model



import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

class EmbeddingModel(pl.LightningModule):
    def __init__(self, model):
        super(EmbeddingModel, self).__init__()
        self.model = model

    def forward(self, batch):
        return self.model.single_pass(batch)

def compute_embeddings(model, raw_embedding_test, raw_embedding_train, output, batch_size):
    print("Generating embeddings ...")
    
    # Wrap your model in a LightningModule
    embedding_model = EmbeddingModel(model)

    # PyTorch Lightning Trainer for multi-GPU inference
    trainer = pl.Trainer(
        accelerator='gpu',        # Use GPU(s)
        devices=torch.cuda.device_count(),  # Automatically use all available GPUs
        precision=16,             # Use mixed precision for faster inference
        strategy="dp",           # Use Distributed Data Parallel strategy for efficient multi-GPU usage
        inference_mode=True       # Enable inference mode for optimizations
    )

    def process_dataset(dataset, save_path):
        print(type(dataset),type(dataset[0]),dataset.shape,"eeeeeeeeeeeeeee")
        # Create DataLoader for the entire dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    
        # Use the trainer to predict on the entire dataset
        predictions = trainer.predict(embedding_model, dataloaders=dataloader)
        
        # Concatenate all predictions into one tensor
        embeddings = torch.cat(predictions, dim=0).cpu().detach().numpy()
        
        # Save the embeddings to a file
        np.save(save_path, embeddings)
        print(f"Saved embeddings to {save_path}. Shape: {embeddings.shape}")
        return embeddings


    # Process test and train datasets
    triplet_embedding_test = process_dataset(raw_embedding_test, f"{output}/val_embeddings.npy")
    triplet_embedding_train = process_dataset(raw_embedding_train, f"{output}/db_embeddings.npy")

    return triplet_embedding_test, triplet_embedding_train


def create_index(embedding,ouput):
    print("Creating index ....")
    num_features = embedding.shape[1]
    index = faiss.IndexFlatL2(num_features)
    index.add(embedding)
    index_file = os.path.join(ouput, 'faiss_index')
    faiss.write_index(index, index_file)
    return index




def save_args_to_yaml(args, yaml_file):
    with open(yaml_file, 'w') as file:
        yaml.dump(vars(args), file, default_flow_style=False)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_and_sort_headers(metadata_file):
    df = pd.read_csv(metadata_file)
    headers = df.columns.tolist()
    h_headers = [header for header in headers if header.startswith('h')]
    sorted_headers = sorted(h_headers, key=lambda x: int(x[1:].split('-')[0]))
    
    # Find unique values count in h0 category columns and get the maximum value
    h0_headers = [header for header in sorted_headers if 'h0' in header]
    if h0_headers:
        unique_values_counts = df[h0_headers].nunique()
        max_unique_value_count = unique_values_counts.max()
    else:
        max_unique_value_count = 0

    # Calculate number of hits based on the maximum unique value count
    # Ensuring an equal number of positive and negative values for Faiss search
    number_hit = max_unique_value_count * 2 if max_unique_value_count !=0 else 1 

    return sorted_headers, number_hit,df


def perform_search(index, dataset, indices,train_indices,number_hit=1):
    print(f"Faiss Searching (Number of hits={number_hit}) ....")
    D_p, I_p = index.search(dataset,number_hit)
    result_df = pd.DataFrame({
        0: np.repeat(indices, number_hit), #  target index
        1: np.array(train_indices)[I_p.flatten()],   # query index
        2: D_p.flatten()  # distance 
    })
    return result_df




def main(batch_size, max_len, output, scorpio_model, db_fasta, db_embedding, cal_kmer_freq, val_fasta, metadata, val_embedding, num_distance):

    model_name = f"{scorpio_model}"
    weights_p =os.path.join(model_name, 'checkpoint.pt')
    
    raw_embedding_train,raw_embedding_test,train_indices,test_indices  = load_data(max_len,db_fasta,val_fasta,cal_kmer_freq)

    np.save(os.path.join(output, 'train_indices.npy'), train_indices)

    if db_embedding :
        raw_embedding_train= np.load(db_embedding)
        raw_embedding_test= np.load(val_embedding)
        from_embedding = True
        embedding_size=raw_embedding_train.shape[1]
    else:    
        from_embedding = True if cal_kmer_freq else False
        embedding_size=len(raw_embedding_train[0])
        
        
    model = load_model(weights_p,from_embedding,embedding_size)

    raw_embedding_train = torch.as_tensor(raw_embedding_train)
    raw_embedding_test = torch.as_tensor(raw_embedding_test)

    print("..........")
    
    triplet_embedding_test, triplet_embedding_train = compute_embeddings(model, raw_embedding_test, raw_embedding_train,output,batch_size)
    
    
    index=create_index(triplet_embedding_train,output)
    
    if torch.cuda.device_count() > 1:
        index = faiss.index_cpu_to_all_gpus(index)
    
    print("Indexing Done.")
    
    hierarchy,number_hit,metadata = extract_and_sort_headers(metadata)

    result_df = perform_search(index, triplet_embedding_test, test_indices,train_indices,int(number_hit))
    
    level_to_check=hierarchy
    print(hierarchy)
    
    confidence_score(metadata, level_to_check,hierarchy, result_df,output, num_distance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scorpio Create Db Command Line Program")
    parser.add_argument("--scorpio_model", type=str, help="Path to the Scorpio model file. This argument is required.", required=True)
    parser.add_argument("--output", type=str, help="Directory to save the output database files. This argument is required.", required=True)
    parser.add_argument("--db_fasta", type=str, help="Path to the input FASTA file for the database.", required=True)
    parser.add_argument("--val_fasta", type=str, help="Path to the validation FASTA file.", required=True)
    parser.add_argument("--metadata", type=str, help="Path to the metadata file containing additional information about the sequences. Default is an empty string.",required=True)
    parser.add_argument("--max_len", type=int, help="Maximum allowed length of sequences. Sequences longer than this will be truncated.")
    parser.add_argument("--batch_size", type=int, help="Number of sequences to process in a single batch. Default is 60.", default=60)
    parser.add_argument("--db_embedding", type=str, help="Path to the embedding file for the database sequences. Default is an empty string.", default="")
    parser.add_argument("--val_embedding", type=str, help="Path to the embedding file for the validation sequences. Default is an empty string.", default="")
    parser.add_argument('--cal_kmer_freq', type=str2bool, default=False, help="Boolean flag to indicate whether to calculate k-mer frequency. Default is False.")
    parser.add_argument('--num_distance', type=int, default=2000, help="Number of distances to be calculated. Default is 2000.")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    save_args_to_yaml(args, f'{args.output}/params.yaml')

    main(
        args.batch_size, args.max_len, args.output, args.scorpio_model, args.db_fasta, 
        args.db_embedding, args.cal_kmer_freq, args.val_fasta, args.metadata, args.val_embedding, args.num_distance
    )