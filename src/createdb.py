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




def read_fasta(file_path):
    x_header = []
    x_seq = []
    current_sequence = []

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        current_header = None
        indx = 0 
        for xxx, line in enumerate(lines):
            line = line.strip()
            if line.startswith(">"):
                indx = indx+1
                # Save the previous sequence if it exists
                if current_sequence:
                    x_seq.append("".join(current_sequence))
                    current_sequence = []
                
                # Regular expression pattern to extract the index string
                pattern = r'seqid\|(\d+)'
                current_header_match = re.search(pattern, line)
                if current_header_match is not None:
                    index_string = current_header_match.group(1)  # Extract the matched string
                    current_header = int(index_string)
                    x_header.append(current_header)  # Convert to integer and append to x_header
                else:
                    current_header = indx
                    x_header.append(indx)
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



def kmer_tokenize(seq_list, kmerlen=6, overlapping=True, maxlen=81):
    
    VOCAB = [''.join(i) for i in itertools.product(*(['ATCG'] * int(kmerlen)))]
    VOCAB_SIZE = len(VOCAB) + 5  

    tokendict = dict(zip(VOCAB, range(5,VOCAB_SIZE)))
    tokendict['[UNK]'] = 0
    tokendict['[SEP]'] = 1
    tokendict['[CLS]'] = 3
    tokendict['[MASK]'] = 4
    
    tokendict['[PAD]'] = 4

    seq_ind_list = []
    for seq in tqdm(seq_list):
        if overlapping:
            stoprange = len(seq) - (kmerlen - 1)  
            tokenlist = [tokendict[seq[k:k + kmerlen]] for k in range(0, stoprange) if set(seq[k:k + kmerlen]).issubset('ATCG')]
        else:
            stoprange = len(seq) - (kmerlen - 1)
            tokenlist = [tokendict[seq[k:k + kmerlen]] for k in range(0, stoprange, kmerlen) if set(seq[k:k + kmerlen]).issubset('ATCG')]
        # Padding if necessary
        if len(tokenlist) < maxlen:
            tokenlist.extend([tokendict['[PAD]']] * (maxlen - len(tokenlist)))
        seq_ind_list.append(tokenlist[:maxlen])
    return seq_ind_list
    



def read_fasta_or_fastq(file_path):
    # Decide whether the file is FASTA or FASTQ based on its extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() in [".fq", ".fastq"]:  # Common extensions for FASTQ
        return read_fastq(file_path)
    elif ext.lower() in [".fa", ".fasta"]:  # Common extensions for FASTA
        return read_fasta(file_path)
    else:
        raise ValueError("Unrecognized file extension. Please provide a FASTA or FASTQ file.")

def calculate_kmer_frequencies(sequences,k=6):

    possible_kmers = [''.join(p) for p in product('ATCG', repeat=k)]  # All possible 6-mers

    # Initialize an array to store the frequencies
    frequencies = np.zeros((len(sequences), len(possible_kmers)))

    # Calculate k-mer frequencies for each sequence
    for i, sequence in tqdm(enumerate(sequences)):
        sequence_kmer_counts = Counter([sequence[j:j+k] for j in range(len(sequence) - k + 1)])
        for j, kmer in enumerate(possible_kmers):
            frequencies[i, j] = sequence_kmer_counts[kmer]

    return frequencies







def load_data(max_len,db_fasta,test_fasta,cal_kmer_freq):

    print("Loading data ....")


    data_test, test_indices = read_fasta_or_fastq(test_fasta)
    data_train, train_indices = read_fasta_or_fastq(db_fasta)

    if cal_kmer_freq :
        data_train=calculate_kmer_frequencies(data_train)
        data_test=calculate_kmer_frequencies(data_test)
    else:
        data_train=kmer_tokenize(data_train,maxlen=max_len) 
        data_test=kmer_tokenize(data_test,maxlen=max_len)



    return data_train,data_test,train_indices,test_indices





def load_model(weights_p,from_embedding,embedding_size):
    print("Loading Model ....")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state = torch.load(weights_p, map_location=torch.device(device))['model_state_dict']
    model = torch.load(weights_p, map_location=torch.device(device))["Tuner"]
    model.from_embedding=from_embedding
    model.embedding_size=embedding_size
    model.load_state_dict(state)
    model = model.eval()
    return model







def compute_embeddings(model, raw_embedding_test, raw_embedding_train,output,batch_size):
    print("Generating embeddings ....")
    
    # Process test set
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model=model.to(device)
    model.eval()
    with torch.no_grad(): 
        triplet_embedding_test = []
        for i in tqdm(range(0, len(raw_embedding_test), batch_size)):
            batch = raw_embedding_test[i:i + batch_size].to(device)
            batch = model.single_pass(batch)
            triplet_embedding_test.append(batch)
        triplet_embedding_test = torch.cat(triplet_embedding_test, dim=0).squeeze().to("cpu").detach().numpy()
        np.save(f"{output}/val_embeddings.npy", triplet_embedding_test)
        
        print(triplet_embedding_test.shape,"lenght of embeddings")
        # Process test set
        triplet_embedding_train = []
        for i in tqdm(range(0, len(raw_embedding_train), batch_size)):
            batch = raw_embedding_train[i:i + batch_size].to(device)
            batch = model.single_pass(batch)
            triplet_embedding_train.append(batch)
        triplet_embedding_train = torch.cat(triplet_embedding_train, dim=0).squeeze().to("cpu").detach().numpy()
        np.save(f"{output}/db_embeddings.npy", triplet_embedding_train)

    
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
    number_hit = max_unique_value_count * 2

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

    raw_embedding_train = torch.tensor(raw_embedding_train).to("cpu")
    raw_embedding_test = torch.tensor(raw_embedding_test).to("cpu")

    
    triplet_embedding_test, triplet_embedding_train = compute_embeddings(model, raw_embedding_test, raw_embedding_train,output,batch_size)
    
    
    index=create_index(triplet_embedding_train,output)

    
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