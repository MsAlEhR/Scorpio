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
import pandas as pd
import os
import faiss
import argparse
from TripletModel import AveragePool1dAlongAxis,Tuner
import re
import itertools 
from collections import defaultdict, Counter
from itertools import product
from confidence_score import confidence_score,ConfNN
import yaml
from joblib import Parallel, delayed

from createdb import (
    read_fasta,
    read_fastq,
    kmer_tokenize,
    read_fasta_or_fastq,
    calculate_kmer_frequencies,
    load_model,
    # compute_embeddings,
    create_index,
    str2bool,
    extract_and_sort_headers,
    perform_search,
)



def load_args_from_yaml(yaml_file):
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)
    return {}



def load_data(max_len,test_fasta,cal_kmer_freq):

    print("Loading data ....")

    data_test, test_indices = read_fasta_or_fastq(test_fasta)

    if cal_kmer_freq :
        data_test=calculate_kmer_frequencies(data_test)
    else:
        data_test=kmer_tokenize(data_test)

    return data_test,test_indices



# def compute_embeddings(model, raw_embedding_test,output,batch_size):
#     print("Generating embeddings ....")
#     # Process test set
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     if isinstance(model, torch.nn.DataParallel):
#         model = model.module
    
    
#     model=model.to(device)
#     model.eval()
#     with torch.no_grad(): 
#         triplet_embedding_test = []
#         for i in tqdm(range(0, len(raw_embedding_test), batch_size)):
#             batch = raw_embedding_test[i:i + batch_size].to(device)
#             batch = model.single_pass(batch)
#             triplet_embedding_test.append(batch)
#         triplet_embedding_test = torch.cat(triplet_embedding_test, dim=0).squeeze().to("cpu").detach().numpy()
#         np.save(f"{output}/test_embeddings.npy", triplet_embedding_test)

    
#     return triplet_embedding_test











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

def compute_embeddings(model, raw_embedding_test,output,batch_size):
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

    return triplet_embedding_test









def load_models_from_folder(folder_path, model_class):
    model_dict = {}
    # Regular expression to match files starting with 'confidence' and ending with '.pth'
    pattern = re.compile(r'^confidence_.*\.pth$')

    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            # Extract the confidence level from the filename
            level = filename[11:-4]
            model_path = os.path.join(folder_path, filename)
            
            # Load the model
            model = model_class()  # Initialize the model class
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set the model to evaluation mode
            
            # Store the model in the dictionary
            model_dict[level] = model
            print(f"Model for level {level} loaded successfully.")
    
    return model_dict


def test_distance_confidence(model, new_X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32).reshape(-1, 1).to(device)
    
    with torch.no_grad():
        new_y_pred = model(new_X_tensor).cpu().numpy()  # Move the result back to CPU for numpy conversion
    
    return new_y_pred



def check_match(index, level, metadata):
    if str(metadata.loc[index, level]).isspace():
        return None
    return metadata.loc[index, level]



def calculate_confidence(x, levels):
    probabilities = {}
    
    for i, level in enumerate(levels):
        if i > 0:  # For levels beyond the first (e.g., Phylum, Class, etc.)
            previous_level = levels[i-1]
            previous_class = probabilities[f"{previous_level}_predicted"]
            
            # Filter rows based on the previous level's selected class
            x_filtered = x[x[previous_level + '_target'] == previous_class]
        else:
            # No filtering for the first level (e.g., Kingdom)
            x_filtered = x
            
        # Get all unique classes in the target column for this level
        unique_classes = np.unique(x_filtered[level + '_target'])
        
        # Create a dictionary to store probabilities for each class
        class_probabilities = {}
        total_in_class = len(x_filtered)
        
        for cls in unique_classes:
            matches = (x_filtered[level + '_target'] == cls)
            probability = matches.sum() / total_in_class
            probability_dist = np.max(x_filtered[matches][f'{level}_dist'])
            
            if probability_dist > probability:
                class_probabilities[f'probability_class_{cls}'] = probability * probability_dist
            else:
                class_probabilities[f'probability_class_{cls}'] = probability_dist
        
        # If no classes remain after filtering, use the class with the highest confidence score
        if not class_probabilities:
            class_probabilities = {f'probability_class_{cls}': probability_dist for cls in unique_classes}
        
        max_class = max(class_probabilities, key=class_probabilities.get)
        max_probability = class_probabilities[max_class]
        
        # Store results
        probabilities[f"{level}_predicted"] = max_class.replace("probability_class_", "")
        probabilities[f"{level}_confidence_score"] = max_probability
        probabilities[f"{level}_mean_dist_threshold"] = np.mean(x_filtered[f'{level}_dist'])
        
    return pd.Series(probabilities)



def parallel_group_apply(group,hierarchy):
    return calculate_confidence(group, levels=hierarchy)


def main(db_path, scorpio_model, output, test_fasta, max_len, batch_size, test_embedding,  cal_kmer_freq, number_hit,metadata):

    model_name = f"{scorpio_model}"
    weights_p = model_name+"/checkpoint.pt"

    # Loading Index
    index = faiss.read_index(os.path.join(db_path, 'faiss_index'))
    
    if torch.cuda.device_count() > 1:
        index = faiss.index_cpu_to_all_gpus(index)
        
    print("Index File Loaded")
    
    os.makedirs(output, exist_ok=True)
    
    raw_embedding_test,test_indices  = load_data(max_len,test_fasta,cal_kmer_freq)

    train_indices = np.load(os.path.join(db_path, 'train_indices.npy'))
    
    if test_embedding!="" :
        raw_embedding_test= np.load(test_embedding)
        from_embedding = True
        embedding_size=raw_embedding_test.shape[1]
    else:    
        from_embedding = True if cal_kmer_freq else False
        embedding_size=len(raw_embedding_test[0])
        
        
    model = load_model(weights_p,from_embedding,embedding_size)

    raw_embedding_test = torch.tensor(raw_embedding_test).to("cpu")

    
    triplet_embedding_test = compute_embeddings(model, raw_embedding_test,output,batch_size)

    

    result_df = perform_search(index, triplet_embedding_test, test_indices,train_indices,int(number_hit))
    


    model_dict = load_models_from_folder(db_path, ConfNN)

    
    
    for level,model in tqdm(model_dict.items()):
        result_df[f"{level}_dist"]=test_distance_confidence(model, result_df[2].values)

    
    result_df.to_csv(f"{output}/Output.tsv",sep="\t",index=False)
    

    hierarchy,_,metadata = extract_and_sort_headers(metadata)


    
    # Prepare a dictionary to store results temporarily
    target_results = {level: [] for level in hierarchy}
    
    # Iterate over the rows once and store the results
    for _, row in tqdm(result_df.iterrows()):
        for level in hierarchy:
            target_results[level].append(check_match(row[1], level, metadata))
    
    # Assign the stored results to the DataFrame
    for level in hierarchy:
        result_df[level + '_target'] = target_results[level]


    
    result_df.dropna(subset=[level + '_target' for level in hierarchy], inplace=True)


    print("Generating Prediction....")
    # Group by the first column (group_key)
    grouped = result_df.groupby(0)
    
    # probabilities = grouped.apply(calculate_confidence, levels=hierarchy).reset_index()
    
    results = Parallel(n_jobs=16)(delayed(parallel_group_apply)(group,hierarchy) for name, group in tqdm(grouped))

    # Combine the results into a single DataFrame
    probabilities = pd.DataFrame(results)

    probabilities.to_csv(f"{output}/Prediction.tsv",sep="\t",index=False)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scorpio Inference Command Line Program")
    
    parser.add_argument("--db_path", type=str, required=True, help="Path to the database.")
    parser.add_argument("--number_hit", type=int, required=True, default=1, help="Number of hits to be retrieved. Default is 1.")
    parser.add_argument("--test_fasta", type=str, required=True, help="Path to the test FASTA file.")
    parser.add_argument("--test_embedding", type=str, default="", help="Path to the test embedding file. Default is an empty string.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--scorpio_model", type=str, help="Path to the Scorpio model file.")
    parser.add_argument("--max_len", type=int, help="Maximum allowed length of sequences. Sequences longer than this will be truncated.")
    parser.add_argument("--batch_size", type=int, help="Number of sequences to process in a single batch.")
    parser.add_argument("--cal_kmer_freq", type=str2bool, help="Boolean flag to indicate whether to calculate k-mer frequency.")


    args = parser.parse_args()

    # Load parameters from the YAML file
    config_params = load_args_from_yaml(f"{args.db_path}/params.yaml")

    # Override config parameters with command line arguments if provided
    for key, value in vars(args).items():
        if value is not None:
            config_params[key] = value


    # Pass the final parameters to your main function
    main(
        config_params['db_path'], config_params.get('scorpio_model'), config_params.get('output'), 
        config_params.get('test_fasta'), 
        config_params.get('max_len'), config_params.get('batch_size'),
        config_params.get('test_embedding'), 
        config_params.get('cal_kmer_freq'), config_params.get('number_hit'),config_params.get('metadata')
    )