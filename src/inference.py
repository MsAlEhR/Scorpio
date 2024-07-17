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

from createdb import (
    read_fasta,
    read_fastq,
    kmer_tokenize,
    read_fasta_or_fastq,
    calculate_kmer_frequencies,
    load_model,
    compute_embeddings,
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
        data_test=kmer_tokenize(data_test,maxlen=max_len)

    return data_test,test_indices



def compute_embeddings(model, raw_embedding_test,output,batch_size):
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
        np.save(f"{output}/test_embeddings.npy", triplet_embedding_test)

    
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


# Function to make predictions with a trained model and evaluate R^2
def test_distance_confidence(model, new_X):
    # Convert the new data to PyTorch tensors
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32).reshape(-1, 1)
    # Predict using the trained model
    with torch.no_grad():
        new_y_pred = model(new_X_tensor).numpy()


    return new_y_pred



def check_match(index, level, metadata):
    if str(metadata.loc[index, level]).isspace():
        return None
    return metadata.loc[index, level]



# Define a function to calculate class probabilities for each level in levels
def calculate_confidence(x, levels):
    probabilities = {}

    for i,level in enumerate(levels):

        ################ should delete further
        probabilities[level + '_query'] = x[level + '_query'].iloc[0]
        
        # Get all unique classes in the target column for this level
        unique_classes = np.unique(x[level + '_target'])


        ############# need to fix this based on the models !!!!!!!!!!!!!!!!!
        # x[f'{level}_dist']=test_distance_confidence(model_list[i], list(x[2]))
        
        # Create a dictionary to store probabilities for each class
        class_probabilities = {}

        total_in_class = len(x)
        
        # Calculate the probability for each class
        for cls in unique_classes:
            matches = (x[level + '_target'] == cls)
            probability = matches.sum() / total_in_class
            probability_dist = np.max(x[x[level + '_target'] == cls][f'{level}_dist'])
            # class_probabilities[f'probability_class_{cls}'] = ( probability*probability_dist)
            if probability_dist >  probability :
                class_probabilities[f'probability_class_{cls}'] = (probability* probability_dist)
            else:
                class_probabilities[f'probability_class_{cls}'] = (probability_dist)
        # Find the class with the maximum probability
        max_class = max(class_probabilities, key=class_probabilities.get)
        max_probability = class_probabilities[max_class]

        # Store results
        probabilities[f"{level}_predicted"] = max_class.replace("probability_class_", "")
        probabilities[f"{level}_probability"] = max_probability
        
        probabilities[f"{level}_dist"] = np.mean(x[x[level + '_target'] == cls][2])

    return pd.Series(probabilities)






def main(db_path, scorpio_model, output, test_fasta, max_len, batch_size, test_embedding,  cal_kmer_freq, number_hit,metadata):

    model_name = f"{scorpio_model}"
    weights_p = model_name+"/checkpoint.pt"

    # Loading Index
    index = faiss.read_index(os.path.join(db_path, 'faiss_index'))
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
        result_df[f"{level}_dist"]=test_distance_confidence(model, result_df[2])

    
    result_df.to_csv(f"{output}/Output.tsv",sep="\t")
    

    hierarchy,_,metadata = extract_and_sort_headers(metadata)



    for level in tqdm(hierarchy):
        result_df[level + '_query'] = result_df.apply(lambda row: check_match(row[0], level, metadata), axis=1)
        result_df[level + '_target'] = result_df.apply(lambda row: check_match(row[1], level, metadata), axis=1)
    
    
    result_df.dropna(subset=[level + '_query' for level in hierarchy] + [level + '_target' for level in hierarchy], inplace=True)

    # Group by the first column (group_key)
    grouped = result_df.groupby(0)
    
    probabilities = grouped.apply(calculate_confidence, levels=hierarchy).reset_index()
    


    probabilities.to_csv(f"{output}/Prediction.tsv",sep="\t")

    

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