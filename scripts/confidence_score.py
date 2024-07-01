import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, r2_score
from joblib import Parallel, delayed
from tqdm import tqdm

# Ensure reproducibility
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the neural network
class ConfNN(nn.Module):
    def __init__(self):
        super(ConfNN, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Function to train a PyTorch model and evaluate its performance
def train_pytorch_model(X, y, num_epochs=130, learning_rate=0.01):
    set_random_seed(42)
    X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    model = ConfNN()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        predictions = model(X_tensor)
        loss = loss_fn(predictions, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()
    r2 = r2_score(y, y_pred)
    return model, r2, y_pred

# Function to calculate F1 score
def calculate_f1_score(distance_value, level, df_mms):
    filtered_df = df_mms[df_mms[2] <= distance_value]
    score = f1_score(filtered_df[f'{level}_query'], filtered_df[f'{level}_target'], average='micro')
    return score

def train_distance_confidence(df,hierarchy,level,num_distance=4000) :

    
    # Find the index of the given level
    level_index = hierarchy.index(level)

    filtered_df = df
    
    # Filter rows where all upper levels are correct
    for i in range(level_index):
        upper_level = hierarchy[i]
        filtered_df = filtered_df[filtered_df[f'{upper_level}_query'] == filtered_df[f'{upper_level}_target']]
    

    df = filtered_df
    # df_mms = df_mms.groupby(0).nth(5)
    # Calculate the minimum and maximum values of df_mms[2]
    min_distance = df[2].min()
    max_distance = df[2].max()
    
    # Generate evenly spaced distance values within the range
    distance_values = np.linspace(min_distance, max_distance, num_distance)

    metrics_values = []
    num_filtered_values = []
    
    # Number of CPU cores for parallel processing
    num_cores = 12
    
    # Parallel execution of the loop
    metrics_values = Parallel(n_jobs=num_cores)(
        delayed(calculate_f1_score)(distance_value, level, df)
        for distance_value in tqdm(distance_values)
    )
    X = np.array(distance_values[1:])  # Ensure X is a 2D array
    X = X.reshape(-1, 1) 
    y =metrics_values[1:]
        
    model, r2, y_pred = train_pytorch_model(X, y)

    return model,X,y
 

# Function to make predictions with a trained model and evaluate R^2
def test_distance_confidence(model, new_X):
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        new_y_pred = model(new_X_tensor).numpy()
    return new_y_pred

# Main execution function
def confidence_score(metadata, levels_to_check,hierarchy, df_mms,output, num_distance=4000):
    
    # df_mms = pd.read_csv(filename, sep="\t", header=None)

    def check_match(index, level, metadata):
        if str(metadata.loc[index, level]).isspace():
            return None
        return metadata.loc[index, level]

    for level in levels_to_check:
        df_mms[level + '_query'] = df_mms.apply(lambda row: check_match(row[0], level, metadata), axis=1)
        df_mms[level + '_target'] = df_mms.apply(lambda row: check_match(row[1], level, metadata), axis=1)

    df_mms.dropna(subset=[level + '_query' for level in levels_to_check] + [level + '_target' for level in levels_to_check], inplace=True)

    model_list = []
    for level in levels_to_check:
        model, X, y = train_distance_confidence(df_mms,hierarchy, level, num_distance)
        model_list.append(model)

    for i, model in enumerate(model_list):
        torch.save(model.state_dict(), f"{output}/confidence_{levels_to_check[i]}.pth")

# # Example usage
# if __name__ == "__main__":
#     metadata_file = "../../../datasets/cds_gene_basic_dataset/process_data/meta_data_all.csv"
#     levels_to_check = ["gene", "phylum"]
#     hierarchy = ['gene', 'phylum', 'class', 'order', 'family']
#     # filename = "./confidence_score_data/faiss_00_epoch_numclass_5_gene_256_btchsize_prev_loss_batch_har_true_triplet_val_data-full-gene-6mer-freq_4000.tsv"
#     num_distance = 2000
#     confidence_score(metadata_file, levels_to_check,hierarchy, df_mms,output, num_distance)
