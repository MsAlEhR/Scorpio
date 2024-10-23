"""
Author: Saleh Refahi
Email: sr3622@drexel.edu
"""


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, r2_score
from joblib import Parallel, delayed
from tqdm import tqdm
import yaml

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


def fast_micro_f1(y_true, y_pred):
    # Calculate TP, FP, FN using vectorized operations
    tp = np.sum(y_true == y_pred)
    fp_fn = np.sum(y_true != y_pred)
    
    # Compute precision and recall
    precision = tp / (tp + fp_fn)
    recall = tp / (tp + fp_fn)
    
    # Calculate micro F1 score
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Function to calculate F1 score
def calculate_f1_score(distance_value, df, query_col, target_col):
    mask = df[2].values <= distance_value
    score = fast_micro_f1(query_col[mask], target_col[mask])
    return score, np.sum(mask)


def train_distance_confidence(df,hierarchy,level,num_distance=4000) :

    print(f"Training Confidence Score for {level}")
    # Find the index of the given level
    level_index = hierarchy.index(level)

    
    # Filter rows where all upper levels are correct
    for i in range(level_index):
        upper_level = hierarchy[i]
        df = df[df[f'{upper_level}_query'] ==df[f'{upper_level}_target']]
    



    # # Let's assume df[2] is the column you are sampling from
    # column_values = df[2].values
    
    # # Calculate histogram to identify dominant regions
    # hist, bin_edges = np.histogram(column_values, bins=50)  # Adjust the number of bins if necessary
    
    # # Find the most frequent (dominant) regions
    # dominant_bins = bin_edges  # Taking top 5 dominant bins, adjust if needed
    # print(dominant_bins)
    # # Generate more samples around the dominant bins
    # dominant_samples = []
    # for bin_start, bin_end in zip(dominant_bins[:-1], dominant_bins[1:]):
    #     dominant_samples.extend(np.linspace(bin_start, bin_end, int(num_distance / 5)))  # Denser sampling around dominant regions
    
    # # Generate evenly spaced distance values across the full range
    # min_distance = column_values.min()
    # max_distance = column_values.max()
    # even_samples = np.linspace(min_distance, max_distance, int(num_distance * 0.7))  # Fewer samples in non-dominant regions
    
    # # Combine both dominant and evenly spaced samples
    # distance_values = dominant_samples
    

    # Calculate the minimum and maximum values of df_mms[2]
    min_distance = df[2].min()
    max_distance = df[2].max()
    print(min_distance,max_distance,len(df))
    # Generate evenly spaced distance values within the range
    distance_values = np.linspace(min_distance, max_distance, num_distance)

    metrics_values = []
    num_filtered_values = []
    
    # Number of CPU cores for parallel processing
    num_cores = 14
    
    query_col = df[f'{level}_query'].values
    target_col = df[f'{level}_target'].values
    
    
    # Parallel execution
    values = Parallel(n_jobs=num_cores)(
        delayed(calculate_f1_score)(distance_value, df, query_col, target_col)
        for distance_value in tqdm(distance_values)
    )
    metrics_values, num_filtered_values = zip(*values)
    
    X = np.array(distance_values[1:])  # Ensure X is a 2D array
    X = X.reshape(-1, 1) 
    y =metrics_values[1:]
    model, r2, y_pred = train_pytorch_model(X, y)


    df.loc[:, f"{level}_dist"] = test_distance_confidence(model, df[2].values)

    # Group by the first column (group_key)
    grouped = df.groupby(0)
    
    probabilities= grouped.apply(lambda group: cal_prob(group, level)).reset_index()
    
    threshold = None
    dist_th = None
    
    min_diff = float('inf')
    min_dist = float('inf')
    
    for distance, metric, number in zip(distance_values[1:], metrics_values[1:], num_filtered_values[1:]):
        normalized_number = number / len(num_filtered_values)
        diff = abs(normalized_number - metric)
        if diff < min_diff:
            min_diff = diff
            dist_th = distance
    
    for _, row in probabilities.iterrows():
        diff = abs(dist_th - row[f"{level}_dist"])
        if diff < min_dist:
            min_dist = diff
            threshold = row[f"{level}_confidence_score"]

    
    return model,X,y,threshold
 

# Function to make predictions with a trained model and evaluate R^2
def test_distance_confidence(model, new_X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32).reshape(-1, 1).to(device)
    
    with torch.no_grad():
        new_y_pred = model(new_X_tensor).cpu().numpy()  # Move the result back to CPU for numpy conversion
    
    return new_y_pred



# Define a function to calculate class probabilities for each level in levels , we have similiar function for inference as well with name calculate_confidence
def cal_prob(x, level):
    probabilities = {}
    # Get all unique classes in the target column for this level
    unique_classes = np.unique(x[level + '_target'])
    
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

    probabilities[f"{level}_predicted"] = max_class.replace("probability_class_", "")
    probabilities[f"{level}_confidence_score"] = max_probability
    probabilities[f"{level}_dist"] = np.mean(x[f'{level}_dist'])

    return pd.Series(probabilities)



# Main execution function
def confidence_score(metadata, levels_to_check,hierarchy, df_mms,output, num_distance=4000):
    
    print("Performing Confidence Score function...")
    
    def check_match(index, level, metadata):
        value = str(metadata.loc[index, level])
        if value.isspace():
            return None
        return value
    
    # Prepare a dictionary to store results temporarily
    query_results = {level: [] for level in levels_to_check}
    target_results = {level: [] for level in levels_to_check}
    
    # Iterate over the rows once and store the results
    for _, row in tqdm(df_mms.iterrows()):
        for level in levels_to_check:
            query_results[level].append(check_match(row[0], level, metadata))
            target_results[level].append(check_match(row[1], level, metadata))
    
    # Assign the stored results to the DataFrame
    for level in levels_to_check:
        df_mms[level + '_query'] = query_results[level]
        df_mms[level + '_target'] = target_results[level]
    
    # Drop rows with NaN values
    df_mms.dropna(subset=[level + '_query' for level in levels_to_check] + [level + '_target' for level in levels_to_check], inplace=True)



    with open(f'{output}/params.yaml', 'a') as f:
        for i, level in enumerate(levels_to_check):
            model, X, y, threshold = train_distance_confidence(df_mms, hierarchy, level, num_distance)
            torch.save(model.state_dict(), f"{output}/confidence_{levels_to_check[i]}.pth")
            # Write the new information to the YAML file
            yaml.dump({f'threshold_{levels_to_check[i]}': float(threshold)}, f, default_flow_style=False)



# # Example usage
# if __name__ == "__main__":
#     metadata_file = "../../../datasets/cds_gene_basic_dataset/process_data/meta_data_all.csv"
#     levels_to_check = ["gene", "phylum"]
#     hierarchy = ['gene', 'phylum', 'class', 'order', 'family']
#     # filename = "./confidence_score_data/faiss_00_epoch_numclass_5_gene_256_btchsize_prev_loss_batch_har_true_triplet_val_data-full-gene-6mer-freq_4000.tsv"
#     num_distance = 2000
#     confidence_score(metadata_file, levels_to_check,hierarchy, df_mms,output, num_distance)
