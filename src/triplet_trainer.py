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
from utils import *
from TripletModel import Tuner
import argparse
from KmerTokenizer import KmerTokenizer
from Bio import SeqIO
from torch.cuda.amp import GradScaler, autocast
from joblib import Parallel, delayed


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help="Output Directory")
    parser.add_argument('--input', type=str, help="Input Directory")
    parser.add_argument('--exp_name', type=str, help="Experiment Name", default="Triplet_results")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
    parser.add_argument('--num_epochs', type=int, default=150, help="Number of Epochs")
    parser.add_argument('--n_classes', type=int, default=5, help="Number of Levels")
    parser.add_argument('--pre_trained_model', type=str, default="", help="Path to your LLM model or Huggingface model")
    parser.add_argument('--embedding_size', type=int, default=4096, help="Size of the embedding vectors")
    parser.add_argument('--from_embedding', type=str2bool, required=True, help='Use Pre-computed embedding')
    args = parser.parse_args()
    return args

plt.switch_backend('agg')
plt.clf()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def initialize_tokenizer():
    global tokenizer
    tokenizer = KmerTokenizer(kmerlen=6, overlapping=True, maxlen=4096)

def process_record(record):
    header_parts = record.id.split('|')
    if len(header_parts) > 1:
        seq_id = header_parts[1]
        try:
            embedding = tokenizer.kmer_tokenize([record.seq])
            return seq_id, embedding
        except Exception as e:
            print(f"Error processing record {seq_id}: {e}")
            return None
    return None

def process_batch(records):
    local_dict = {}
    try:
        results = [process_record(record) for record in records]
        for tokenized_result in results:
            if tokenized_result:
                seq_id, embedding = tokenized_result
                local_dict[seq_id] = embedding
    except Exception as e:
        print(f"Error in process_batch: {e}")
    return local_dict

def read_and_tokenize(file_path, id2embedding, batch_size=2000, n_jobs=8):
    initialize_tokenizer()  
    # Read all records into a list
    records = list(SeqIO.parse(file_path, "fasta"))
    print(f"Total records read from {file_path}: {len(records)}")

    # Process in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(records[i:i + batch_size]) 
        for i in tqdm(range(0, len(records), batch_size))
    )

    # Update the main dictionary with the results
    for local_dict in results:
        if local_dict:
            id2embedding.update(local_dict)



def trainer():
    args = parse_arguments()

    # Access the parameters
    output_path = args.output
    input_path = args.input
    exp_name = args.exp_name
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    n_classes = args.n_classes
    pre_trained_model = args.pre_trained_model
    embedding_size = args.embedding_size
    from_embedding = args.from_embedding

    # Hyperparameters
    start_overall = time.time()
    SEED = 83
    seed_all(SEED)

    data_dir = Path(input_path)
    log_dir = Path(output_path) / 'trainer'

    #reording the levels
    # order = [6,7,1,2,3,4,5]
    # reorder(data_dir,order)

    ids = pd.read_csv(os.path.join(data_dir, 'hierarchical-level.txt'), sep="\t", header=None)[0].tolist()

    if from_embedding:
        embedding_p = os.path.join(data_dir, "embeddings.npy")
        embedding_list = np.load(embedding_p, mmap_mode='r')
        id2embedding = {str(seq_id): embedding_list[int(seq_id)][np.newaxis, :] for seq_id in ids}
        print("Loading dataset from: {}".format(embedding_p))
    else:
        print("Tokenizing.....")
        
        # Initialize the shared dictionary
        id2embedding = dict()
        
        
        start_time = time.time()
        read_and_tokenize(data_dir / "val.fasta", id2embedding)
        val_time = time.time() - start_time
        print(f"Time to process validation set: {val_time:.6f} seconds")
        
        start_time = time.time()
        read_and_tokenize(data_dir / "train.fasta", id2embedding)
        train_time = time.time() - start_time
        print(f"Time to process training set: {train_time:.6f} seconds")
    
        print(f"Number of embeddings: {len(id2embedding)}")  



    experiment_dir = log_dir / exp_name

    if not experiment_dir.is_dir():
        print("Creating new log-directory: {}".format(experiment_dir))
        experiment_dir.mkdir(parents=True)

    n_bad = 3  # counter for number of epochs that did not improve (counter for early stopping)
    n_thresh = num_epochs  # threshold for number of epochs that did not improve (threshold for early stopping)
    batch_hard = True  # whether to activate batch_hard sampling (recommended)
    exclude_easy = False  # whether to exclude trivial samples (did not improve performance)
    margin = 0.6  # set this to a float to activate threshold-dependent loss functions (see TripletLoss)
    monitor_plot = True

    # Initialize plotting class (used to monitor loss etc during training)
    pltr = plotter(experiment_dir)

    # Prepare datasets
    datasplitter = DataSplitter(data_dir, id2embedding, n_classes)
    train_splits, val, val_lookup = datasplitter.get_predef_splits()

    val20 = Eval(val_lookup, val, datasplitter, n_classes)

    train = CustomDataset(train_splits, datasplitter, n_classes)
    train_loader = dataloader(train, batch_size)

    # Get the size of the dataset
    dataset_size = len(train_loader.dataset)
    print("Dataset size:", dataset_size)

    model = Tuner(pretrained_model=pre_trained_model, embedding_size=embedding_size, from_embedding=from_embedding)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    criterion = TripletLoss(exclude_easy=exclude_easy, batch_hard=batch_hard, margin=margin, n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    scaler = GradScaler()

    saver = Saver(experiment_dir, n_classes)
    saver.save_checkpoint(model, 0, optimizer, experiment_dir / 'checkpoint.pt')
    baseline_acc, baseline_err = get_baseline(val20, n_classes)

    print('###### Training parameters ######')
    print('Experiment name: {}'.format(experiment_dir))
    print('LR: {}, BS: {}, free Paras.: {}, n_epochs: {}'.format(learning_rate, batch_size, count_parameters(model), num_epochs))
    print('#############################\n')
    print('Start training now!')

    monitor = init_monitor()
    for epoch in tqdm(range(num_epochs)):  # for each epoch

        # =================== testing =====================
        start = time.time()
        if isinstance(model, nn.DataParallel):
            acc, err = testing(model.module, val20)  # Access original model methods using .module
        else:
            acc, err = testing(model, val20)
            
        test_time = time.time() - start
        
        new_best = saver.check_performance(acc, model, epoch, optimizer)  # early stopping class

        if new_best is None:  # if the new accuracy was worse than a previous one
            n_bad += 1
            if n_bad >= n_thresh:  # if more than n_bad consecutive epochs were worse, break training
                break
        else:  # if the new accuracy is larger than the previous best one by epsilon, reset counter
            n_bad = 0

        # =================== training =====================
        # monitor epoch-wise performance
        epoch_monitor = init_monitor()
        start = time.time()

        for train_idx, (X, Y, sim) in enumerate(train_loader):  # for each batch in the training set
            X, Y = X.to(device), Y.to(device)
        
            if torch.cuda.is_available():
                with autocast():
                    anchor, pos, neg = model(X)
                    loss = criterion(anchor, pos, neg, Y, sim, epoch_monitor, epoch)
        
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                anchor, pos, neg = model(X)
                loss = criterion(anchor, pos, neg, Y, sim, epoch_monitor, epoch)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        train_time = time.time() - start

        # Monitor various metrics during training
        if monitor_plot:
            monitor['loss'].append(sum(epoch_monitor['loss']) / len(epoch_monitor['loss']))
            monitor['norm'].append(sum(epoch_monitor['norm']) / len(epoch_monitor['norm']))
            monitor['pos'].append(sum(epoch_monitor['pos']) / len(epoch_monitor['pos']))
            monitor['neg'].append(sum(epoch_monitor['neg']) / len(epoch_monitor['neg']))
            monitor['min'].append(sum(epoch_monitor['min']) / len(epoch_monitor['min']))
            monitor['max'].append(sum(epoch_monitor['max']) / len(epoch_monitor['max']))
            monitor['mean'].append(sum(epoch_monitor['mean']) / len(epoch_monitor['mean']))

        # Log results and plot
        if epoch % 2 == 0 or epoch == num_epochs - 1:  # draw plots only every fifth epoch
            pltr.plot_acc(acc, baseline_acc, n_classes)
            pltr.plot_distances(monitor['pos'], monitor['neg'])
            pltr.plot_loss(monitor['loss'], file_name='loss.pdf')
            pltr.plot_loss(monitor['norm'], file_name='norm.pdf')
            pltr.plot_minMaxMean(monitor)

        print(('epoch [{}/{}], train loss: {:.3f}, train-time: {:.1f}[s], test-time: {:.1f}[s], ' +
              ', '.join('ACC-{}: {:.2f}'.format(i, acc[i][-1]) for i in range(n_classes)) +
              ' ## Avg. Acc: {:.2f}').format(
            epoch + 1, num_epochs,
            sum(epoch_monitor['loss']) / len(epoch_monitor['loss']),
            train_time, test_time,
            *(acc[i][-1] for i in range(n_classes)),
            sum(acc[i][-1] for i in range(n_classes)) / n_classes
        ))

    end_overall = time.time()
    print(end_overall - start_overall)
    print("Total training time: {:.1f}[m]".format((end_overall - start_overall) / 60))
    return None

if __name__ == '__main__':
    trainer()
