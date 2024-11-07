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
from utils_prev import *
from TripletModel import Tuner
import argparse
from KmerTokenizer import KmerTokenizer
from Bio import SeqIO
from torch.cuda.amp import GradScaler, autocast
from joblib import Parallel, delayed
import logging
import subprocess


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
    parser.add_argument('--motif_freq', type=str2bool, required=True, help='Use motif Frequency')
    parser.add_argument('--max_len', type=int, default=1024, help='The Sequence max Length(Tokenizer)')
    parser.add_argument('--order_levels', nargs='+', default=[1, 2, 3, 4, 5, 6], type=int, help='A list of order H-levels')
    parser.add_argument('--exclude_easy', type=str2bool, default=False, help='Exclude easy examples')
    parser.add_argument('--batch_hard', type=str2bool, default=False, help='Use hard negative sampling in batches')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')
    
    args = parser.parse_args()
    return args

plt.switch_backend('agg')
plt.clf()

def get_available_gpus(min_free_memory_gb=70):
    available_gpus = []
    try:
        # Convert the required memory to MB
        min_free_memory_mb = min_free_memory_gb * 1024

        # Run nvidia-smi and parse the output
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Process output to find GPUs with enough free memory
        free_memories = result.stdout.strip().split("\n")
        for i, free_memory in enumerate(free_memories):
            if int(free_memory) >= min_free_memory_mb:
                available_gpus.append(i)
                
    except subprocess.CalledProcessError as e:
        print("Error querying GPU memory:", e)
    
    return available_gpus



  # Adjust this as needed
if torch.cuda.is_available():
    available_gpus = get_available_gpus(min_free_memory_gb=30)
    selected_gpu = available_gpus[0] 
    device = torch.device(f'cuda:{selected_gpu}')
else :
    device = torch.device('cpu')
    print("No GPU with sufficient memory found. Using CPU!!!!!!")


def process_record(record, tokenizer):
    """Process a single record."""
    header_parts = record.id.split('|')
    if len(header_parts) > 1:
        seq_id = header_parts[1]
        try:
            embedding = tokenizer.kmer_tokenize(record.seq.upper())
            return seq_id, np.array(embedding, dtype=np.float16)
        except Exception as e:
            logging.error(f"Error processing record {seq_id}: {e}")
            return None
    return None

def process_batch(records, tokenizer):
    """Process a batch of records."""
    local_dict = {}
    for record in records:
        result = process_record(record, tokenizer)
        if result:
            seq_id, embedding = result
            local_dict[seq_id] = embedding
    return local_dict

def batch_iterator(iterator, batch_size):
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def read_and_tokenize(file_path,max_len, id2embedding,kmerlen=6, batch_size=500, n_jobs=32):
    
    tokenizer = KmerTokenizer(kmerlen=kmerlen, overlapping=True, maxlen=max_len)
    records = SeqIO.parse(file_path, "fasta")
    
    logging.info(f"Processing records in batches from {file_path}...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch, tokenizer)
        for batch in batch_iterator(records, batch_size)
    )
    
    for local_dict in results:
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
    motif_freq = args.motif_freq

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

    if motif_freq:
        embedding_p = os.path.join(data_dir, "embeddings.npy")
        embedding_list = np.load(embedding_p, mmap_mode='r')
        id2embedding = {str(seq_id): embedding_list[int(seq_id)][np.newaxis, :] for seq_id in ids}
        print("Loading dataset from: {}".format(embedding_p))
    else:
        print("Tokenizing.....")
        
        # Initialize the shared dictionary
        id2embedding = dict()
                
        start_time = time.time()
        read_and_tokenize(data_dir / "val.fasta",args.max_len, id2embedding)
        val_time = time.time() - start_time
        print(f"Time to process validation set: {val_time:.6f} seconds")
        
        start_time = time.time()
        read_and_tokenize(data_dir / "train.fasta",args.max_len, id2embedding)
        train_time = time.time() - start_time
        print(f"Time to process training set: {train_time:.6f} seconds")
    
        print(f"Number of embeddings: {len(id2embedding)}")  



    experiment_dir = log_dir / exp_name

    if not experiment_dir.is_dir():
        print("Creating new log-directory: {}".format(experiment_dir))
        experiment_dir.mkdir(parents=True)

    n_bad = 3  # counter for number of epochs that did not improve (counter for early stopping)
    n_thresh = num_epochs  # threshold for number of epochs that did not improve (threshold for early stopping)
    batch_hard = args.batch_hard  # whether to activate batch_hard sampling (recommended)
    exclude_easy = args.exclude_easy  # whether to exclude trivial samples (did not improve performance)
    margin = args.margin  # set this to a float to activate threshold-dependent loss functions (see TripletLoss)
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

    model = Tuner(pretrained_model=pre_trained_model, embedding_size=embedding_size, motif_freq=motif_freq)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)
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

        acc, err = testing(model, val20,batch_size)
            
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
