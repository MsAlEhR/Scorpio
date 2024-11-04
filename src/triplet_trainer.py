"""
Author: Saleh Refahi
Email: sr3622@drexel.edu
"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
from pathlib import Path
import argparse
from KmerTokenizer import KmerTokenizer
from Bio import SeqIO
from joblib import Parallel, delayed
from utils import *
from TripletModel import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from torch.cuda.amp import GradScaler, autocast
import warnings
import torch.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import logging
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
from multiprocessing import Manager
import pickle
from pathlib import Path
import csv
import yaml

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
    parser.add_argument('--output', type=str, help="Output Directory (Model Output)")
    parser.add_argument('--input', type=str, help="Input Directory (Data)")
    parser.add_argument('--exp_name', type=str, help="Experiment Name", default="Triplet_results")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
    parser.add_argument('--num_epochs', type=int, default=150, help="Number of Epochs")
    parser.add_argument('--n_classes', type=int, default=5, help="Number of Levels")
    parser.add_argument('--pre_trained_model', type=str, default="", help="Path to your LLM model or Huggingface model")
    parser.add_argument('--motif_freq', type=str2bool, required=True, help='Use motif Frequency')
    parser.add_argument('--max_len', type=int, default=1024, help='The Sequence max Length(Tokenizer)')
    parser.add_argument('--order_levels', nargs='+', default=[1, 2, 3, 4, 5, 6], type=int, help='A list of order H-levels')
    parser.add_argument('--exclude_easy', type=str2bool, default=False, help='Exclude easy examples')
    parser.add_argument('--batch_hard', type=str2bool, default=False, help='Use hard negative sampling in batches')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')

    
    args = parser.parse_args()
    return args



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


def create_sorted_embedding_array(id2embedding, max_len):

    # Convert id2embedding keys to integers and find the maximum index
    int_indices = np.array([int(k) for k in id2embedding.keys()])
    max_index = np.max(int_indices)
    
    embedding_array = np.zeros((max_index + 1, max_len), dtype=np.float16)

    # Populate the array with the embeddings
    for idx in int_indices:
        embedding_array[idx] = id2embedding[str(idx)]

    return embedding_array




class TripletLightningModel(pl.LightningModule):
    def __init__(self, args, data_module):
        super(TripletLightningModel, self).__init__()
        self.args = args
        self.model = Tuner(pretrained_model=args.pre_trained_model, motif_freq=args.motif_freq)
        self.criterion = TripletLoss(exclude_easy=args.exclude_easy, batch_hard=args.batch_hard, margin=args.margin, n_classes=args.n_classes)
        self.monitor = init_monitor()
        self.data_module=data_module

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, Y, sim = batch
        X, Y = X.to(self.device), Y.to(self.device) 
        anchor, pos, neg = self.model(X)
        loss = self.criterion(anchor, pos, neg, Y, sim, self.monitor,self.current_epoch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss
            

    def validation_step(self, batch, batch_idx):
        # Ensure no gradient computation
        with torch.no_grad():
            X, Y, sim = batch
            X, Y = X.to(self.device), Y.to(self.device)
            
            # Forward pass
            anchor, pos, neg = self.model(X)
            loss = self.criterion(anchor, pos, neg, Y, sim, self.monitor, self.current_epoch)
            
            # Calculate and log metrics
            dist_ap = torch.norm(anchor - pos, p=2, dim=1)  # Distance for positive samples
            dist_an = torch.norm(anchor - neg, p=2, dim=1)  # Distance for negative samples
            
            embeddings = torch.cat([anchor, pos, neg], dim=0)
            self.monitor['pos'].append(dist_ap.mean().cpu().item())
            self.monitor['neg'].append(dist_an.mean().cpu().item())
            self.monitor['min'].append(embeddings.min(dim=1)[0].mean().cpu().item())
            self.monitor['max'].append(embeddings.max(dim=1)[0].mean().cpu().item())
            self.monitor['mean'].append(embeddings.mean(dim=1).mean().cpu().item())
            self.monitor['loss'].append(loss.cpu().item())
            self.monitor['norm'].append(torch.norm(embeddings, p='fro').cpu().item())
            
        return loss  # Return the loss for logging

    
    def on_validation_epoch_end(self):
        with open(self.args.monitor_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                len(self.monitor['loss']),
                (sum(self.monitor['loss'])) / len(self.monitor['loss']),
                (sum(self.monitor['norm'])) / len(self.monitor['norm']) ,
                (sum(self.monitor['pos'])) / len(self.monitor['pos']),
                (sum(self.monitor['neg'])) / len(self.monitor['neg']),
                (sum(self.monitor['min'])) / len(self.monitor['min']),
                (sum(self.monitor['max'])) / len(self.monitor['max']),
                (sum(self.monitor['mean'])) / len(self.monitor['mean']) ,
            ])
        self.log('val_loss',  sum(self.monitor['loss']) / len(self.monitor['loss']) , on_epoch=True, prog_bar=True)            
        self.monitor = init_monitor()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate, amsgrad=True)
        
        # Define the scheduler
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3),
            'monitor': 'val_loss',  # Monitors 'val_loss' to decide on reducing the lr
        }
        
        
        return [optimizer], [scheduler]        





class LazyEmbeddingDict:
    def __init__(self, embedding_file):
        # Load the .npy file as a memory-mapped array (no need to specify shape)
        self.id2embedding = np.load(embedding_file, mmap_mode='r')

    def __getitem__(self, key):
        # Access the specific embedding based on the provided ID (key)
        embedding_np = self.id2embedding[int(key)][np.newaxis, :]  # Add new axis to match shape
        # Convert to PyTorch tensor and return
        return torch.tensor(embedding_np)


class TripletDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = Path(args.input)
        self.id2embedding = dict() 
        self.val20 = None
        self.embedding_dict_path = self.data_dir / "encodings.npy"
        
    def prepare_data(self):
        # pass
        reorder(self.data_dir,self.args.order_levels)
        print("Tokenizing.....")
        read_and_tokenize(self.data_dir / "val.fasta",self.args.max_len, self.id2embedding)
        read_and_tokenize(self.data_dir / "train.fasta",self.args.max_len, self.id2embedding)
        print("Saving the id2embedding array",len(self.id2embedding))
        np.save(self.embedding_dict_path, create_sorted_embedding_array(self.id2embedding, self.args.max_len))
        self.id2embedding = dict()
        
        

    def setup(self, stage=None):
        
        self.id2embedding = LazyEmbeddingDict(self.embedding_dict_path)
        self.datasplitter = DataSplitter(self.data_dir, self.id2embedding, self.args.n_classes)
        self.train_splits, self.val, self.val_lookup = self.datasplitter.get_predef_splits()
        self.train_dataset = CustomDataset(self.train_splits, self.datasplitter, self.args.n_classes)
        self.train_dataset.get_example()
        self.test_dataset = CustomDataset(self.val, self.datasplitter, self.args.n_classes)
        # self.val20 = Eval(self.val_lookup, self.val, self.datasplitter, self.args.n_classes)


    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, self.args.batch_size)

    def val_dataloader(self):
        return self._create_dataloader(self.test_dataset, self.args.batch_size)



    def _create_dataloader(self, dataset, batch_size):
        my_collator = MyCollator()
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,  # Shuffling is generally desirable for training
                          drop_last=False,
                          collate_fn=my_collator,
                          pin_memory=True,
                          num_workers=10  # Adjust based on your system's capabilities
                          )

def save_args_to_yaml(args, file_path):
    # Convert Namespace to a dictionary
    args_dict = vars(args)

    # Save dictionary to YAML
    with open(file_path, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)

def main():
    args = parse_arguments()

    args.monitor_file = str(Path(args.output) / "monitor.csv")
    
    os.makedirs(args.output, exist_ok=True)

    save_args_to_yaml(args, Path(args.output) / "train_args.yaml")
    
    
    seed_all(seed=83)
    seed_everything(83)
    
    with open(args.monitor_file, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([ 'number','loss', 'norm', 'pos', 'neg', 'min', 'max', 'mean'])    

   # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',  # Metric to monitor
        dirpath=args.output,  # Directory to save checkpoints
        filename='model-{epoch:02d}-{step:02d}-{train_loss:.3f}',  # Filename format
        save_top_k=1,  # Save only the best model
        mode='min',  # Mode to minimize the monitored metric (e.g., 'min' for loss)
        every_n_train_steps=1000  # Save checkpoint every 500 training steps
    )

    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        strategy = 'dp' if num_gpus > 1 else None
    else:
        num_gpus = 0
        strategy = None  # Use CPU if no GPU is available
    
    # Set devices based on the available hardware
    devices = num_gpus if num_gpus > 0 else 1  # 1 for CPU
    devices=[1,2,3]
    trainer = Trainer(
        max_epochs=args.num_epochs,
        # val_check_interval=0.1,
        strategy=strategy,  # Switch between dp and None
        precision=16 if num_gpus > 0 else 32,  # 16-bit precision for GPUs, 32-bit for CPU
        accelerator='gpu' if num_gpus > 0 else 'cpu',  # Automatically use GPU or CPU
        devices=devices,
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback]
    )
    
    data_module = TripletDataModule(args)
        
    # Initialize the model, passing the val20 object
    model = TripletLightningModel(args,data_module)
    
    print("Training Started...")

    saver = Saver(Path(args.output))
    
    
    trainer.fit(model, data_module)
    
    # After training, get the best checkpoint path
    best_checkpoint_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint saved at: {best_checkpoint_path}")
    
    if trainer.is_global_zero:
        saver.save_checkpoint(model, args.num_epochs, {},Path(args.output)/'checkpoint.pt')
        

if __name__ == '__main__':
    main()
