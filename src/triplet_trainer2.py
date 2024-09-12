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
from pytorch_lightning import Trainer
from torch.cuda.amp import GradScaler, autocast
import warnings
import torch.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import logging
from pytorch_lightning.strategies import DDPStrategy
import datetime
from multiprocessing import Manager
import pickle


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
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
    parser.add_argument('--num_epochs', type=int, default=150, help="Number of Epochs")
    parser.add_argument('--n_classes', type=int, default=5, help="Number of Levels")
    parser.add_argument('--pre_trained_model', type=str, default="", help="Path to your LLM model or Huggingface model")
    parser.add_argument('--embedding_size', type=int, default=4096, help="Size of the embedding vectors")
    parser.add_argument('--from_embedding', type=str2bool, required=True, help='Use Pre-computed embedding')
    args = parser.parse_args()
    return args

def initialize_tokenizer():
    # Initialize the tokenizer once and pass it around
    return KmerTokenizer(kmerlen=6, overlapping=True, maxlen=4096)

def process_record(record, tokenizer):
    """Process a single record."""
    header_parts = record.id.split('|')
    if len(header_parts) > 1:
        seq_id = header_parts[1]
        try:
            embedding = tokenizer.kmer_tokenize(record.seq)
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
    """Yield batches from an iterator."""
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def read_and_tokenize(file_path, id2embedding, batch_size=500, n_jobs=32):
    """Read and tokenize records in parallel."""
    tokenizer = initialize_tokenizer()
    records = SeqIO.parse(file_path, "fasta")
    
    logging.info(f"Processing records in batches from {file_path}...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch, tokenizer)
        for batch in batch_iterator(records, batch_size)
    )
    
    for local_dict in results:
        id2embedding.update(local_dict)


def create_sorted_embedding_array(id2embedding, embedding_size):

    # Convert id2embedding keys to integers and find the maximum index
    int_indices = np.array([int(k) for k in id2embedding.keys()])
    max_index = np.max(int_indices)
    
    embedding_array = np.zeros((max_index + 1, embedding_size), dtype=np.float32)

    # Populate the array with the embeddings
    for idx in int_indices:
        embedding_array[idx] = id2embedding[str(idx)]

    return embedding_array



class TripletLightningModel(pl.LightningModule):
    def __init__(self, args, data_module):
        super(TripletLightningModel, self).__init__()
        self.args = args
        self.model = Tuner(pretrained_model=args.pre_trained_model, embedding_size=args.embedding_size, from_embedding=args.from_embedding)
        self.criterion = TripletLoss(exclude_easy=False, batch_hard=True, margin=0.6, n_classes=args.n_classes)
        self.monitor = init_monitor()
        self.data_module=data_module

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, Y, sim = batch
        X, Y = X.to(self.device), Y.to(self.device) 
        with autocast():
            anchor, pos, neg = self.model(X)
            loss = self.criterion(anchor, pos, neg, Y, sim, self.monitor,self.current_epoch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=False)

        return loss
            

    def validation_step(self, batch, batch_idx):
        # self.log('val_loss', 0, on_epoch=True, prog_bar=False)
        return 0

    
    def validation_epoch_end(self, outputs):
        # Perform custom validation tasks
        if self.data_module.val20:
            start = time.time()
            # acc_dict, err_dict = testing(self.model, self.data_module.val20)  # Assuming these are dictionaries
            validation_time = time.time() - start
            # Log each accuracy and error value separately
            # print("Each Task ,acc across levels :",[(f'val_acc_{key}', float(acc_dict[key][-1])) for key in acc_dict.keys()])  
            self.log('val_time', validation_time, prog_bar=True, sync_dist=True)

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate, amsgrad=True)
        
        
####### need to fix later
class DummyDataset(Dataset):
    def __len__(self):
        return 1  # Just one batch

    def __getitem__(self, idx):
        return 0, 1  # Dummy data, adjust as needed


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
        self.embedding_dict_path = self.data_dir / "embeddings2.npy"
        
    def prepare_data(self):
        # pass
        if self.args.from_embedding:
            pass

        else:
            print("Tokenizing.....")
            read_and_tokenize(self.data_dir / "val.fasta", self.id2embedding)
            read_and_tokenize(self.data_dir / "train.fasta", self.id2embedding)
            ###### 6mer model
            print("Saving the id2embedding array")
            np.save(self.embedding_dict_path, create_sorted_embedding_array(self.id2embedding, 4096))
            self.id2embedding = dict()
            

    def setup(self, stage=None):
        
        self.id2embedding = LazyEmbeddingDict(self.data_dir / "embeddings2.npy")
        self.datasplitter = DataSplitter(self.data_dir, self.id2embedding, self.args.n_classes)
        self.train_splits, self.val, self.val_lookup = self.datasplitter.get_predef_splits()
        self.train_dataset = CustomDataset(self.train_splits, self.datasplitter, self.args.n_classes)
        # self.val20 = Eval(self.val_lookup, self.val, self.datasplitter, self.args.n_classes)


    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, self.args.batch_size)

    def val_dataloader(self):
        # Return a DataLoader with a dummy dataset
        dummy_dataset = DummyDataset()
        return DataLoader(dummy_dataset, batch_size=1)

    def _create_dataloader(self, dataset, batch_size):
        my_collator = MyCollator()
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,  # Shuffling is generally desirable for training
                          drop_last=True,
                          collate_fn=my_collator,
                          pin_memory=True,
                          num_workers=10  # Adjust based on your system's capabilities
                          )

def main():
    args = parse_arguments()

   # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',  # Metric to monitor
        dirpath=args.output,  # Directory to save checkpoints
        filename='model-{epoch:02d}-{step:02d}-{train_loss:.2f}',  # Filename format
        save_top_k=1,  # Save only the best model
        mode='min',  # Mode to minimize the monitored metric (e.g., 'min' for loss)
        every_n_train_steps=5000  # Save checkpoint every 500 training steps
    )

    
    trainer = Trainer(
        max_epochs=args.num_epochs,
        strategy="dp",  # Switch from ddp_spawn to ddp
        precision=16,
        accelerator='auto',  # Use GPU if available, otherwise fallback to CPU
        devices=4,  # Automatically select the number of devices
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback])
    
    data_module = TripletDataModule(args)
        
    # Initialize the model, passing the val20 object
    model = TripletLightningModel(args,data_module)
    
    print("Training Started...")
    
    trainer.fit(model, data_module)
    
    # After training, get the best checkpoint path
    best_checkpoint_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint saved at: {best_checkpoint_path}")
    
    if trainer.is_global_zero:
        torch.save(model, best_checkpoint_path)
        

if __name__ == '__main__':
    main()
