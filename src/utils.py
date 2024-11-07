"""
Author: Saleh Refahi
Email: sr3622@drexel.edu
Date: 2024-06-26

Some functions in this script are cloned and rewritten from the EAT repository.
https://github.com/Rostlab/EAT
"""



import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd

import matplotlib.pyplot as plt
import colorsys  
from matplotlib.colors import to_hex

import itertools  
import numpy as np
import time
import random
import copy
from tqdm import tqdm
from Bio import SeqIO
import faiss
from concurrent.futures import ThreadPoolExecutor


plt.switch_backend('agg')  # GPU is only available via SSH (no display)
plt.clf()  # clear previous figures if already existing

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
    available_gpus = get_available_gpus(min_free_memory_gb=10)
   
    selected_gpu = available_gpus[0] 
    device = torch.device(f'cuda:{selected_gpu}')
else :
    device = torch.device('cpu')
    print("No GPU with sufficient memory found. Using CPU!!!!!!")

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, train, datasplitter, n_classes, balanced_sampling=False):
        self.balanced_sampling = balanced_sampling
        self.seq_id=train

        self.id2label, self.label2id = datasplitter.parse_label_mapping(
            set(train))

        # if classes should be sampled evenly (not all training samples are used in every epoch)
        if self.balanced_sampling:
            pass
        else:  # if you want to iterate over all training samples
            self.data_len = len(self.seq_id)

        self.id2embedding = datasplitter.id2embedding
        self.n_classes = n_classes  # number of class levels
        
        self.neg_sim = 1 

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.balanced_sampling:  # get a dataset class, instead of a trainings sample
           pass
        else:  # get a training sample (over-samples large dataset families according to occurance)
            anchor = self.id2embedding[index] # get embedding of anchor
            anchor_id = self.seq_id[index] # get dataset ID of anchor
            anchor_label = self.id2label[anchor_id] # get dataset label of anchor
        pos, neg, pos_label, neg_label, pos_sim = self.get_pair(
            anchor_id, anchor_label)
        self.neg_sim = self.neg_sim +1 
        return (anchor, pos, neg, anchor_label, pos_label, neg_label, pos_sim)

    def get_unique_labels(self):
        unique_set = set()
        unique_labels = list()
        for _, cath_label in self.id2label.items():
            cath_str = '.'.join([str(cath_hierarchy_lvl)
                                for cath_hierarchy_lvl in cath_label])
            if cath_str in unique_set:
                continue
            unique_labels.append(cath_label)
            unique_set.add(cath_str)
        print("Number of unique dataset labels in train: {}".format(len(unique_set)))
        return unique_labels

    def get_rnd_label(self, labels, is_pos, existing_label=None):
        
        n_labels = len(labels)
        # if alternative labels are available, ensure difference between existing and new label
        if n_labels > 1 and existing_label is not None:
            labels = [label for label in labels if label != existing_label]
            n_labels -= 1

        rnd_idx = np.random.randint(0, n_labels)

        i = iter(labels)
        for _ in range(rnd_idx):
            next(i)
        rnd_label = next(i)
        # do not accidentaly draw the same label; instead draw again if necessary
        if existing_label is not None and rnd_label == existing_label:
            if is_pos:  # return the label itself for positives
                # Allow positives to have the same class as the anchor (relevant for rare classes)
                return existing_label
            else:
                # if there exists no negative sample for a certain combination of anchor and similarity-level
                return None
        return rnd_label
################################################################################
    
    def get_rnd_candidates(self, anchor_label, similarity_level,n_classes,is_pos):
        
        anchor_label2 = copy.deepcopy(anchor_label)
        candidates =self.label2id
        
        for i in range(n_classes):
            
            if is_pos and self.n_classes ==1 :  
                candidates = candidates[anchor_label2[0]]
                break
            
            if i> similarity_level:
                anchor_label2[i]= self.get_rnd_label(
                    candidates.keys(), is_pos)
            elif i == similarity_level :
                class_value = anchor_label2[i]
                anchor_label2[i]= self.get_rnd_label(
                    candidates.keys(), is_pos ,class_value)             
            candidates=candidates[anchor_label2[i]]
            
        # if type(candidates) != list :    
        #     candidates=list(candidates.keys())
             
        return candidates
    
###########################################################################################    
    

    def check_triplet(self, anchor_label, pos_label, neg_label, neg_hardness, pos_hardness):
        
        if self.n_classes !=1 :
            assert neg_hardness < pos_hardness, print(
                "Neg sample more similar than pos sample")
            
        for i in range(0, pos_hardness):
            assert anchor_label[i] == pos_label[i], print("Pos label not overlapping:\n" +
                                                         "Diff: {}\nanchor:{}\npos:{}\nneg:{}".format(pos_hardness, anchor_label, pos_label, neg_label))
        for j in range(0, neg_hardness):
            assert anchor_label[j] == neg_label[j], print("Neg label not overlapping:\n" +
                                                         "Diff: {}\nanchor:{}\npos:{}\nneg:{}".format(neg_hardness, anchor_label, pos_label, neg_label))
        assert anchor_label[neg_hardness] != neg_label[neg_hardness], print(
            "Neg label not different from anchor")
        return None
    
    def save_to_file(self, data):
        with open("log.txt", 'a') as f:
            f.write(f"{data}\n")    

    def get_pair(self,  anchor_id, anchor_label, hardness_level=None, verbose=False):
        pos, neg = None, None
        pos_label, neg_label = None, None
        # result_list = ["batch"]
        i =0 
        while pos is None or neg is None:
            i=i+1
            
            neg_similarity = np.random.randint(self.n_classes)
            # I changed this line because if number of classes is one i want to both have same level
            pos_similarity = neg_similarity + 1  if self.n_classes !=1 else neg_similarity
            try:
                neg_candidates = self.get_rnd_candidates(
                    anchor_label, neg_similarity,self.n_classes, is_pos=False) # get set of negative candidates
                
                neg_id = random.choice(neg_candidates) # randomly pick one of the neg. candidates
                ########################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
             
                neg_label = self.id2label[neg_id] # get label of randomly picked neg.
                neg = self.id2embedding[neg_id] # get embedding of randomly picked neg.
                
                # repeat the same for the positive sample
                pos_candidates = self.get_rnd_candidates(
                    anchor_label, pos_similarity,self.n_classes, is_pos=True)
                
                ########################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pos_id = random.choice(pos_candidates)
                # pos_id = pos_candidates[int(int(anchor_id)%len(pos_candidates))]
                
                # ensure that we do not randomly pick the same protein as anchor and positive
                if pos_id == anchor_id and len(pos_candidates) > 1:
                    while pos_id == anchor_id: # re-draw from the pos. candidates if possible
                        pos_id = random.choice(pos_candidates)
                # if there is only one protein in a superfamily (anchor==positive without other candidates), re-start picking process
                   ###################################### i added this line because super family I dont have such thing sometimes!   
                elif pos_id == anchor_id and len(pos_candidates) == 1 and i >100:
                    pass
                elif pos_id == anchor_id and len(pos_candidates) == 1:
                    # print("Fffffffffffffffffffuuuuuuuuuuuuu",i)
                    continue

                pos = self.id2embedding[pos_id]
                pos_label = self.id2label[pos_id]
                # if we successfully picked anchor, positive and negative candidates, do same sanity checks
                if pos_label is not None and neg_label is not None:
                    self.check_triplet(anchor_label, pos_label,
                                       neg_label, neg_similarity, pos_similarity)
                else: # if no triplet could be formed for a given combination of similarities/classes
                    continue

            except NotImplementedError: #  if you try to create triplets for a class level that is not yet implemented in get_rnd_candidates
                raise NotImplementedError

            except KeyError:
                # if get_rnd_label returned None because no negative could be found
                # for a certain combination of anchor protein and similarity-lvl
                # re-start picking process
                continue

        if verbose:
            print('#### Example ####')
            print('Anc ({}) label: {}'.format(anchor_id, anchor_label))
            print('Pos ({}) label: {}'.format(pos_id, self.id2label[pos_id]))
            print('Neg ({}) label: {}'.format(neg_id, self.id2label[neg_id]))
            print('#### Example ####')
            
        # self.save_to_file(result_list) 
        
        
        return pos, neg, pos_label, neg_label, pos_similarity

    def get_example(self):
        example_id= random.choice(list(self.id2label.keys()))
        print(type(example_id), example_id) 
        example_label = self.id2label[example_id]
        self.get_pair(example_id, example_label, verbose=True)
        example_id= random.choice(list(self.id2label.keys()))
        example_label = self.id2label[example_id]
        self.get_pair(example_id, example_label, verbose=True)   
        self.get_pair(example_id, example_label, verbose=True)
        return None
    
  
class DataSplitter():
    def __init__(self,data_dir, id2embedding,n_classes=4, verbose=True,num_workers=12):
        self.verbose = verbose
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.cath_label_path = self.data_dir / 'hierarchical-level.txt'
        self.id2embedding = id2embedding
        self.num_workers = num_workers
        # self.id2label, self.label2id = self.parse_label_mapping()

    def get_id2embedding(self):
        return self.id2embedding

    def parse_label_mapping(self,id_subset):
        id2label = dict()
        label2id = dict()
        with open(self.cath_label_path, 'r') as f:
            for n_domains, line in enumerate(f):
                # skip header lines
                if line.startswith("#"):
                    continue
                data = line.split()
                identifier = int(data[0])
                if identifier not in id_subset:
                    continue

                def insert_into_dict(dct, keys, value):
                    current_dict = dct
                    for index, key in enumerate(keys[:-1]):
                        current_dict = current_dict.setdefault(int(key), {})
                        
                    current_dict.setdefault(int(keys[-1]), []).append(value)
                        
                #####################################################
                id2label[identifier] = [int(i) for i in data[1:self.n_classes+1]]

                insert_into_dict(label2id, data[1:self.n_classes+1], identifier)
                #########################################################################


        if self.verbose:
            print('Finished parsing n_domains: {}'.format(n_domains))
            print("Total length of id2label: {}".format(len(id2label)))
            
        return id2label, label2id


    def get_embeddings(self, fasta_path):
        ids = self.read_ids(fasta_path)
        return ids


    
    def read_ids(self, path):
        with open(path, 'r') as f:
            records = list(SeqIO.parse(f, "fasta"))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            id_list = list(executor.map(self.extract_id, records))
        
        return id_list
    
    def extract_id(self, record):
        if '|' in record.id:
            return int(record.id.split('|')[1])
        return None

    def get_predef_splits(self, p_train=None, p_test=None):

        if p_train is None or p_test is None:
            p_train = self.data_dir / "train.fasta"
            p_val = self.data_dir / "val.fasta"
            
        train = self.get_embeddings(p_train)   
        val = self.get_embeddings(p_val)
        
        

        # Determine the number of samples for the validation lookup table
        num_samples = int(len(train) * 0.1)

        # Randomly sample the training keys for the validation lookup table
        val_lookup = random.sample(train, num_samples)
        
        # val_lookup = {key: train[key] for key in sampled_keys}
                
        # valLookup20 = train
        
        if self.verbose:
            print('##########')
            print('Finished splitting data!')
            print('Train set size: {}'.format(len(train)))
            print('Val set size: {}'.format(len(val)))
            print('ValLookup20 size: {}'.format(len(val_lookup)))
            print('##########')
        return train, val,val_lookup

    
    
class MyCollator(object):
    def __call__(self, batch):
        X = list()
        Y = list()
        sim = list()

        for (anchor, pos, neg, anchor_label, pos_label, neg_label, pos_sim) in batch:

            x = torch.cat([anchor, pos, neg], dim=0)
            ################################ I changed these lines for cnn
            emb_shape= anchor.shape[-1]

            if anchor.dim()> 2: 
              X.append(x.view(1, 3, -1,emb_shape))
            else :
              X.append(x.view(1, 3, -1))
            ################################################
            Y.append(self.get_label_vector(anchor_label, pos_label, neg_label))
            sim.append(pos_sim)
        return (torch.cat(X, dim=0), torch.cat(Y, dim=0), torch.tensor(sim))
            
    def get_label_vector(self, anchor_label, pos_label, neg_label):
        anc = torch.tensor(anchor_label).view(1,-1)
        pos = torch.tensor(pos_label).view(1,-1)
        neg = torch.tensor(neg_label).view(1,-1)
        y = torch.cat([anc, pos, neg], dim=0)
        return y.view(1, 3, -1)


class plotter():
    def __init__(self, log_dir):
        self.init_plotting()
        self.log_dir = log_dir

    def init_plotting(self):
        params = {
            'axes.labelsize': 13,  # increase font size for axis labels
        }
        plt.rc(params)  # apply parameters
        return plt, None

    def merge_pdfs(self, pdf1_path, pdf2_path, output_path):
        # Merge two PDFs
        from PyPDF2 import PdfFileMerger
        pdfs = [pdf1_path, pdf2_path]

        merger = PdfFileMerger()

        for pdf in pdfs:
            merger.append(pdf)

        merger.write(str(output_path))
        merger.close()
        return None

    def plot_minMaxMean(self, train_minMax, file_name='minMaxMean.pdf'):
        plt, _ = self.init_plotting()

        # Plot first three samples in Batch in one figure
        fig, axes = plt.subplots(1, 1)

        x = np.asarray(train_minMax['min'])
        y = np.asarray(train_minMax['max'])
        z = np.asarray(train_minMax['mean'])
        L = np.arange(1, x.size+1)

        axes.plot(L, x, 'g', label='Min')
        axes.plot(L, y, 'r', label='Max')
        axes.plot(L, z, 'b', label='Mean')

        axes.set_xlabel('Steps/Batches')
        axes.set_ylabel('min/max/mean')

        _ = plt.legend()
        plt.title('Min/Max/Mean development')

        pdf_path = self.log_dir / file_name
        fig.savefig(str(pdf_path), format='pdf')

        plt.close(fig)  # close figure handle
        return None

    def plot_distances(self, dist_pos, dist_neg, file_name='distances.pdf'):
        plt, _ = self.init_plotting()

        # Plot first three samples in Batch in one figure
        fig, axes = plt.subplots(1, 1)

        x = np.asarray(dist_pos)
        y = np.asarray(dist_neg)
        L = np.arange(1, x.size+1)

        axes.plot(L, x, 'g',  label='Dist. Pos')
        axes.plot(L, y, 'r', label='Dist. Neg')

        axes.set_xlabel('Steps/Batches')
        axes.set_ylabel('Distances')

        _ = plt.legend()
        plt.title('Distance development')

        pdf_path = self.log_dir / file_name
        fig.savefig(str(pdf_path), format='pdf')

        plt.close(fig)  # close figure handle
        return None

    def plot_acc(self, acc, baseline, diff_classes=4, file_name='acc.pdf'):

        plt, _ = self.init_plotting()


        fig, axes = plt.subplots(1, 1)
        

        cmap = plt.get_cmap('viridis')  # You can choose a different colormap here
        norm = plt.Normalize(0, diff_classes)
        colors = [to_hex(cmap(norm(i))) for i in range(diff_classes)]
        for diff_class in range(diff_classes):
            x = np.asarray(acc[diff_class])
            max_acc_idx = np.argmax(x)
            max_acc = x[max_acc_idx]
            L = np.arange(1, x.size+1)
            b = np.ones(x.size) * baseline[diff_class]
            axes.plot(L, x, colors[diff_class],  label='LvL.: {} # {:.3f} in epoch {}'.format(
                diff_class, max_acc, max_acc_idx))
            axes.plot(L, b, linestyle='-.', color=colors[diff_class])

        axes.set_xlabel('Epochs')
        axes.set_ylabel('Accuracy')

        _ = plt.legend()
        plt.title(file_name.replace('.pdf', ''))

        pdf_path = self.log_dir / file_name
        fig.savefig(str(pdf_path), format='pdf')

        plt.close(fig)  # close figure handle
        return None

    def plot_loss(self, train, test=None, file_name='loss.pdf'):
        test = train if test is None else test
        plt, _ = self.init_plotting()
        fig, axes = plt.subplots(1, 1)

        x = np.asarray(train)
        y = np.asarray(test)
        L = np.arange(1, x.size+1)

        axes.plot(L, x, 'g',  label='Train')
        axes.plot(L, y, 'r--', label='Test')

        axes.set_xlabel('Epochs')
        axes.set_ylabel('Loss')

        _ = plt.legend()
        plt.title(file_name.replace('loss.pdf', ''))

        pdf_path = self.log_dir / file_name
        fig.savefig(str(pdf_path), format='pdf')

        plt.close(fig)  # close figure handle
        return None


class Eval():
    def __init__(self, lookup, test, datasplitter, n_classes, name='Eval'):
        # self.lookup, self.lookupIdx2label = self.preproc(lookup)
        self.lookupIdx2label = self.preproc(lookup)
        self.testIdx2label = self.preproc(test)
        self.test = torch.tensor(np.array(list(test.values()))).squeeze(dim=1)
        self.lookup = torch.tensor(np.array(list(lookup.values()))).squeeze(dim=1)      
        self.id2label, self.label2id = datasplitter.parse_label_mapping(
            # use only keys from the given lookup set
            set(lookup) | set(test),
        )
        self.name = name
        self.n_classes = n_classes
        self.accs = self.init_log()
        self.errs = self.init_log()


    def get_test_set(self):
        return self.test

    def get_lookup_set(self):
        return self.lookup

    def get_acc(self):
        return self.acc

    def get_err(self):
        return self.err

    def init_log(self):
        log = dict()
        for i in range(self.n_classes):
            log[i] = list()
        return log

    def init_confmats(self):
        confmats = list()
        for i in range(self.n_classes):
            confmat = np.zeros((1, 2, 2))
            confmats.append(confmat)
        confmats = np.concatenate(confmats, axis=0)
        return confmats 

    def preproc(self, data):
        idx2label = dict()
        for idx, (seq_id, embd) in enumerate(data.items()):
            idx2label[idx] = seq_id
        # dataset = torch.cat(dataset, dim=0)
        return idx2label

    def add_sample(self, y, yhat, confmats):
        for class_lvl, true_class in enumerate(y):
            if np.isnan(true_class):
                continue
            if true_class == yhat[class_lvl]:
                correct = 1
            else:
                correct = 0
            confmats[class_lvl, correct, correct] += 1
        return confmats

    def mergeTopK(self, yhats):
        yhats = np.vstack(yhats)
        final_yhat = [None for i in range(self.n_classes)]
        for i in range(self.n_classes):
            (values, counts) = np.unique(yhats[:, i], return_counts=True)
            idxs = np.argmax(counts)
            nn_class = values[idxs]
            final_yhat[i] = nn_class
            mask = yhats[:, i] == nn_class
            yhats = yhats[mask, :]
        return final_yhat

    def compute_err(self, confmat, n_bootstrap=10000):
        n_total = int(confmat.sum())
        n_wrong, n_correct = int(confmat[0, 0]), int(confmat[1, 1])
        preds = [0 for _ in range(n_wrong)] + [1 for _ in range(n_correct)]
        subset_accs = list()
        for _ in range(n_bootstrap):
            rnd_subset = random.choices(preds, k=n_total)
            subset_accs.append(sum(rnd_subset) / n_total)
        return np.std(np.array(subset_accs), axis=0, ddof=1)

    def evaluate(self, lookup, queries, n_nearest=1, update=True):
        self.index = faiss.IndexFlatL2(lookup.shape[1])
        self.index.add(lookup)
        
        distances, nn_idxs = self.index.search(queries, n_nearest)
        
        confmats = self.init_confmats()
        
        n_test = len(self.testIdx2label)
        for test_idx in range(n_test):
            y_id = self.testIdx2label[test_idx]
            y = copy.deepcopy(self.id2label[y_id])

            nn_idx = nn_idxs[test_idx]
            yhats = []
            for nn_i in nn_idx:
                nn_i = int(nn_i)
                yhat_id = self.lookupIdx2label[nn_i]
                yhat = self.id2label[yhat_id]
                yhat = np.asarray(yhat)
                yhats.append(yhat)
                    
            if n_nearest == 1:
                assert len(yhats) == 1, print("More than one NN retrieved, though, n_nearest=1!")
                yhat = yhats[0]
            else:
                yhat = self.mergeTopK(yhats)
                
            confmats = self.add_sample(y, yhat, confmats)

        if update:
            for i in range(self.n_classes):
                acc = confmats[i, 1, 1] / confmats[i].sum()
                err = self.compute_err(confmats[i])
                self.accs[i].append(acc)
                self.errs[i].append(err)
            return self.accs, self.errs
        else:
            accs, errs = list(), list()
            for i in range(self.n_classes):
                acc = confmats[i, 1, 1] / confmats[i].sum()
                err = self.compute_err(confmats[i])
                accs.append(acc)
                errs.append(err)
                print(f"Samples for class {i}: {sum(confmats[i, :, :])}")
            return accs, errs        





class Saver():
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.checkpoint_p = experiment_dir / 'checkpoint.pt'
        self.checkpoint_BEST = experiment_dir / 'checkpoint_BEST.pt'
        self.best_performance = 0
        self.epsilon = 1e-3

    def load_checkpoint(self):
        state = torch.load(self.checkpoint_p)
        model = Tuner().to(device)
        model.load_state_dict(state['state_dict'])
        print('Loaded model from epch: {:.1f} with avg. acc: {:.3f}'.format(
            state['epoch'], self.best_avg_acc))
        return model, state['epoch']

    def save_checkpoint(self, model, epoch, optimizer,dir_p):
        state = {
            'epoch': epoch,
            'best_performance': self.best_performance,
            'Tuner': model.model,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
        }
        
        torch.save(state, dir_p)

        return None

    def check_performance(self, acc, model, epoch, optimizer):
        # print(acc,"ACCCCCCCCCA")
        if isinstance(acc, dict):  # if a list of accuracies is passed
            #accuracy of first class 
            new_performance = acc[0][-1]
        else:  # if a single Silhouette score is passed
            new_performance = acc
            
        self.save_checkpoint(model, epoch, optimizer,self.checkpoint_p)   
        if new_performance > self.best_performance + self.epsilon:
            self.save_checkpoint(model, epoch, optimizer,self.checkpoint_BEST)
            self.best_performance = new_performance
            print('New best performance found:  {:.3f}!'.format(
                self.best_performance))
            return self.best_performance
        return None

    
class TripletLoss(object):


    def __init__(self, margin=None, exclude_easy=False, batch_hard=True,n_classes=4,temperature=3):
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)
        self.exclude_easy = exclude_easy
        self.reduction = 'none' if self.exclude_easy else 'mean'
        self.batch_hard = batch_hard
        self.n_classes = n_classes
        self.sample = False
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=0)
        self.min = -10**10
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(
                margin=margin, reduction=self.reduction)
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction=self.reduction)

    def __call__(self, anchor, pos, neg, Y,sim,monitor,epoch):
        

        
        if self.batch_hard:
            dist_ap, dist_an = self.get_batch_hard(anchor, pos, neg, Y)
        else:
            dist_ap = self.distance(anchor, pos)
            dist_an = self.distance(anchor, neg)
    
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        
        def weighted_hinge_loss(x1, x2, y, margin, weight):
            loss = torch.max(torch.zeros_like(y), -y * (x1 - x2) + margin)
            loss = weight * loss
            return torch.mean(loss)

        
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y )
            
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)            
            


        if self.exclude_easy:
            loss = loss.sum() / (loss < 0).sum()

        # embeddings = torch.cat((anchor, pos, neg))
        # monitor['pos'].append(toCPU(dist_ap.mean()))
        # monitor['neg'].append(toCPU(dist_an.mean()))

        # monitor['min'].append(toCPU(embeddings.min(dim=1)[0].mean()))
        # monitor['max'].append(toCPU(embeddings.max(dim=1)[0].mean()))
        # monitor['mean'].append(toCPU(embeddings.mean(dim=1).mean()))

        # monitor['loss'].append(toCPU(loss))
        # monitor['norm'].append(toCPU(torch.norm(embeddings, p='fro')))

        return loss

    # https://gist.github.com/rwightman/fff86a015efddcba8b3c8008167ea705
    def get_hard_triplets(self, pdist, y, prev_mask_pos):
        n = y.size()[0]
        mask_pos = y.expand(n, n).eq(y.expand(n, n).t()).to(device)

        mask_pos = mask_pos if prev_mask_pos is None else prev_mask_pos * mask_pos

        # every protein that is not a positive is automatically a negative for this lvl
        mask_neg = ~mask_pos
        if device.type !="cpu":
            mask_pos[torch.eye(n).bool().cuda()] = 0  # mask self-interactions
            mask_neg[torch.eye(n).bool().cuda()] = 0
        else : 
            mask_pos[torch.eye(n).bool()] = 0  # mask self-interactions
            mask_neg[torch.eye(n).bool()] = 0

        if self.sample:
            # weighted sample pos and negative to avoid outliers causing collapse
            posw = (pdist + 1e-12) * mask_pos.float()
            posw[posw == 0] = self.min
            posw = self.softmax(posw)
            posi = torch.multinomial(posw, 1)

            dist_ap = pdist.gather(0, posi.view(1, -1))
            # There is likely a much better way of sampling negatives in proportion their difficulty, based on distance
            # this was a quick hack that ended up working better for some datasets than hard negative
            negw = (1 / (pdist + 1e-12)) * mask_neg.float()
            negw[posw == 0] = self.min
            negw = self.softmax(posw)
            negi = torch.multinomial(negw, 1)
            dist_an = pdist.gather(0, negi.view(1, -1))
        else:
            ninf = torch.ones_like(pdist) * float('-inf')
            dist_ap = torch.max(pdist * mask_pos.float(), dim=1)[0]
            nindex = torch.max(torch.where(mask_neg, -pdist, ninf), dim=1)[1]
            dist_an = pdist.gather(0, nindex.unsqueeze(0)).view(-1)

        return dist_ap, dist_an, mask_pos

    def get_batch_hard(self, anchor, pos, neg, Y):
        Y = torch.cat([Y[:, 0, :], Y[:, 1, :], Y[:, 2, :]], dim=0)
        X = torch.cat([anchor, pos, neg], dim=0)
        pdist = self.pdist(X)

        dist_ap, dist_an = list(), list()
        mask_pos = None

        for i in range(self.n_classes):
            y = Y[:, i]

            dist_pos, dist_neg, mask_pos = self.get_hard_triplets(
                pdist, y, mask_pos)
            dist_ap.append(dist_pos.view(-1))
            dist_an.append(dist_neg.view(-1))
        # return False    
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        return dist_ap, dist_an

    def pdist(self, v):
        # Efficient pairwise distance calculation
        dist = torch.cdist(v, v, p=2)
        return dist
    # def pdist(self,v):
    #     """
    #     Efficient pairwise angular (cosine) distance calculation using ArcCos.
        
    #     Args:
    #         v (torch.Tensor): Input tensor of shape (batch_size, embedding_dim)
        
    #     Returns:
    #         dist (torch.Tensor): Pairwise angular distances (cosine distances converted to angles)
    #     """
    #     # Normalize the vectors to unit length
    #     v = F.normalize(v, p=2, dim=1)
        
    #     # Compute cosine similarities
    #     cosine_sim = torch.matmul(v, v.t())
        
    #     # Clamp cosine similarity values to ensure they fall in the valid range for arccos
    #     cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
        
    #     # Calculate angular distances (arc cosine of the similarities)
    #     angular_dist = torch.acos(cosine_sim)
        
    #     return angular_dist
    




# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.5, scale=30.0, n_classes=4, temperature=3, exclude_easy=False, batch_hard=True):
#         """
#         margin: Arc margin for angular distance
#         scale: Scale factor applied to the logits
#         n_classes: Number of classes for each dimension
#         temperature: For softmax scaling
#         exclude_easy: Exclude easy triplets (non-hard)
#         batch_hard: Apply batch-hard triplet mining
#         """
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.scale = scale
#         self.temperature = temperature
#         self.n_classes = n_classes
#         self.exclude_easy = exclude_easy
#         self.batch_hard = batch_hard
#         self.distance = nn.PairwiseDistance(p=2)
#         self.softmax = nn.Softmax(dim=0)
#         self.min_value = -1e10
#         self.ranking_loss = nn.SoftMarginLoss(reduction='mean')

#     def forward(self, anchor, pos, neg, labels,sim,monitor,epoch):
#         # Normalize the embeddings
#         anchor = F.normalize(anchor)
#         pos = F.normalize(pos)
#         neg = F.normalize(neg)

#         # Batch-hard strategy
#         if self.batch_hard:
#             dist_ap, dist_an = self.get_batch_hard(anchor, pos, neg, labels)
#         else:
#             dist_ap = self.distance(anchor, pos)
#             dist_an = self.distance(anchor, neg)

#         # Arc margin addition: apply angular margin to anchor-positive distance
#         theta_ap = torch.acos(torch.clamp(dist_ap, -1.0, 1.0))
#         theta_an = torch.acos(torch.clamp(dist_an, -1.0, 1.0))

#         margin_ap = torch.cos(theta_ap + self.margin)
#         logits = margin_ap * self.scale

#         # Use ranking loss (triplet-like behavior)
#         y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
#         loss = self.ranking_loss(logits, dist_an - dist_ap)

#         # Monitoring and logging
#         embeddings = torch.cat((anchor, pos, neg))
#         monitor['pos'].append(dist_ap.mean().item())
#         monitor['neg'].append(dist_an.mean().item())
#         monitor['loss'].append(loss.item())
#         monitor['min'].append(embeddings.min(dim=1)[0].mean().item())
#         monitor['max'].append(embeddings.max(dim=1)[0].mean().item())
#         monitor['mean'].append(embeddings.mean(dim=1).mean().item())
#         monitor['norm'].append(torch.norm(embeddings, p='fro').item())

#         return loss

#     def get_hard_triplets(self, pdist, y, prev_mask_pos):
#         n = y.size(0)
#         mask_pos = y.expand(n, n).eq(y.expand(n, n).t())
#         mask_pos = mask_pos if prev_mask_pos is None else prev_mask_pos * mask_pos
#         mask_neg = ~mask_pos
#         mask_pos.fill_diagonal_(0)
#         mask_neg.fill_diagonal_(0)

#         dist_ap = torch.max(pdist * mask_pos.float(), dim=1)[0]
#         dist_an = torch.min(pdist * mask_neg.float() + (mask_neg.float() * 1e10), dim=1)[0]

#         return dist_ap, dist_an

#     def get_batch_hard(self, anchor, pos, neg, labels):
#         Y = torch.cat([labels[:, 0], labels[:, 1], labels[:, 2]], dim=0)
#         X = torch.cat([anchor, pos, neg], dim=0)
#         pdist = self.pdist(X)
#         dist_ap, dist_an = [], []

#         mask_pos = None
#         for i in range(self.n_classes):
#             y = Y[:, i]
#             dist_pos, dist_neg = self.get_hard_triplets(pdist, y, mask_pos)
#             dist_ap.append(dist_pos.view(-1))
#             dist_an.append(dist_neg.view(-1))

#         return torch.cat(dist_ap), torch.cat(dist_an)

#     def pdist(self, v):
#         dist = torch.cdist(v, v, p=2)
#         return dist













    
def init_monitor():
    monitor = dict()

    monitor['loss'] = []
    monitor['norm'] = []

    monitor['pos'] = []
    monitor['neg'] = []

    monitor['min'] = []
    monitor['max'] = []
    monitor['mean'] = []
    return monitor


# move torch/GPU tensor to numpy/CPU
def toCPU(data):
    if isinstance(data, torch.Tensor):
        return data.float().cpu().detach().numpy()
    else:
        print(f"Unexpected data type: {type(data)}")
        return data  # Or handle this case accordingly


# count number of free parameters in the network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# # Create dataloaders with custom collate function
# def dataloader(customdata, batch_size):
#     my_collator = MyCollator()
#     return torch.utils.data.DataLoader(dataset=customdata,
#                                        batch_size=batch_size,
#                                        shuffle=True,
#                                        drop_last=True,
#                                        collate_fn=my_collator,
#                                        num_workers=0
#                                        )



def get_baseline(test,n_classes):
    test_set = test.get_test_set()
    train_set = test.get_lookup_set()
    acc, err = test.evaluate(train_set, test_set, update=False)
    for i in range(n_classes):
        print('ACC-baseline-{}: {:.2f} +/- {:.2f}\n'.format(i, acc[i], err[i]))

    return acc, err    


def testing(mdl, test, batch_size=40):
    mdl = mdl.to(device)
    mdl.eval()
    with torch.no_grad():
        test_emb = test.get_test_set()
        lookup_emb = test.get_lookup_set()
        # Process test set
        test_tucker_batches = []
        for i in range(0, len(test_emb), batch_size):
            batch = test_emb[i:i + batch_size]
            test_tucker_batch = mdl.single_pass(batch.to(device))
            test_tucker_batches.append(test_tucker_batch)
        test_tucker = torch.cat(test_tucker_batches, dim=0)
        
        # Process lookup set
        lookup_tucker_batches = []
        ################### 1000 =len(lookup_emb)
        for i in tqdm(range(0, len(lookup_emb), batch_size)):
            batch = lookup_emb[i:i + batch_size]
            lookup_tucker_batch = mdl.single_pass(batch.to(device))
            lookup_tucker_batches.append(lookup_tucker_batch)
        lookup_tucker = torch.cat(lookup_tucker_batches, dim=0)


        # Move tensors back to CPU and remove from GPU memory
    test_emb = test_emb.cpu()
    lookup_emb = lookup_emb.cpu()
    test_tucker = test_tucker.cpu()
    lookup_tucker = lookup_tucker.cpu()

    acc, err = test.evaluate(lookup_tucker, test_tucker)

    # Delete tensors
    del test_emb
    del lookup_emb
    del test_tucker
    del lookup_tucker

    torch.cuda.empty_cache()
    
    mdl.train()
    return acc, err





# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
def seed_all(seed=42):
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None




import os
import shutil

def reorder(data_dir,order):
    # keep the index value
    reorder_list=copy.deepcopy(order)
    reorder_list.append(len(reorder_list)+1)
    reorder_list.insert(0, 0)
    # # Step 1: Check if ref.txt is available, if not, use ref.txt-like content
    ref_path = os.path.join(data_dir, 'ref.txt')


    with open(ref_path, 'r') as input_file:
        lines = input_file.readlines()
    
    with open(os.path.join(data_dir, 'hierarchical-level.txt'), 'w') as output_file:
        for line in lines:
            # Split the line into columns
            columns = line.strip().split()
            # Reorder columns based on the reorder_list
            reordered_columns = [columns[i] for i in reorder_list]

            # Write the reordered line to the output file
            output_file.write('\t'.join(reordered_columns) + '\n')
