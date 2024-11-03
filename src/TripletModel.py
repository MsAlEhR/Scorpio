"""
Author: Saleh Refahi
Email: sr3622@drexel.edu
Date: 2024-06-26

"""


import torch
import torch.utils.data
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel




 
class AveragePool1dAlongAxis(nn.Module):
    def __init__(self, axis):
        super(AveragePool1dAlongAxis, self).__init__()
        self.axis = axis

    def forward(self, x, mask=None):
        if mask is not None:
            # Zero out padded tokens
            x = x * mask.unsqueeze(-1).float()
            # Sum and divide by non-zero counts to get the mean of non-padded tokens
            summed = torch.sum(x, dim=self.axis)
            counts = torch.clamp(mask.sum(dim=self.axis, keepdim=True), min=1)  # Avoid division by zero
            return summed / counts
        else:
            # Default mean along the axis
            return torch.mean(x, dim=self.axis)


class Tuner(nn.Module):
    def __init__(self, pretrained_model="", embedding_size=4101, motif_freq=True):
        super(Tuner, self).__init__()

        self.motif_freq = motif_freq
        self.avg_pooling = AveragePool1dAlongAxis(1)  # Averaging layer
        self.embedding_size =embedding_size
        if not self.motif_freq:
            # Load the pre-trained model
            self.pretrained_model = AutoModel.from_pretrained(pretrained_model, output_hidden_states=True)
    
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
    
            for param in self.pretrained_model.encoder.layer[-1].parameters():
                param.requires_grad = True 
                
            # for param in self.pretrained_model.encoder.layer[-2].parameters():
            #     param.requires_grad = True     
            
            # for param in self.pretrained_model.encoder.layer[-2].parameters():
            #     param.requires_grad = True

            self.embedding_size = self.pretrained_model.config.hidden_size
            


        self.Tuner = nn.Sequential(
            nn.Linear(self.embedding_size , 768), 
            nn.ReLU(),
            nn.Linear(768, 256))



    def calc_frequency(self, X):
        ignored_tokens = {0, 1, 2, 3, 4}  # Tokens to ignore
        batch_size = X.size(0)
        freq_vectors = torch.zeros(batch_size, self.embedding_size, device=X.device)  # Initialize frequency vector
    
        for i in range(batch_size):
            indices, counts = torch.unique(X[i], return_counts=True)
            
            # Filter out ignored tokens
            mask = ~torch.isin(indices, torch.tensor(list(ignored_tokens), device=X.device))
            indices = indices[mask]
            counts = counts[mask]
            
            # Update frequency vector for valid tokens only
            freq_vectors[i, indices.long()] += counts.float()
    
            # Calculate sequence length excluding ignored tokens
            seq_len = counts.sum()  # Sum of valid counts
            if seq_len > 0:
                freq_vectors[i] /= seq_len  # Normalize by valid sequence length
                freq_vectors[i] *= 10 # Scale the vector to surpass sparse
        return freq_vectors
    
    def single_pass_freq(self, X):
        # Calculate frequency vector for the anchor input and add 1 to all values
        pooled_output = self.calc_frequency(X)
        pooled_output = self.Tuner(pooled_output)
        return pooled_output


    
    def single_pass(self, X):
        X = X.long() 
        attention_mask = (X != 2).int()  # Mask where padding tokens (ID = 2) are marked as False
        outputs = self.pretrained_model(X,encoder_attention_mask=attention_mask, attention_mask=attention_mask)
        last_hidden_state = outputs.hidden_states[-1]
        avg = self.avg_pooling(last_hidden_state, mask=attention_mask)
        pooled_output = self.Tuner(avg)
        return pooled_output

    def forward(self, X):
        
        if self.motif_freq:
            # Use frequency-based processing only for the anchor
            anchor = self.single_pass_freq(X[:, 0, :])
            pos = self.single_pass_freq(X[:, 1, :])
            neg = self.single_pass_freq(X[:, 2, :])
        else:
            # Use regular single_pass processing
            anchor = self.single_pass(X[:, 0, :])
            pos = self.single_pass(X[:, 1, :])
            neg = self.single_pass(X[:, 2, :])
        
        return anchor, pos, neg



