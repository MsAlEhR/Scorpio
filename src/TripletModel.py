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

    def forward(self, x):
        return torch.mean(x, dim=self.axis)


class Tuner(nn.Module):
    def __init__(self, pretrained_model="", embedding_size=4096, from_embedding=True):
        super(Tuner, self).__init__()

        self.from_embedding = from_embedding

        if not self.from_embedding:
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
            
            self.Anchor = nn.Sequential(
                nn.Linear(4101, 768), 
                nn.ReLU(),
                nn.Linear(768, 256),
            )
    
            self.Tuner = nn.Sequential(
                AveragePool1dAlongAxis(1),
                nn.ReLU(),
                nn.Linear(self.embedding_size, 768), 
                nn.ReLU(),
                nn.Linear(768, 256),
            )

        else:
            self.embedding_size = embedding_size
            self.Tuner = nn.Sequential(
                nn.Linear(self.embedding_size, 768), 
                nn.ReLU(),
                nn.Linear(768, 256),
            )


    def calc_frequency(self, X, vocab_size=4101):
        """
        Convert the anchor (X) to a frequency vector of size vocab_size,
        and add 1 to all counts (ensure no zeros).
        Args:
            X (torch.Tensor): Input batch of shape (batch_size, sequence_length)
            vocab_size (int): Size of the frequency vector (e.g., 4101)
        Returns:
            torch.Tensor: Frequency vector of shape (batch_size, vocab_size)
        """
        batch_size = X.size(0)
        freq_vectors = torch.ones(batch_size, vocab_size).to(X.device)  # Initialize with 1s instead of 0s
        seq_len = X.size(1)
        for i in range(batch_size):
            indices, counts = torch.unique(X[i], return_counts=True)
            indices = indices.long()  # Ensure indices are of type long
            freq_vectors[i, indices] += counts.float()  # Add counts to frequency vector, starting from 1
            freq_vectors[i]=freq_vectors[i]/seq_len   
        return freq_vectors
    
    def single_pass_anchor(self, X):
        # Calculate frequency vector for the anchor input and add 1 to all values
        freq_X = self.calc_frequency(X)
        pooled_output = self.Anchor(freq_X)
        return pooled_output
    

    def single_pass(self, X):
        if not self.from_embedding:
            X = X.long() 
            outputs = self.pretrained_model(X)
            last_hidden_state = outputs.hidden_states[-1]
            pooled_output = self.Tuner(last_hidden_state)
        else:
            X = X.float()
            pooled_output = self.Tuner(X)

        return pooled_output

    def forward(self, X):
        anchor = self.single_pass(X[:, 0, :])
        pos = self.single_pass(X[:, 1, :])
        neg = self.single_pass(X[:, 2, :])
        return anchor, pos, neg

    
    # def forward(self, X):
    #         # Calculate frequency-based input only for the anchor
    #         anchor = self.single_pass_anchor(X[:, 0, :])
    #         # Regular input pass for pos and neg
    #         pos = self.single_pass(X[:, 1, :])
    #         neg = self.single_pass(X[:, 2, :])
            
    #         return anchor, pos, neg



