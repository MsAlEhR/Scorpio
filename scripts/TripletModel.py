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
    
            self.embedding_size = self.pretrained_model.config.hidden_size

    
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





