import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import spacy

spacy = spacy.load("en_core_web_sm")


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Syntactic_GraphConvolution(nn.Module):
    def __init__(self, args):
        super(Syntactic_GraphConvolution, self).__init__()

    def forward(self, sen_encoder_out, sentences):
        
        return out