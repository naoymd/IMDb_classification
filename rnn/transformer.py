import torch
import torch.nn as nn
import torch.nn.init as init

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        input_size = args.input_size
        hidden_size = args.hidden_size
        
        self.fc = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size)

    def forward(self, x):
        out = self.fc(x)
        out = self.transformer(out, out)

        return out