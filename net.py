import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from rnn.my_encoder import *
from rnn.my_decoder import *
from rnn.my_transformer import *
from rnn.my_attention import *
from rnn.tcn import MSResNet

def l2norm(out):
  norm = torch.norm(out, dim=-1, keepdim=True)
  return out / norm

class Net(nn.Module):
  def __init__(self, word_embeddings, kwargs):
    super(Net, self).__init__()
    self.rnn = kwargs.rnn
    bidirection = kwargs.bidirection
    self.attention_rnn = kwargs.attention_rnn
    self.self_attention = kwargs.self_attention
    input_size = kwargs.input_size
    hidden_size = kwargs.hidden_size
    num_layers = kwargs.num_layers
    output_size = kwargs.output_size
    seq_length = kwargs.fix_length
    print('-'*50)
    print('rnn:', self.rnn)
    print('bidirection:', bidirection)
    print('attention_rnn:', self.attention_rnn)
    print('self_attention:', self.self_attention)
    print('-'*50)

    self.embed = nn.Embedding.from_pretrained(embeddings=word_embeddings, freeze=True)
    if self.rnn == 'GRU':
      if bidirection == True:
        if self.attention_rnn == True:
          self.encoder = Bidirectional_GRU_Encoder(input_size, hidden_size, num_layers)
          self.decoder = Attention_Bidirectional_GRU_Decoder(input_size, hidden_size, num_layers)
        else:
          self.encoder = Bidirectional_GRU_Encoder(input_size, hidden_size, num_layers)
          self.decoder = Bidirectional_GRU_Decoder(input_size, hidden_size, num_layers)
      else:
        if self.attention_rnn == True:
          self.encoder = GRU_Encoder(input_size, hidden_size, num_layers)
          self.decoder = Attention_GRU_Decoder(input_size, hidden_size, num_layers)
        else:
          self.encoder = GRU_Encoder(input_size, hidden_size, num_layers)
          self.decoder = GRU_Decoder(input_size, hidden_size, num_layers)
    elif self.rnn == 'GRU_':
      if kwargs.skip:
        self.encoder = GRU_Encoder_(input_size, hidden_size, num_layers, bidirection)
        self.decoder = GRU_Skip_Decoder_(input_size, hidden_size, num_layers, bidirection)
      else:
        self.encoder = GRU_Encoder_(input_size, hidden_size, num_layers, bidirection)
        self.decoder = GRU_Decoder_(input_size, hidden_size, num_layers, bidirection)
    elif self.rnn == 'LSTM':
      if bidirection == True:
        if self.attention_rnn == True:
          self.encoder = Bidirectional_LSTM_Encoder(input_size, hidden_size, num_layers)
          self.decoder = Attention_Bidirectional_LSTM_Decoder(input_size, hidden_size, num_layers)
        else:
          self.encoder = Bidirectional_LSTM_Encoder(input_size, hidden_size, num_layers)
          self.decoder = Bidirectional_LSTM_Decoder(input_size, hidden_size, num_layers)
      else:
        if self.attention_rnn == True:
          self.encoder = LSTM_Encoder(input_size, hidden_size, num_layers)
          self.decoder = Attention_LSTM_Decoder(input_size, hidden_size, num_layers)
        else:
          self.encoder = LSTM_Encoder(input_size, hidden_size, num_layers)
          self.decoder = LSTM_Decoder(input_size, hidden_size, num_layers)
    elif self.rnn == 'LSTM_':
      if kwargs.skip:
        self.encoder = LSTM_Encoder_(input_size, hidden_size, num_layers, bidirection)
        self.decoder = LSTM_Skip_Decoder_(input_size, hidden_size, num_layers, bidirection)
      else:
        self.encoder = LSTM_Encoder_(input_size, hidden_size, num_layers, bidirection)
        self.decoder = LSTM_Decoder_(input_size, hidden_size, num_layers, bidirection)
    elif self.rnn == 'Transformer':
      self.transformer = Transformer(input_size, hidden_size, num_layers)
      self.transformer_fc = nn.Linear(seq_length, 1)
      self.pool = nn.AdaptiveAvgPool1d(1)

    if bidirection == True:
      self.attention = Attention(hidden_size*2, **kwargs)
      self.fc = nn.Linear(hidden_size*2, output_size)
    else:
      self.attention = Attention(hidden_size, **kwargs)
      self.fc = nn.Linear(hidden_size, output_size)
    
    self.relu = nn.ReLU()
    self.prelu = nn.PReLU()
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(p=kwargs.dropout)
    init.xavier_uniform_(self.fc.weight)


  def forward(self, x):
    # print('x', x.size())
    embed = self.embed(x)
    # print('embed', embed.size())

    if 'GRU' in self.rnn:
      out, h = self.encoder(embed)
      out, h = self.decoder(embed, h, out)
      out = l2norm(out)
      h = l2norm(h[-1, :, :])
      # print('decoder', out.size(), h.size())
    elif  'LSTM' in self.rnn:
      out, h = self.encoder(embed)
      out, h = self.decoder(embed, h, out)
      h, _ = h
      out = l2norm(out)
      h = l2norm(h[-1, :, :])
    elif 'Transformer' in self.rnn:
      out = self.transformer(embed)
      out = l2norm(out)
      h = self.transformer_fc(out.permute(0, 2, 1)).squeeze(dim=-1)
      # h = self.pool(out.permute(0, 2, 1)).squeeze()
    
    if self.self_attention:
      out, attention_map = self.attention(h, out)
      # print('self attention', out.size())
    else:
      out = h
    
    out = self.dropout(out)
    # print('dropout', out.size())
    out = self.fc(out)
    # print('fc', out.size())
    # out = self.relu(out)
    # print('relu', out.size())
    out = self.prelu(out)
    # print('prelu', out.size())
    # out = self.softmax(out)
    # print('softmax', out.size())
    if self.self_attention:
      return out, attention_map
    else:
      return out
    


class Model(nn.Module):
  def __init__(self, word_embeddings, kwargs):
    super(Model, self).__init__()
    self.rnn = kwargs.rnn
    bidirection = kwargs.bidirection
    self.self_attention = kwargs.self_attention
    input_size = kwargs.input_size
    hidden_size = kwargs.hidden_size
    num_layers = kwargs.num_layers
    output_size = kwargs.output_size
    seq_length = kwargs.fix_length
    print('-'*50)
    print('rnn:', self.rnn)
    print('bidirection:', bidirection)
    print('self attention:', self.self_attention)
    print('-'*50)

    self.embed = nn.Embedding.from_pretrained(embeddings=word_embeddings, freeze=True)
    if self.rnn == 'GRU':
      if bidirection == True:
        self.encoder = Bidirectional_GRU_Encoder(input_size, hidden_size, num_layers)
      else:
        self.encoder = GRU_Encoder(input_size, hidden_size, num_layers)
    elif self.rnn == 'GRU_':
      self.encoder = GRU_Encoder_(input_size, hidden_size, num_layers, bidirection)
    elif self.rnn == 'LSTM':
      if bidirection == True:
        self.encoder = Bidirectional_LSTM_Encoder(input_size, hidden_size, num_layers)
      else:
        self.encoder = LSTM_Encoder(input_size, hidden_size, num_layers)
    elif self.rnn == 'LSTM_':
      self.encoder = LSTM_Encoder_(input_size, hidden_size, num_layers, bidirection)
    elif self.rnn == 'Transformer':
      self.transformer = Transformer_Encoder(input_size, hidden_size, num_layers)
      self.transformer_fc = nn.Linear(seq_length, 1)
      self.pool = nn.AdaptiveAvgPool1d(1)

    if bidirection == True and self.rnn != 'Transformer':
      self.attention = Attention(hidden_size*2, **kwargs)
      self.fc = nn.Linear(hidden_size*2, output_size)
    else:
      self.attention = Attention(hidden_size, **kwargs)
      self.fc = nn.Linear(hidden_size, output_size)
    
    self.relu = nn.ReLU()
    self.prelu = nn.PReLU()
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(p=kwargs.dropout)
    init.xavier_uniform_(self.fc.weight)


  def forward(self, x):
    # print('x', x.size())
    embed = self.embed(x)
    # print('embed', embed.size())

    if 'GRU' in self.rnn:
      out, h = self.encoder(embed)
      out = l2norm(out)
      h = l2norm(h[-1, :, :])
    elif 'LSTM' in self.rnn:
      out, h = self.encoder(embed)
      h, _ = h
      out = l2norm(out)
      h = l2norm(h[-1, :, :])
    elif 'Transformer' in self.rnn:
      out = self.transformer(embed)
      out = l2norm(out)
      h = self.transformer_fc(out.permute(0, 2, 1)).squeeze(dim=-1)
      # h = self.pool(out.permute(0, 2, 1)).squeeze()
    
    if self.self_attention:
      out, attention_map = self.attention(h, out)
      # print('self attention', out.size())
    else:
      out = h
    
    out = self.dropout(out)
    # print('dropout', out.size())
    out = self.fc(out)
    # print('fc', out.size())
    # out = self.relu(out)
    # print('relu', out.size())
    out = self.prelu(out)
    # print('prelu', out.size())
    # out = self.softmax(out)
    # print('softmax', out.size())
    if self.self_attention:
      return out, attention_map
    else:
      return out



class TCN(nn.Module):
  def __init__(self, word_embeddings, kwargs):
    super(TCN, self).__init__()
    self.embed = nn.Embedding.from_pretrained(embeddings=word_embeddings, freeze=True)
    self.tcn = MSResNet(kwargs.input_size, kwargs)
    self.length_fc = nn.Linear(kwargs.fix_length, 1)
    self.self_attention = kwargs.self_attention
    self.attention = Attention(kwargs.hidden_size, **kwargs)
    self.fc = nn.Linear(kwargs.hidden_size, kwargs.output_size)


  def forward(self, x):
    x = self.embed(x)
    x = self.tcn(x)
    out = self.length_fc(x.permute(0, 2, 1)).squeeze()
    if self.self_attention:
      out, attention_map = self.attention(out, x)
      out = F.relu(self.fc(out))
      return out, attention_map
    else:
      out = F.softmax(F.relu(self.fc(out)))
      return out



class GRU_Layer(nn.Module):
  def __init__(self, word_embeddings, kwargs):
    super(GRU_Layer, self).__init__()
    print('-'*50)
    print('bidirection:', kwargs.bidirection)
    print('self attention:', kwargs.self_attention)
    print('-'*50)
    self.embed = nn.Embedding.from_pretrained(embeddings=word_embeddings, freeze=True)
    self.h0 = torch.zeros(1, kwargs.batch_size, kwargs.hidden_size)
    self.gru = nn.GRU(kwargs.input_size, kwargs.hidden_size, num_layers=kwargs.num_layers, batch_first=True, bidirectional=kwargs.bidirection)
    self.self_attention = kwargs.self_attention
    self.attention = Attention(kwargs.hidden_size, **kwargs)
    self.fc = nn.Linear(kwargs.hidden_size, kwargs.output_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=kwargs.dropout)
    init.xavier_uniform_(self.fc.weight)


  def forward(self, x):
    x = self.embed(x)
    out, h = self.gru(x, self.h0)
    h = h.squeeze(dim=0)
    # print('out: {}, h: {}'.format(out.size(), h.size()))
    if self.self_attention:
      h, attention_map = self.attention(h, out)
      h = self.fc(h)
      # h = self.relu(h)
      # h = self.dropout(h)
      return h, attention_map
    else:
      h = self.fc(h)
      # h = self.relu(h)
      # h = self.dropout(h)
      return h