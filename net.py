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
    batch_size = kwargs.batch_size
    input_size = kwargs.input_size
    hidden_size = kwargs.hidden_size
    output_size = kwargs.output_size
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
          self.encoder = Bidirectional_GRU_Encoder(batch_size, input_size)
          self.decoder = Attention_Bidirectional_GRU_Decoder(batch_size, input_size, **kwargs)
        else:
          self.encoder = Bidirectional_GRU_Encoder(batch_size, input_size)
          self.decoder = Bidirectional_GRU_Decoder(batch_size, input_size, **kwargs)
      else:
        if self.attention_rnn == True:
          self.encoder = GRU_Encoder(batch_size, input_size)
          self.decoder = Attention_GRU_Decoder(batch_size, input_size, **kwargs)
        else:
          self.encoder = GRU_Encoder(batch_size, input_size)
          self.decoder = GRU_Decoder(batch_size, input_size, **kwargs)
    elif self.rnn == 'LSTM':
      if bidirection == True:
        if self.attention_rnn == True:
          self.encoder = Bidirectional_LSTM_Encoder(batch_size, input_size)
          self.decoder = Attention_Bidirectional_LSTM_Decoder(batch_size, input_size, **kwargs)
        else:
          self.encoder = Bidirectional_LSTM_Encoder(batch_size, input_size)
          self.decoder = Bidirectional_LSTM_Decoder(batch_size, input_size, **kwargs)
      else:
        if self.attention_rnn == True:
          self.encoder = LSTM_Encoder(batch_size, input_size)
          self.decoder = Attention_LSTM_Decoder(batch_size, input_size, **kwargs)
        else:
          self.encoder = LSTM_Encoder(batch_size, input_size)
          self.decoder = LSTM_Decoder(batch_size, input_size, **kwargs)
    elif self.rnn == 'Transformer':
      self.transformer = Transformer(input_size)

    if bidirection == True:
      self.attention = Attention(hidden_size*2, **kwargs)
      self.fc = nn.Linear(hidden_size*2, output_size)
    else:
      self.attention = Attention(hidden_size, **kwargs)
      self.fc = nn.Linear(hidden_size, output_size)

    if self.self_attention == True or self.rnn == 'Transformer':
      self.length_fc = nn.Linear(kwargs.fix_length, 1)
      init.xavier_uniform_(self.length_fc.weight)
    
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=kwargs.dropout)
    init.xavier_uniform_(self.fc.weight)

  def forward(self, x):
    # print('x', x.size())
    embed = self.embed(x)
    # print('embed', embed.size())

    if self.rnn == 'GRU':
      out, h = self.encoder(embed)
      out, h = self.decoder(embed, out, h)
      # out = l2norm(out)
      # h = l2norm(h)
      # print('decoder', out.size(), h.size())
    elif  self.rnn == 'LSTM':
      out, h = self.encoder(embed)
      out, h = self.decoder(embed, out, h)
      h, _ = h
      # out = l2norm(out)
      # h = l2norm(h)
    elif self.rnn == 'Transformer':
      out = self.transformer(embed)
      # out = l2norm(out)

    if self.self_attention:
      if self.rnn == 'GRU' or self.rnn =='LSTM':
        out, attention_map = self.attention(h, out)
        # print('self attention', out.size(), attention_map.size())
      elif self.rnn == 'Transformer':
        out, attention_map = self.attention(out, out)
        out = self.length_fc(out.permute(0, 2, 1)).squeeze(dim=2)
        # print('self attention', out.size(), attention_map.size())
    else:
      if self.rnn == 'GRU' or self.rnn == 'LSTM':
        out = h
        # print('length fc', out.size())
      elif self.rnn == 'Transformer':
        out = self.length_fc(out.permute(0, 2, 1)).suqueeze(dim=2)
        # print('length_fc', out.size())

    out = self.fc(out)
    # print('fc', out.size())
    out = self.relu(out)
    # print('relu', out.size())
    out = self.dropout(out)
    # print('dropout', out.size())
    if self.self_attention:
      return out, attention_map
    else:
      return out
    

class Model(nn.Module):
  def __init__(self, word_embeddings, kwargs):
    super(Model, self).__init__()
    self.rnn = kwargs.rnn
    self.bidirection = kwargs.bidirection
    self.self_attention = kwargs.self_attention
    batch_size = kwargs.batch_size
    input_size = kwargs.input_size
    hidden_size = kwargs.hidden_size
    output_size = kwargs.output_size
    print('-'*50)
    print('rnn:', self.rnn)
    print('bidirection:', self.bidirection)
    print('self attention:', self.self_attention)
    print('-'*50)

    self.embed = nn.Embedding.from_pretrained(embeddings=word_embeddings, freeze=True)
    if self.rnn == 'GRU':
      if self.bidirection == True:
        self.encoder = Bidirectional_GRU_Encoder(batch_size, input_size)
      else:
        self.encoder = GRU_Encoder(batch_size, input_size)
    elif self.rnn == 'LSTM':
      if self.bidirection == True:
        self.encoder = Bidirectional_LSTM_Encoder(batch_size, input_size)
      else:
        self.encoder = LSTM_Encoder(batch_size, input_size)
    elif self.rnn == 'Transformer':
      self.transformer = Transformer(input_size)

    if self.bidirection == True:
      self.attention = Attention(hidden_size*2, **kwargs)
      self.fc = nn.Linear(hidden_size*2, output_size)
    else:
      self.attention = Attention(hidden_size, **kwargs)
      self.fc = nn.Linear(hidden_size, output_size)

    if self.self_attention == True or self.rnn == 'Transformer':
      self.length_fc = nn.Linear(kwargs.fix_length, 1)
      init.xavier_uniform_(self.length_fc.weight)
    
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=kwargs.dropout)
    init.xavier_uniform_(self.fc.weight)

  def forward(self, x):
    # print('x', x.size())
    embed = self.embed(x)
    # print('embed', embed.size())

    if self.rnn == 'GRU':
      out, h = self.encoder(embed)
      # out = l2norm(out)
      # h = l2norm(h)
    elif  self.rnn == 'LSTM':
      out, h = self.encoder(embed)
      out, _ = out
      h, _ = h
      # out = l2norm(out)
      # h = l2norm(h)
    elif self.rnn == 'Transformer':
      out = self.transformer(embed)
      # out = l2norm(out)
    
    if self.self_attention:
      if self.rnn == 'GRU' or self.rnn == 'LSTM':
        out, attention_map = self.attention(h, out)
        # print('self attention', out.size())
      elif self.rnn == 'Transformer':
        out, attention_map = self.attention(out, out)
        out = self.length_fc(out.permute(0, 2, 1)).squeeze(dim=2)
        # print('self attention', out.size())
    else:
      if self.rnn == 'GRU' or self.rnn == 'LSTM':
        out = h
      elif self.rnn == 'Transformer':
        out = self.length_fc(out.permute(0, 2, 1)).squeeze(dim=2)
        # print('length_fc', out.size())
    
    out = self.fc(out)
    # print('fc', out.size())
    out = self.relu(out)
    # print('relu', out.size())
    out = self.dropout(out)
    # print('dropout', out.size())
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
      out = F.relu(self.fc(out))
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