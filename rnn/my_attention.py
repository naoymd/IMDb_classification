import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dimension, args):
        super(Attention, self).__init__()
        self.qkv_linear = args.attention_qkv_linear
        self.query_linear = nn.Linear(dimension, dimension)
        self.key_linear = nn.Linear(dimension, dimension)
        self.value_linear = nn.Linear(dimension, dimension)

        self.mask = args.attention_mask
        self.scaled = args.scaled_dot_product
        self.scale = 1.0 / math.sqrt(dimension)
        self.softmax = nn.Softmax(dim=-1)

        self.output_mode = args.attention_output
        self.cat_linear = nn.Linear(dimension*2, dimension)
        self.relu = nn.ReLU()
        self.output_linear = nn.Linear(dimension, dimension)
        
        self.tanh = nn.Tanh()

        init.xavier_uniform_(self.query_linear.weight)
        init.xavier_uniform_(self.key_linear.weight)
        init.xavier_uniform_(self.value_linear.weight)
        init.xavier_uniform_(self.cat_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)

    def subsequent_mask(self, attention_map):
        # To Do
        mask = torch.ones_like(attention_map).tril().bool()
        return mask

    def pad_mask(self, attention_map, query, memory):
        # To Do
        mask = None
        return mask

    def forward(self, input, memory, mask=None):
        """
        Inputs:
            input : (batch_size, output_size, dimension) or (batch_size, dimension)
            -> memoryの情報をattentionされる特徴量
            memory : (batch_size, queried_size, dimension)
            -> inputにattentionする特徴量
        Outputs:
            output : (batch_size, output_size, dimension)
            -> attentionされたinput特徴量
            attention_map : (batch_size, output_size, queried_size)
            -> attention map
        """
        query = input
        key = value = memory

        query_squeeze = False
        if len(query.size()) == 2:
            query = query.unsqueeze(dim=1)
            query_squeeze = True

        if self.qkv_linear:
            query = self.query_linear(query)
            key = self.key_linear(key)
            value = self.value_linear(value)
        """
        query : (batch_size, output_size, dimension) or (batch_size, 1, dimension)
        key : (batch_size, queried_size, dimension)
        value: (batch_size, queried_size, dimension)
        """
        """
        attention_map :
            (batch_size, output_size, dimension) * (batch_size, dimension, queried_size)
            -> (batch_size, output_size, queried_size)
        """
        attention_map = torch.matmul(query, key.permute(0, 2, 1))
        if self.scaled:
            attention_map = attention_map.mul_(self.scale)
        attention_map = self.softmax(attention_map)
        if self.mask and input != memory:
            #To Do (mask for <'pad'>)
            # mask = self.pad_mask()
            attention_map = attention_map * mask
            pass
        if self.mask and input == memory:
            # To Do (mask for self attention)
            # mask = self.pad_mask()
            # attention_map = attention_map * mask
            mask = self.subsequent_mask(attention_map)
            attention_map = attention_map * mask
        """
        output :
            (batch_size, output_size, queried_size) * (batch_size, queried_size, dimension) 
            -> (batch_size, output_size, dimension)
        (option)
            cat : output と query のconcat
            add : output と query の和
        """
        output = torch.matmul(attention_map, value)
        if self.output_mode == 'cat':
            output = torch.cat([output, query], dim=2)
            output = self.cat_linear(output)
            output = self.relu(output)
        elif self.output_mode == 'add':
            output = self.relu(output + query)
        else:
            pass
        output = self.output_linear(output)
        output = self.tanh(output)
        if query_squeeze:
            output = output.squeeze(dim=1)
        """
        output:
            (batch_size, output_size, dimension) or (batch_size, dimension)
        attention_map:
            (batch_size, output_size, queried_size) or (batch_size, 1, queried_size)
        """
        return output, attention_map


class MultiheadAttention(nn.Module):
    def __init__(self, dimension, args):
        super(MultiheadAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(dimension, num_heads=8)

    def forward(self, input, memory):
        """
        Inputs:
            input : (batch_size, output_size, dimension) or (batch_size, dimension)
            -> memoryの情報をattentionされる特徴量
            memory : (batch_size, queried_size, dimension)
            -> inputにattentionする特徴量
        Outputs:
            output : (batch_size, output_size, dimension)
            -> attentionされたinput特徴量
            attention_map : (batch_size, output_size, queried_size)
            -> attention map
        """
        query = input
        key = value = memory

        output, attention_map = self.multihead_attention(query, key, value)
        return output, attention_map