import torch
import torch.nn as nn
import torch.nn.init as init
from rnn.my_attention import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRU_Encoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(GRU_Encoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h0 = torch.zeros(batch_size, hidden_size).to(device)
        self.inter_modal_h_linear = nn.Linear(hidden_size*2, hidden_size)
        self.gru = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        init.xavier_uniform_(self.inter_modal_h_linear.weight)
        print(self.gru)

    def forward(self, input, inter_modal_h=None):
        seq_len = input.size(1)
        if inter_modal_h is None:
            h_in = [self.h0]*self.num_layers
        else:
            if input.size(0) != inter_modal_h.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_hの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_h = inter_modal_h.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_h.size())
            if inter_modal_h.size(1) == self.hidden_size:
                pass
            elif inter_modal_h.size(1) == self.hidden_size*2:
                inter_modal_h = self.inter_modal_h_linear(inter_modal_h)
            h_in = [inter_modal_h]*self.num_layers
        out = []
        for i in range(seq_len):
            h = input[:, i, :]
            for j, layer in enumerate(self.gru):
                h = layer(h, h_in[j])
                h_in[j] = h
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class Bidirectional_GRU_Encoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Bidirectional_GRU_Encoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h0 = torch.zeros(batch_size, hidden_size).to(device)
        self.gru_f = nn.ModuleList()
        self.gru_b = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru_f.append(nn.GRUCell(input_size, hidden_size))
            self.gru_b.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        print(self.gru_f)
        print(self.gru_b)

    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input

    def forward(self, input, inter_modal_h=None):
        seq_len = input.size(1)
        if inter_modal_h is None:
            h_f_in = h_b_in = [self.h0]*self.num_layers
        else:
            if input.size(0) != inter_modal_h.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_hの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_h = inter_modal_h.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_h.size())
            if inter_modal_h.size(1) == self.hidden_size:
                h_f_in = h_b_in = [inter_modal_h]*self.num_layers
            elif inter_modal_h.size(1) == self.hidden_size*2:
                h_f_in = [inter_modal_h[:, :self.hidden_size]]*self.num_layers
                h_b_in = [inter_modal_h[:, self.hidden_size:]]*self.num_layers
        out = []
        for i in range(seq_len):
            h_f = input[:, i, :]
            h_b = self.reverse_input(h_f)
            for j, (layer_f, layer_b) in enumerate(zip(self.gru_f, self.gru_b)):
                h_f = layer_f(h_f, h_f_in[j])
                h_b = layer_b(h_b, h_b_in[j])
                h_f_in[j] = h_f
                h_b_in[j] = h_b
            h = torch.cat([h_f, h_b], dim=1)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class LSTM_Encoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(LSTM_Encoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h0 = torch.zeros(batch_size, hidden_size).to(device)
        self.c0 = torch.zeros(batch_size, hidden_size).to(device)
        self.inter_modal_h_linear = nn.Linear(hidden_size*2, hidden_size)
        self.inter_modal_c_linear = nn.Linear(hidden_size*2, hidden_size)
        self.lstm = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        init.xavier_uniform_(self.inter_modal_h_linear.weight)
        init.xavier_uniform_(self.inter_modal_c_linear.weight)
        print(self.lstm)

    def forward(self, input, inter_modal_h=None):
        seq_len = input.size(1)
        if inter_modal_h is None:
            h_in = [self.h0]*self.num_layers
            c_in = [self.c0]*self.num_layers
        else:
            inter_modal_h, inter_modal_c = inter_modal_h
            if input.size(0) != inter_modal_h.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_h, inter_modal_cの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_h = inter_modal_h.repeat_interleave(input.size(0), dim=0)
                inter_modal_c = inter_modal_c.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_h.size(), inter_modal_c.size())
            if inter_modal_h.size(1) == self.hidden_size:
                pass
            elif inter_modal_h.size(1) == self.hidden_size*2:
                inter_modal_h = self.inter_modal_h_linear(inter_modal_h)
                inter_modal_c = self.inter_modal_c_linear(inter_modal_c)
            h_in = [inter_modal_h]*self.num_layers
            c_in = [inter_modal_c]*self.num_layers
        out = []
        c_out = []
        for i in range(seq_len):
            h = input[:, i, :]
            for j, layer in enumerate(self.lstm):
                h, c = layer(h, (h_in[j], c_in[j]))
                h_in[j] = h
                c_in[j] = c
            out.append(h)
            c_out.append(c)
        out = torch.stack(out, dim=1)
        c_out = torch.stack(c_out, dim=1)
        return (out, c_out), (h, c)


class Bidirectional_LSTM_Encoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Bidirectional_LSTM_Encoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h0 = torch.zeros(batch_size, hidden_size).to(device)
        self.c0 = torch.zeros(batch_size, hidden_size).to(device)
        self.lstm_f = nn.ModuleList()
        self.lstm_b = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm_f.append(nn.LSTMCell(input_size, hidden_size))
            self.lstm_b.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        print(self.lstm_f)
        print(self.lstm_b)

    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input

    def forward(self, input, inter_modal_h=None):
        seq_len = input.size(1)
        if inter_modal_h is None:
            h_f_in = h_b_in = [self.h0]*self.num_layers
            c_f_in = c_b_in = [self.c0]*self.num_layers
        else:
            inter_modal_h, inter_modal_c = inter_modal_h
            if input.size(0) != inter_modal_h.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_h, inter_modal_cの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_h = inter_modal_h.repeat_interleave(input.size(0), dim=0)
                inter_modal_c = inter_modal_c.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_h.size(), inter_modal_c.size())
            if inter_modal_h.size(1) == self.hidden_size:
                h_f_in = h_b_in = [inter_modal_h]*self.num_layers
                c_f_in = c_b_in = [inter_modal_c]*self.num_layers
            elif inter_modal_h.size(1) == self.hidden_size*2:
                h_f_in = [inter_modal_h[:, :self.hidden_size]]*self.num_layers
                h_b_in = [inter_modal_h[:, self.hidden_size:]]*self.num_layers
                c_f_in = [inter_modal_c[:, :self.hidden_size]]*self.num_layers
                c_b_in = [inter_modal_c[:, self.hidden_size:]]*self.num_layers
        out = []
        c_out = []
        for i in range(seq_len):
            h_f = input[:, i, :]
            h_b = self.reverse_input(h_f)
            for j, (layer_f, layer_b) in enumerate(zip(self.lstm_f, self.lstm_b)):
                h_f, c_f = layer_f(h_f, (h_f_in[j], c_f_in[j]))
                h_b, c_b = layer_b(h_b, (h_b_in[j], c_b_in[j]))
                h_f_in[j] = h_f
                h_b_in[j] = h_b
                c_f_in[j] = c_f
                c_b_in[j] = c_b
            h = torch.cat([h_f, h_b], dim=1)
            c = torch.cat([c_f, c_b], dim=1)
            out.append(h)
            c_out.append(c)
        out = torch.stack(out, dim=1)
        c_out = torch.stack(c_out, dim=1)
        return (out, c_out), (h, c)