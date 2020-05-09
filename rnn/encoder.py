import torch
import torch.nn as nn
import torch.nn.init as init
from rnn.attention import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRU_Encoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(GRU_Encoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h0 = torch.zeros(batch_size, hidden_size).to(device)
        self.inter_modal_h_linear = nn.Linear(hidden_size*2, hidden_size)
        self.gru0 = nn.GRUCell(input_size=input_size,
                               hidden_size=hidden_size)
        self.gru1 = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        init.xavier_uniform_(self.inter_modal_h_linear.weight)

    def gru_encoder(self, input, h0_in, h1_in):
        h0_out = self.gru0(input, h0_in)
        h1_out = self.gru1(h0_out, h1_in)
        return h0_out, h1_out

    def forward(self, input, inter_modal_h=None):
        seq_len = input.size(1)
        if inter_modal_h is None:
            h0 = h1 = self.h0
        else:
            if input.size(0) != inter_modal_h.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_hの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                print(input.size(), inter_modal_h.size())
            if inter_modal_h.size(1) == self.hidden_size:
                h0 = h1 = inter_modal_h
            elif inter_modal_h.size(1) == self.hidden_size*2:
                inter_modal_h = self.inter_modal_h_linear(inter_modal_h)
                h0 = h1 = inter_modal_h
        out = []
        for i in range(seq_len):
            h0, h1 = self.gru_encoder(input[:, i, :], h0, h1)
            out.append(h1)
        out = torch.stack(out, dim=1)
        return out, h1


class Bidirectional_GRU_Encoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Bidirectional_GRU_Encoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h0 = torch.zeros(batch_size, hidden_size).to(device)
        self.gru0_f = nn.GRUCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.gru0_b = nn.GRUCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.gru1_f = nn.GRUCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.gru1_b = nn.GRUCell(
            input_size=hidden_size, hidden_size=hidden_size)

    def gru_encoder(self, input, h0_f_in, h0_b_in, h1_f_in, h1_b_in):
        reverse_input = self.reverse_input(input)
        h0_f_out = self.gru0_f(input, h0_f_in)
        h0_b_out = self.gru0_b(reverse_input, h0_b_in)
        h1_f_out = self.gru1_f(h0_f_out, h1_f_in)
        h1_b_out = self.gru1_b(h0_b_out, h1_b_in)
        return h0_f_out, h0_b_out, h1_f_out, h1_b_out

    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input

    def forward(self, input, inter_modal_h=None):
        seq_len = input.size(1)
        if inter_modal_h is None:
            h0_f = h0_b = h1_f = h1_b = self.h0
        else:
            if input.size(0) != inter_modal_h.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_hの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                print(input.size(), inter_modal_h.size())
            if inter_modal_h.size(1) == self.hidden_size:
                h0_f = h0_b = h1_f = h1_b = inter_modal_h
            elif inter_modal_h.size(1) == self.hidden_size*2:
                h0_f = h1_f = inter_modal_h[:, :self.hidden_size]
                h0_b = h1_b = inter_modal_h[:, self.hidden_size:]
        out = []
        for i in range(seq_len):
            h0_f, h0_b, h1_f, h1_b = self.gru_encoder(
                input[:, i, :], h0_f, h0_b, h1_f, h1_b)
            h = torch.cat([h1_f, h1_b], dim=1)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class LSTM_Encoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(LSTM_Encoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h0 = torch.zeros(batch_size, hidden_size).to(device)
        self.c0 = torch.zeros(batch_size, hidden_size).to(device)
        self.inter_modal_h_linear = nn.Linear(hidden_size*2, hidden_size)
        self.inter_modal_c_linear = nn.Linear(hidden_size*2, hidden_size)
        self.lstm0 = nn.LSTMCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.lstm1 = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)
        init.xavier_uniform_(self.inter_modal_h_linear.weight)
        init.xavier_uniform_(self.inter_modal_c_linear.weight)

    def lstm_encoder(self, input, h0_in, c0_in, h1_in, c1_in):
        h0_out, c0_out = self.lstm0(input, (h0_in, c0_in))
        h1_out, c1_out = self.lstm1(h0_out, (h1_in, c1_in))
        return h0_out, c0_out, h1_out, c1_out

    def forward(self, input, inter_modal_h=None):
        seq_len = input.size(1)
        if inter_modal_h is None:
            h0 = h1 = self.h0
            c0 = c1 = self.c0
        else:
            inter_modal_h, inter_modal_c = inter_modal_h
            if input.size(0) != inter_modal_h.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_h, inter_modal_cの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                print(input.size(), inter_modal_h.size(), inter_modal_c.size())
            if inter_modal_h.size(1) == self.hidden_size:
                h0 = h1 = inter_modal_h
                c0 = c1 = inter_modal_c
            elif inter_modal_h.size(1) == self.hidden_size*2:
                inter_modal_h = self.inter_modal_h_linear(inter_modal_h)
                inter_modal_c = self.inter_modal_c_linear(inter_modal_c)
                h0 = h1 = inter_modal_h
                c0 = c1 = inter_modal_c
        out = []
        c_out = []
        for i in range(seq_len):
            h0, c0, h1, c1 = self.lstm_encoder(
                input[:, i, :], h0, c0, h1, c1)
            out.append(h1)
            c_out.append(c1)
        out = torch.stack(out, dim=1)
        c_out = torch.stack(c_out, dim=1)
        return (out, c_out), (h1, c1)


class Bidirectional_LSTM_Encoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Bidirectional_LSTM_Encoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h0 = torch.zeros(batch_size, hidden_size).to(device)
        self.c0 = torch.zeros(batch_size, hidden_size).to(device)
        self.lstm0_f = nn.LSTMCell(input_size=input_size,
                                   hidden_size=hidden_size)
        self.lstm0_b = nn.LSTMCell(input_size=input_size,
                                   hidden_size=hidden_size)
        self.lstm1_f = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.lstm1_b = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)

    def lstm_encoder(self, input, h0_f_in, c0_f_in, h0_b_in, c0_b_in, h1_f_in, c1_f_in, h1_b_in, c1_b_in):
        reverse_input = self.reverse_input(input)
        h0_f_out, c0_f_out = self.lstm0_f(input, (h0_f_in, c0_f_in))
        h0_b_out, c0_b_out = self.lstm0_b(reverse_input, (h0_b_in, c0_b_in))
        h1_f_out, c1_f_out = self.lstm1_f(h0_f_out, (h1_f_in, c1_f_in))
        h1_b_out, c1_b_out = self.lstm1_b(h0_b_out, (h1_b_in, c1_b_in))
        return h0_f_out, c0_f_out, h0_b_out, c0_b_out, h1_f_out, c1_f_out, h1_b_out, c1_b_out

    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input

    def forward(self, input, inter_modal_h=None):
        seq_len = input.size(1)
        if inter_modal_h is None:
            h0_f = h0_b = h1_f = h1_b = self.h0
            c0_f = c0_b = c1_f = c1_b = self.c0
        else:
            inter_modal_h, inter_modal_c = inter_modal_h
            if input.size(0) != inter_modal_h.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_h, inter_modal_cの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                print(input.size(), inter_modal_h.size(), inter_modal_c.size())
            if inter_modal_h.size(1) == self.hidden_size:
                h0_f = h0_b = h1_f = h1_b = inter_modal_h
                c0_f = c0_b = c1_f = c1_b = inter_modal_c
            elif inter_modal_h.size(1) == self.hidden_size*2:
                h0_f = h1_f = inter_modal_h[:, :self.hidden_size]
                h0_b = h1_b = inter_modal_h[:, self.hidden_size:]
                c0_f = c1_f = inter_modal_c[:, :self.hidden_size]
                c0_b = c1_b = inter_modal_c[:, self.hidden_size:]
        out = []
        c_out = []
        for i in range(seq_len):
            h0_f, c0_f, h0_b, c0_b, h1_f, c1_f, h1_b, c1_b = self.lstm_encoder(
                input[:, i, :], h0_f, c0_f, h0_b, c0_b, h1_f, c1_f, h1_b, c1_b)
            h = torch.cat([h1_f, h1_b], dim=1)
            c = torch.cat([c1_f, c1_b], dim=1)
            out.append(h)
            c_out.append(c)
        out = torch.stack(out, dim=1)
        c_out = torch.stack(c_out, dim=1)
        return (out, c_out), (h, c)

