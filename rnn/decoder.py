import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from rnn.my_attention import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size, args)
        self.gru0 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.gru1 = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)

    def gru_decoder(self, input, h0_in, h1_in):
        h0_out = self.gru0(input, h0_in)
        h1_out = self.gru1(h0_out, h1_in)
        return h0_out, h1_out

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        if inter_modal_out is not None:
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
        out = []
        h0 = h1 = encoder_h
        for i in range(seq_len):
            h0, h1 = self.gru_decoder(input[:, i, :], h0, h1)
            out.append(h1)
        out = torch.stack(out, dim=1)
        return out, h1


class Bidirectional_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Bidirectional_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size*2, args)
        self.gru0_f = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.gru0_b = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.gru1_f = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.gru1_b = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)

    def gru_decoder(self, input, h0_f_in, h0_b_in, h1_f_in, h1_b_in):
        reverse_input = self.reverse_input(input)
        h0_f_out = self.gru0_f(input, h0_f_in)
        h0_b_out = self.gru0_b(reverse_input, h0_b_in)
        h1_f_out = self.gru1_f(h0_f_out, h1_f_in)
        h1_b_out = self.gru1_b(h0_b_out, h1_b_in)
        return h0_f_out, h0_b_out, h1_f_out, h1_b_out

    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        if inter_modal_out is not None:
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
        out = []
        h0_f = h1_f = encoder_h[:, :self.hidden_size]
        h0_b = h1_b = encoder_h[:, self.hidden_size:]
        for i in range(seq_len):
            h0_f, h0_b, h1_f, h1_b = self.gru_decoder(
                input[:, i, :], h0_f, h0_b, h1_f, h1_b)
            h = torch.cat([h1_f, h1_b], dim=1)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class Attention_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Attention_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size, args)
        self.gru0 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.gru1 = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_attention = Attention(hidden_size, args)

    def gru_decoder(self, input, h0_in, h1_in):
        h0_out = self.gru0(input, h0_in)
        h1_out = self.gru1(h0_out, h1_in)
        return h0_out, h1_out

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        if inter_modal_out is not None:
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
        out = []
        h0 = h1 = encoder_h
        for i in range(seq_len):
            h0, h1 = self.gru_decoder(input[:, i, :], h0, h1)
            h, _ = self.rnn_attention(h1, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class Modal_Attention_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Modal_Attention_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size, args)
        self.out_attention = Attention(hidden_size, args)
        self.gru0 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.gru1 = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_attention = Attention(hidden_size, args)

    def gru_decoder(self, input, h0_in, h1_in):
        h0_out = self.gru0(input, h0_in)
        h1_out = self.gru1(h0_out, h1_in)
        return h0_out, h1_out

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        if inter_modal_out is not None:
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
            encoder_out, _ = self.out_attention(encoder_out, inter_modal_out)
        else:
            encoder_out = self.out_attention(encoder_out, encoder_out)
        out = []
        h0 = h1 = encoder_h
        for i in range(seq_len):
            h0, h1 = self.gru_decoder(input[:, i, :], h0, h1)
            h, _ = self.rnn_attention(h1, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class Attention_Bidirectional_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Attention_Bidirectional_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size*2, args)
        self.gru0_f = nn.GRUCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.gru0_b = nn.GRUCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.gru1_f = nn.GRUCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.gru1_b = nn.GRUCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_attention = Attention(hidden_size*2, args)

    def gru_decoder(self, input, h0_f_in, h0_b_in, h1_f_in, h1_b_in):
        reverse_input = self.reverse_input(input)
        h0_f_out = self.gru0_f(input, h0_f_in)
        h0_b_out = self.gru0_b(reverse_input, h0_b_in)
        h1_f_out = self.gru1_f(h0_f_out, h1_f_in)
        h1_b_out = self.gru1_b(h0_b_out, h1_b_in)
        return h0_f_out, h0_b_out, h1_f_out, h1_b_out
        
    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        if inter_modal_out is not None:
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
        out = []
        h0_f = h1_f = encoder_h[:, :self.hidden_size]
        h0_b = h1_b = encoder_h[:, self.hidden_size:]
        for i in range(seq_len):
            h0_f, h0_b, h1_f, h1_b = self.gru_decoder(
                input[:, i, :], h0_f, h0_b, h1_f, h1_b)
            h = torch.cat([h1_f, h1_b], dim=1)
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class Modal_Attention_Bidirectional_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Modal_Attention_Bidirectional_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size*2, args)
        self.out_attention = Attention(hidden_size*2, args)
        self.gru0_f = nn.GRUCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.gru0_b = nn.GRUCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.gru1_f = nn.GRUCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.gru1_b = nn.GRUCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_attention = Attention(hidden_size*2, args)

    def gru_decoder(self, input, h0_f_in, h0_b_in, h1_f_in, h1_b_in):
        reverse_input = self.reverse_input(input)
        h0_f_out = self.gru0_f(input, h0_f_in)
        h0_b_out = self.gru0_b(reverse_input, h0_b_in)
        h1_f_out = self.gru1_f(h0_f_out, h1_f_in)
        h1_b_out = self.gru1_b(h0_b_out, h1_b_in)
        return h0_f_out, h0_b_out, h1_f_out, h1_b_out
        
    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        if inter_modal_out is not None:
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
            encoder_out, _ = self.out_attention(encoder_out, inter_modal_out)
        out = []
        h0_f = h1_f = encoder_h[:, :self.hidden_size]
        h0_b = h1_b = encoder_h[:, self.hidden_size:]
        for i in range(seq_len):
            h0_f, h0_b, h1_f, h1_b = self.gru_decoder(
                input[:, i, :], h0_f, h0_b, h1_f, h1_b)
            h = torch.cat([h1_f, h1_b], dim=1)
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size, args)
        self.c_attention = Attention(hidden_size, args)
        self.lstm0 = nn.LSTMCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.lstm1 = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)

    def lstm_decoder(self, input, h0_in, c0_in, h1_in, c1_in):
        h0_out, c0_out = self.lstm0(input, (h0_in, c0_in))
        h1_out, c1_out = self.lstm1(h0_out, (h1_in, c1_in))
        return h0_out, c0_out, h1_out, c1_out

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        encoder_out, encoder_c_out = encoder_out
        encoder_h, encoder_c = encoder_h
        if inter_modal_out is not None:
            inter_modal_out, inter_modal_c_out = inter_modal_out
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_out, inter_modal_c_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                inter_modal_c_out = inter_modal_c_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size(), inter_modal_c_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
            encoder_c, _ = self.c_attention(encoder_h, inter_modal_c_out)
        out = []
        h0 = h1 = encoder_h
        c0 = c1 = encoder_c
        for i in range(seq_len):
            h0, c0, h1, c1 = self.lstm_decoder(input[:, i, :], h0, c0, h1, c1)
            out.append(h1)
        out = torch.stack(out, dim=1)
        return out, (h1, c1)


class Bidirectional_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Bidirectional_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size*2, args)
        self.c_attention = Attention(hidden_size*2, args)
        self.lstm0_f = nn.LSTMCell(input_size=input_size,
                                   hidden_size=hidden_size)
        self.lstm0_b = nn.LSTMCell(input_size=input_size,
                                   hidden_size=hidden_size)
        self.lstm1_f = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.lstm1_b = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)

    def lstm_decoder(self, input, h0_f_in, c0_f_in, h0_b_in, c0_b_in, h1_f_in, c1_f_in, h1_b_in, c1_b_in):
        reverse_input = self.reverse_input(input)
        h0_f_out, c0_f_out = self.lstm0_f(input, (h0_f_in, c0_f_in))
        h0_b_out, c0_b_out = self.lstm0_b(reverse_input, (h0_b_in, c0_b_in))
        h1_f_out, c1_f_out = self.lstm1_f(h0_f_out, (h1_f_in, c1_f_in))
        h1_b_out, c1_b_out = self.lstm1_b(h0_b_out, (h1_b_in, c1_b_in))
        return h0_f_out, c0_f_out, h0_b_out, c0_b_out, h1_f_out, c1_f_out, h1_b_out, c1_b_out

    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        encoder_out, encoder_c_out = encoder_out
        encoder_h, encoder_c = encoder_h
        if inter_modal_out is not None:
            inter_modal_out, inter_modal_c_out = inter_modal_out
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_out, inter_modal_c_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                inter_modal_c_out = inter_modal_c_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size(), inter_modal_c_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
            encoder_c, _ = self.c_attention(encoder_h, inter_modal_c_out)
        out = []
        h0_f = h1_f = encoder_h[:, :self.hidden_size]
        h0_b = h1_b = encoder_h[:, self.hidden_size:]
        c0_f = c1_f = encoder_c[:, :self.hidden_size]
        c0_b = c1_b = encoder_c[:, self.hidden_size:]        
        for i in range(seq_len):
            h0_f, c0_f, h0_b, c0_b, h1_f, c1_f, h1_b, c1_b = self.lstm_decoder(
                h0_f, c0_f, h0_b, c0_b, h1_f, c1_f, h1_b, c1_b)
            h = torch.cat([h1_f, h1_b], dim=1)
            c = torch.cat([c1_f, c1_b], dim=1)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c)


class Attention_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Attention_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size, args)
        self.c_attention = Attention(hidden_size, args)
        self.lstm0 = nn.LSTMCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.lstm1 = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.attention = Attention(hidden_size)
        self.rnn_attention = Attention(hidden_size, args)

    def lstm_decoder(self, input, h0_in, c0_in, h1_in, c1_in):
        h0_out, c0_out = self.lstm0(input, (h0_in, c0_in))
        h1_out, c1_out = self.lstm1(h0_out, (h1_in, c1_in))
        return h0_out, c0_out, h1_out, c1_out

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        encoder_out, encoder_c_out = encoder_out
        encoder_h, encoder_c = encoder_h
        if inter_modal_out is not None:
            inter_modal_out, inter_modal_c_out = inter_modal_out
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_out, inter_modal_c_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                inter_modal_c_out = inter_modal_c_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size(), inter_modal_c_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
            encoder_c, _ = self.c_attention(encoder_h, inter_modal_c_out)
        out = []
        h0 = h1 = encoder_h
        c0 = c1 = encoder_c
        for i in range(seq_len):
            h0, c0, h1, c1 = self.lstm_decoder(input[:, i, :], h0, c0, h1, c1)
            h = self.rnn_attention(h1, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c1)


class Modal_Attention_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Modal_Attention_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size, args)
        self.c_attention = Attention(hidden_size, args)
        self.out_attention = Attention(hidden_size, args)
        self.lstm0 = nn.LSTMCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.lstm1 = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.attention = Attention(hidden_size)
        self.rnn_attention = Attention(hidden_size, args)

    def lstm_decoder(self, input, h0_in, c0_in, h1_in, c1_in):
        h0_out, c0_out = self.lstm0(input, (h0_in, c0_in))
        h1_out, c1_out = self.lstm1(h0_out, (h1_in, c1_in))
        return h0_out, c0_out, h1_out, c1_out

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        encoder_out, encoder_c_out = encoder_out
        encoder_h, encoder_c = encoder_h
        if inter_modal_out is not None:
            inter_modal_out, inter_modal_c_out = inter_modal_out
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_out, inter_modal_c_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                inter_modal_c_out = inter_modal_c_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size(), inter_modal_c_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
            encoder_c, _ = self.c_attention(encoder_h, inter_modal_c_out)
            encoder_out, _ = self.out_attention(encoder_out, inter_modal_out)
        out = []
        h0 = h1 = encoder_h
        c0 = c1 = encoder_c
        for i in range(seq_len):
            h0, c0, h1, c1 = self.lstm_decoder(input[:, i, :], h0, c0, h1, c1)
            h = self.rnn_attention(h1, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c1)


class Attention_Bidirectional_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Attention_Bidirectional_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size*2, args)
        self.c_attention = Attention(hidden_size*2, args)
        self.lstm0_f = nn.LSTMCell(input_size=input_size,
                                   hidden_size=hidden_size)
        self.lstm0_b = nn.LSTMCell(input_size=input_size,
                                   hidden_size=hidden_size)
        self.lstm1_f = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.lstm1_b = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_attention = Attention(hidden_size*2, args)

    def lstm_decoder(self, input, h0_f_in, c0_f_in, h0_b_in, c0_b_in, h1_f_in, c1_f_in, h1_b_in, c1_b_in):
        reverse_input = self.reverse_input(input)
        h0_f_out, c0_f_out = self.lstm0_f(input, (h0_f_in, c0_f_in))
        h0_b_out, c0_b_out = self.lstm0_b(reverse_input, (h0_b_in, c0_b_in))
        h1_f_out, c1_f_out = self.lstm1_f(h0_f_out, (h1_f_in, c1_f_in))
        h1_b_out, c1_b_out = self.lstm1_b(h0_b_out, (h1_b_in, c1_b_in))
        return h0_f_out, c0_f_out, h0_b_out, c0_b_out, h1_f_out, c1_f_out, h1_b_out, c1_b_out
    
    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input
        
    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        encoder_out, encoder_c_out = encoder_out
        encoder_h, encoder_c = encoder_h
        if inter_modal_out is not None:
            inter_modal_out, inter_modal_c_out = inter_modal_out
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_out, inter_modal_c_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                inter_modal_c_out = inter_modal_c_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size(), inter_modal_c_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
            encoder_c, _ = self.c_attention(encoder_h, inter_modal_c_out)
        out = []
        h0_f = h1_f = encoder_h[:, :self.hidden_size]
        h0_b = h1_b = encoder_h[:, self.hidden_size:]
        c0_f = c1_f = encoder_c[:, :self.hidden_size]
        c0_b = c1_b = encoder_c[:, self.hidden_size:]        
        for i in range(seq_len):
            h0_f, c0_f, h0_b, c0_b, h1_f, c1_f, h1_b, c1_b = self.lstm_decoder(
                h0_f, c0_f, h0_b, c0_b, h1_f, c1_f, h1_b, c1_b)
            h = torch.cat([h1_f, h1_b], dim=1)
            c = torch.cat([c1_f, c1_b], dim=1)
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c)


class Modal_Attention_Bidirectional_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Modal_Attention_Bidirectional_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.h_attention = Attention(hidden_size*2, args)
        self.c_attention = Attention(hidden_size*2, args)
        self.out_attention = Attention(hidden_size*2, args)
        self.lstm0_f = nn.LSTMCell(input_size=input_size,
                                   hidden_size=hidden_size)
        self.lstm0_b = nn.LSTMCell(input_size=input_size,
                                   hidden_size=hidden_size)
        self.lstm1_f = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.lstm1_b = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_attention = Attention(hidden_size*2, args)

    def lstm_decoder(self, input, h0_f_in, c0_f_in, h0_b_in, c0_b_in, h1_f_in, c1_f_in, h1_b_in, c1_b_in):
        reverse_input = self.reverse_input(input)
        h0_f_out, c0_f_out = self.lstm0_f(input, (h0_f_in, c0_f_in))
        h0_b_out, c0_b_out = self.lstm0_b(reverse_input, (h0_b_in, c0_b_in))
        h1_f_out, c1_f_out = self.lstm1_f(h0_f_out, (h1_f_in, c1_f_in))
        h1_b_out, c1_b_out = self.lstm1_b(h0_b_out, (h1_b_in, c1_b_in))
        return h0_f_out, c0_f_out, h0_b_out, c0_b_out, h1_f_out, c1_f_out, h1_b_out, c1_b_out
    
    def reverse_input(self, input):
        reverse_input = torch.flip(input, [1])
        return reverse_input
        
    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        encoder_out, encoder_c_out = encoder_out
        encoder_h, encoder_c = encoder_h
        if inter_modal_out is not None:
            inter_modal_out, inter_modal_c_out = inter_modal_out
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_out, inter_modal_c_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                inter_modal_c_out = inter_modal_c_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size(), inter_modal_c_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
            encoder_c, _ = self.c_attention(encoder_h, inter_modal_c_out)
            encoder_out, _ = self.out_attention(encoder_out, inter_modal_out)
        out = []
        h0_f = h1_f = encoder_h[:, :self.hidden_size]
        h0_b = h1_b = encoder_h[:, self.hidden_size:]
        c0_f = c1_f = encoder_c[:, :self.hidden_size]
        c0_b = c1_b = encoder_c[:, self.hidden_size:]        
        for i in range(seq_len):
            h0_f, c0_f, h0_b, c0_b, h1_f, c1_f, h1_b, c1_b = self.lstm_decoder(
                h0_f, c0_f, h0_b, c0_b, h1_f, c1_f, h1_b, c1_b)
            h = torch.cat([h1_f, h1_b], dim=1)
            c = torch.cat([c1_f, c1_b], dim=1)
            h = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c)