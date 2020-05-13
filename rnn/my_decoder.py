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
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size, args)
        self.gru = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        print(self.gru)

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        if inter_modal_out is not None:
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
        h_in = [encoder_h]*self.num_layers
        out = []
        for i in range(seq_len):
            h = input[:, i, :]
            for j, layer in enumerate(self.gru):
                h = layer(h, h_in[j])
                h_in[j] = h
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class Bidirectional_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Bidirectional_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size*2, args)
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

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        if inter_modal_out is not None:
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
        h_f_in = [encoder_h[:, :self.hidden_size]]*self.num_layers
        h_b_in = [encoder_h[:, self.hidden_size:]]*self.num_layers
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


class Attention_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Attention_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size, args)
        self.gru = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        self.rnn_attention = Attention(hidden_size, args)
        print(self.gru)

    def forward(self, input, encoder_out, encoder_h, inter_modal_out=None):
        seq_len = input.size(1)
        if inter_modal_out is not None:
            if input.size(0) != inter_modal_out.size(0):
                # test batchでmodal間のサイズが違う分のinter_modal_outの複製(repeat_interleave, repeat, tile, expand)
                # To Do
                inter_modal_out = inter_modal_out.repeat_interleave(input.size(0), dim=0)
                print(input.size(), inter_modal_out.size())
            encoder_h, _ = self.h_attention(encoder_h, inter_modal_out)
        h_in = [encoder_h]*self.num_layers
        out = []
        for i in range(seq_len):
            h = input[:, i, :]
            for j, layer in enumerate(self.gru):
                h = layer(h, h_in[j])
                h_in[j] = h
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class Attention_Bidirectional_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Attention_Bidirectional_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size*2, args)
        self.gru_f = nn.ModuleList()
        self.gru_b = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru_f.append(nn.GRUCell(input_size, hidden_size))
            self.gru_b.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        self.rnn_attention = Attention(hidden_size*2, args)
        print(self.gru_f)
        print(self.gru_b)
        
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
        h_f_in = [encoder_h[:, :self.hidden_size]]*self.num_layers
        h_b_in = [encoder_h[:, self.hidden_size:]]*self.num_layers
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
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class Modal_Attention_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Modal_Attention_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size, args)
        self.out_attention = Attention(hidden_size, args)
        self.gru = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        self.rnn_attention = Attention(hidden_size, args)
        print(self.gru)

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
        h_in = [encoder_h]*self.num_layers
        out = []
        for i in range(seq_len):
            h = input[:, i, :]
            for j, layer in enumerate(self.gru):
                h = layer(h, h_in[j])
                h_in[j] = h
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class Modal_Attention_Bidirectional_GRU_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Modal_Attention_Bidirectional_GRU_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size*2, args)
        self.out_attention = Attention(hidden_size*2, args)
        self.gru_f = nn.ModuleList()
        self.gru_b = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru_f.append(nn.GRUCell(input_size, hidden_size))
            self.gru_b.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        self.rnn_attention = Attention(hidden_size*2, args)
        
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
        h_f_in = [encoder_h[:, :self.hidden_size]]*self.num_layers
        h_b_in = [encoder_h[:, self.hidden_size:]]*self.num_layers
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
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, h


class LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size, args)
        self.c_attention = Attention(hidden_size, args)
        self.lstm = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        print(self.lstm)

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
        h_in = [encoder_h]*self.num_layers
        c_in = [encoder_c]*self.num_layers
        out = []
        for i in range(seq_len):
            h = input[:, i, :]
            for j, layer in enumerate(self.lstm):
                h, c = layer(h, (h_in[j], c_in[j]))
                h_in[j] = h
                c_in[j] = c
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c)


class Bidirectional_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Bidirectional_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size*2, args)
        self.c_attention = Attention(hidden_size*2, args)
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
        h_f_in = [encoder_h[:, :self.hidden_size]]*self.num_layers
        h_b_in = [encoder_h[:, self.hidden_size:]]*self.num_layers
        c_f_in = [encoder_c[:, :self.hidden_size]]*self.num_layers
        c_in_b = [encoder_c[:, self.hidden_size:]]*self.num_layers
        out = []    
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
        out = torch.stack(out, dim=1)
        return out, (h, c)


class Attention_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Attention_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size, args)
        self.c_attention = Attention(hidden_size, args)
        self.lstm = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        self.rnn_attention = Attention(hidden_size, args)
        print(self.lstm)

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
        h_in = [encoder_h]*self.num_layers
        c_in = [encoder_c]*self.num_layers
        out = []
        for i in range(seq_len):
            h = input[:, i, :]
            for j, layer in enumerate(self.lstm):
                h, c = layer(h, (h_in[j], c_in[j]))
                h_in[j] = h
                c_in[j] = c
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c)


class Attention_Bidirectional_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Attention_Bidirectional_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size*2, args)
        self.c_attention = Attention(hidden_size*2, args)
        self.lstm_f = nn.ModuleList()
        self.lstm_b = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm_f.append(nn.LSTMCell(input_size, hidden_size))
            self.lstm_b.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        self.rnn_attention = Attention(hidden_size*2, args)
        print(self.lstm_f)
        print(self.lstm_b)
    
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
        h_f_in = [encoder_h[:, :self.hidden_size]]*self.num_layers
        h_b_in = [encoder_h[:, self.hidden_size:]]*self.num_layers
        c_f_in = [encoder_c[:, :self.hidden_size]]*self.num_layers
        c_in_b = [encoder_c[:, self.hidden_size:]]*self.num_layers
        out = []    
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
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c)


class Modal_Attention_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Modal_Attention_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size, args)
        self.c_attention = Attention(hidden_size, args)
        self.out_attention = Attention(hidden_size, args)
        self.lstm = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        self.rnn_attention = Attention(hidden_size, args)
        print(self.lstm)

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
        h_in = [encoder_h]*self.num_layers
        c_in = [encoder_c]*self.num_layers
        out = []
        for i in range(seq_len):
            h = input[:, i, :]
            for j, layer in enumerate(self.lstm):
                h, c = layer(h, (h_in[j], c_in[j]))
                h_in[j] = h
                c_in[j] = c
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c)


class Modal_Attention_Bidirectional_LSTM_Decoder(nn.Module):
    def __init__(self, batch_size, input_size, args):
        super(Modal_Attention_Bidirectional_LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.h_attention = Attention(hidden_size*2, args)
        self.c_attention = Attention(hidden_size*2, args)
        self.out_attention = Attention(hidden_size*2, args)
        self.lstm_f = nn.ModuleList()
        self.lstm_b = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm_f.append(nn.LSTMCell(input_size, hidden_size))
            self.lstm_b.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        self.rnn_attention = Attention(hidden_size*2, args)
        print(self.lstm_f)
        print(self.lstm_b)
    
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
        h_f_in = [encoder_h[:, :self.hidden_size]]*self.num_layers
        h_b_in = [encoder_h[:, self.hidden_size:]]*self.num_layers
        c_f_in = [encoder_c[:, :self.hidden_size]]*self.num_layers
        c_in_b = [encoder_c[:, self.hidden_size:]]*self.num_layers
        out = []    
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
            h, _ = self.rnn_attention(h, encoder_out)
            out.append(h)
        out = torch.stack(out, dim=1)
        return out, (h, c)