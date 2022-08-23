from enum import IntEnum
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

torch.autograd.set_detect_anomaly(True)


class Dim(IntEnum):
    batch = 0
    seq = 1
    features = 2  # all features = n_units * unit_size


class IndyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        print("Using IndyLSTM")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = 1

        # forget gate
        self.W_f = Parameter(Tensor(input_size, hidden_size))
        self.u_f = Parameter(Tensor(hidden_size))
        self.b_f = Parameter(Tensor(hidden_size))

        # input gate
        self.W_i = Parameter(Tensor(input_size, hidden_size))
        self.u_i = Parameter(Tensor(hidden_size))
        self.b_i = Parameter(Tensor(hidden_size))

        # output gate
        self.W_o = Parameter(Tensor(input_size, hidden_size))
        self.u_o = Parameter(Tensor(hidden_size))
        self.b_o = Parameter(Tensor(hidden_size))

        # cell
        self.W_c = Parameter(Tensor(input_size, hidden_size))
        self.u_c = Parameter(Tensor(hidden_size))
        self.b_c = Parameter(Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                # nn.init.zeros_(p.data)
                nn.init.ones_(p.data)

    def forward(self, x: Tensor, init_states: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Assumes x is of shape (batch, sequence, feature)
        Assumes first dimension of each of the init_states is num_layers
        We currently only support one layer
        :return: hidden_seq -> of shape (batch, sequence, features)
        :return: Tuple(h_t, c_t) -> each of shape (layers, batch, features)
        """

        batch_size, seq_size, _ = x.size()

        if init_states is None:
            h_t, c_t = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(x.device), \
                       torch.zeros(self.n_layers, batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        hidden_seq = []

        layer = 0
        h_tl = h_t[layer]
        c_tl = c_t[layer]
        for t in range(seq_size):
            x_t = x[:, t, :]
            f_t = torch.sigmoid(torch.matmul(x_t, self.W_f) + h_tl * self.u_f + self.b_f)
            i_t = torch.sigmoid(torch.matmul(x_t, self.W_i) + h_tl * self.u_i + self.b_i)
            o_t = torch.sigmoid(torch.matmul(x_t, self.W_o) + h_tl * self.u_o + self.b_o)
            z_t = torch.tanh(torch.matmul(x_t, self.W_c) + h_tl * self.u_c + self.b_c)
            c_tl = f_t * c_tl + i_t * z_t
            h_tl = o_t * torch.tanh(c_tl)
            hidden_seq.append(h_tl.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class RNNStatefulWrapper(nn.Module):
    """
    This class takes care of keeping track of the hidden states after every call.
    Does not work for `AsNet`!
    Or for `EVNN`!
    """

    def __init__(self, rnn: nn.Module):
        super(RNNStatefulWrapper, self).__init__()
        self.rnn = rnn

        self.n_layers = 1
        self.rnn_out = None

        self.initHidden()

    ## NOTE: Needed to reinitialize hidden after every batch
    def initHidden(self):
        self.hidden = None

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    def forward(self, all_inputs):
        self.rnn_out, self.hidden = self.rnn(all_inputs, self.hidden)
        return self.rnn_out, self.hidden

    def get_last_output(self):
        """
        Returns exactly the return value that the forward function returned this step
        Read-only function
        """
        return self.rnn_out, self.hidden


class RNNReadoutWrapper(nn.Module):
    """
    This class puts a readout on top of the passed in RNN.
    For pytorch LSTM, the outputs contain the hidden states of only the last layer.
    BUT DOESN"T USE READOUT. HAS TO BE DONE OUTSIDE.
    """

    def __init__(self, rnn: RNNStatefulWrapper, output_size: int):
        super(RNNReadoutWrapper, self).__init__()
        self.rnn = rnn
        self.output_size = output_size

        # self.hidden2out = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Sigmoid())
        # NOTE: That there is no sigmoid here, since it's applied at the loss function
        # self.hidden2out = nn.Sequential(nn.Linear(hidden_size, output_size))
        self.hidden2out = nn.Linear(self.rnn.hidden_size, self.output_size)
        # self.out_vals = None

    ## NOTE: Needed to reinitialize hidden after every batch
    # def initHidden(self):
    #     self.rnn.initHidden()

    def forward(self, all_inputs):
        rnn_out, hidden = self.rnn(all_inputs)
        # out_vals = self.hidden2out(rnn_out)
        # self.out_vals = out_vals

        return rnn_out, hidden

    # def get_last_output(self):
    #     """
    #     Returns exactly the return value that the forward function returned this step
    #     Read-only function
    #     """
    #     return (self.out_vals, *self.rnn.get_last_output())
