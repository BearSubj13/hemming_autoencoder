import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers)

    def forward(self, input):
        output, hidden = self.lstm(input, (self.c0, self.h0))
        return output, hidden

    def initHidden(self, device='cpu', batch_size=1):
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell0 = nn.LSTMCell(output_size, hidden_size)
        self.lstm_cell1 = nn.LSTMCell(hidden_size, hidden_size)
        #self.lstm_cell2 = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Sequential(nn.Linear(2*hidden_size, int(hidden_size)), nn.ReLU(), \
                                 nn.Linear(int(hidden_size), 2*output_size), nn.ReLU(), \
                                 nn.Linear(2*output_size, output_size))

    def forward(self, input):
        hidden, cell_state = self.lstm_cell0(input, (self.cell_state0, self.hidden_state0))
        self.hidden_state0 = hidden.detach()
        self.cell_state0 = cell_state.detach()
        hidden, cell_state = self.lstm_cell1(self.hidden_state0, (self.cell_state1, self.hidden_state1))
        self.hidden_state1 = hidden.detach()
        self.cell_state1 = cell_state.detach()
        # hidden, cell_state = self.lstm_cell2(self.hidden_state1, (self.cell_state2, self.hidden_state2))
        # self.hidden_state2 = hidden.detach()
        # self.cell_state2 = cell_state.detach()
        combined_hidden = torch.cat((hidden, self.encoder_hidden), dim=1)
        output = self.out(combined_hidden)
        return output

    def initHidden(self, hidden_state, device='cpu'):
        batch_size = hidden_state.shape[1]
        self.encoder_hidden = hidden_state.squeeze(0)
        self.hidden_state0 = hidden_state.squeeze(0)
        self.cell_state0 = torch.zeros(batch_size, self.hidden_size, device=device)
        self.hidden_state1 = torch.zeros(batch_size, self.hidden_size, device=device)
        self.cell_state1 = torch.zeros(batch_size, self.hidden_size, device=device)
        # self.hidden_state2 = torch.zeros(batch_size, self.hidden_size, device=device)
        # self.cell_state2 = torch.zeros(batch_size, self.hidden_size, device=device)