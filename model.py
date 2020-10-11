import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, device='cpu'):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(output_size, hidden_size)
        #torch.nn.init.xavier_normal_(self.lstm_cell)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        hidden, cell_state = self.lstm_cell(input, (self.cell_state, self.hidden_state))
        self.hidden_state = hidden.detach()
        self.cell_state = cell_state.detach()
        output = self.out(hidden)
        return output

    def initHidden(self, hidden_state, device='cpu'):
        self.hidden_state = hidden_state.squeeze(1)
        self.cell_state = torch.zeros(1, self.hidden_size, device=device)
        return