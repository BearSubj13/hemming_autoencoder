import torch
import torch.nn as nn


#from https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=None, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers)
        if not latent_size:
            latent_size = hidden_size
        self.projector_fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.projector_fc2 = nn.Linear(hidden_size, latent_size, bias=False)

    def projector(self, input):
        output = self.projector_fc1(input)
        output = nn.functional.relu_(output)
        output = self.projector_fc2(output)
        output = nn.functional.normalize(output, p=2, dim=1)
        return output

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
        self.out = nn.Sequential(nn.Linear(3*hidden_size, int(hidden_size)), nn.ReLU(), \
                                 nn.Linear(int(hidden_size), 2*output_size), nn.ReLU(), \
                                 nn.Linear(2*output_size, output_size))
        self.attention = Attention(hidden_size)

    def forward(self, input):
        hidden, cell_state = self.lstm_cell0(input, (self.cell_state0, self.hidden_state0))
        self.hidden_state0 = hidden.detach()
        self.cell_state0 = cell_state.detach()
        #hidden, cell_state = self.lstm_cell1(self.hidden_state0, (self.cell_state1, self.hidden_state1))
        #self.hidden_state1 = hidden.detach()
        #self.cell_state1 = cell_state.detach()
        # hidden, cell_state = self.lstm_cell2(self.hidden_state1, (self.cell_state2, self.hidden_state2))
        # self.hidden_state2 = hidden.detach(hidden)
        # self.cell_state2 = cell_state.detach() 
        attention_result, _ = self.attention(query=hidden.unsqueeze(1), context=self.context)
        attention_result = attention_result.squeeze(dim=1)     
        combined_hidden = torch.cat((hidden, self.encoder_hidden, attention_result), dim=1)
        output = self.out(combined_hidden)
        self.context = torch.cat((self.context, hidden.detach().unsqueeze(1)), dim=1)  
        return output


    def initHidden(self, hidden_state, device='cpu'):
        batch_size = hidden_state.shape[1]
        self.encoder_hidden = hidden_state.squeeze(0)
        self.context = self.encoder_hidden.detach().unsqueeze(1) #for attention
        self.hidden_state0 = hidden_state.squeeze(0)
        self.cell_state0 = torch.zeros(batch_size, self.hidden_size, device=device)
        self.hidden_state1 = torch.zeros(batch_size, self.hidden_size, device=device)
        self.cell_state1 = torch.zeros(batch_size, self.hidden_size, device=device)
        # self.hidden_state2 = torch.zeros(batch_size, self.hidden_size, device=device)
        # self.cell_state2 = torch.zeros(batch_size, self.hidden_size, device=device)


class SimpleNet(nn.Module):
    '''
    predicts the hemming distance between words by their latent vectors
    '''
    def __init__(self, latent_size):
        #nn.Module.__init__(self)
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(2*latent_size, latent_size)
        self.bn_1 = nn.BatchNorm1d(latent_size)
        self.fc2 = nn.Linear(latent_size, int(latent_size/2))
        self.bn_2 = nn.BatchNorm1d(int(latent_size/2))
        self.fc3 = nn.Linear(int(latent_size/2), int(latent_size/4))
        self.bn_3 = nn.BatchNorm1d(int(latent_size/4))
        self.fc4 = nn.Linear(int(latent_size/4), 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def prediction(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.bn_1(x)

        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.bn_2(x)

        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.bn_3(x)

        x = self.fc4(x)
        x = torch.exp(x)
        return x

    def forward(self, x, y):
        input = torch.cat((x, y), dim=1)
        output = self.prediction(input).squeeze()
        return output
