import torch
import torch.nn.functional as F
from consts import max_sequence_length, token_size


def decoder_encoder_inferance(input_tokens, encoder, decoder,\
                     temperature=1, max_sequence_length=max_sequence_length):

  batch_size = input_tokens.shape[0]
  device = input_tokens.device

  encoder.initHidden(device, batch_size)
  _, hidden_state = encoder(input_tokens)
  if encoder.num_layers > 1:
    hidden_state = hidden_state[0][1,:,:].unsqueeze(dim=0)
  else:
    hidden_state = hidden_state[0]

  tokens_predicted = input_tokens[:, 0, :].unsqueeze(1)
  decoder.initHidden(hidden_state, device)

  for k in range(max_sequence_length):
    output = decoder(tokens_predicted[:,-1,:])
    output_gumbel = F.gumbel_softmax(output, tau=temperature, hard=True)
    end_tokens = torch.argmax(tokens_predicted[:, -1, :], dim=1) == token_size-1
    output_gumbel[end_tokens,:] = tokens_predicted[:, -1, :][end_tokens].detach()
    tokens_predicted = torch.cat((tokens_predicted, output_gumbel.unsqueeze(1)), dim=1)

  return tokens_predicted



def train_iteration(ground_truth_tokens, encoder, decoder, encoder_optimizer, \
                    decoder_optimizer, temperature, loss_function, \
                    max_sequence_length=max_sequence_length):

  tokens_predicted = decoder_encoder_inferance(ground_truth_tokens, encoder, decoder, temperature, max_sequence_length)
  loss = loss_function(tokens_predicted, ground_truth_tokens)

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()
  return loss.item()