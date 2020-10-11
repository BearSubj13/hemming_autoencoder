import torch
from model import EncoderRNN, DecoderRNN
from dataloader import random_binary_word, tokens2word, tensor2word, word2tokens
from loss_metrics import soft_hemming_loss, hemming_distance
import numpy as np
from utils import mode_word

device = 'cpu'
token_size = 3
hidden_size = 100

#ground_truth_tokens, word = random_binary_word()
word = '010100110100001022222'
ground_truth_tokens = word2tokens(word)
ground_truth_tokens = torch.FloatTensor(ground_truth_tokens)
ground_truth_tokens = ground_truth_tokens.unsqueeze(1)
ground_truth_tokens = ground_truth_tokens.to(device)

encoder = EncoderRNN(hidden_size=hidden_size, input_size=token_size)
decoder = DecoderRNN(hidden_size=hidden_size, output_size=token_size)
encoder = encoder.to(device)
decoder = decoder.to(device)

gumbel = torch.distributions.Gumbel(0, 1)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

def decoder_encoder_inferance(encoder, decoder, binary_word, device='cpu'):
  encoder.eval()
  decoder.eval()
  ground_truth_tokens = word2tokens(binary_word)
  ground_truth_tokens = torch.FloatTensor(ground_truth_tokens)
  ground_truth_tokens = ground_truth_tokens.unsqueeze(1)
  ground_truth_tokens = ground_truth_tokens.to(device)
  hidden_state = encoder.initHidden(device)
  _, hidden_state = encoder(ground_truth_tokens, hidden_state)
  tokens_predicted = ground_truth_tokens[0, :, :]
  decoder.initHidden(hidden_state, device)
  for k in range(20):
    if torch.argmax(tokens_predicted[-1]) == 2:
      end_token = tokens_predicted[-1].detach().unsqueeze(0)
      tokens_predicted = torch.cat((tokens_predicted, end_token), dim=0)
      continue
    output = decoder(tokens_predicted[-1].unsqueeze(0))
    output_gumbel = 2*(output + gumbel.sample([token_size]).to(device))
    tokens_predicted = torch.cat((tokens_predicted, output_gumbel), dim=0)

  return tensor2word(tokens_predicted)


encoder.train()
decoder.train()
for i in range(1000):
  hidden_state = encoder.initHidden(device)
  _, hidden_state = encoder(ground_truth_tokens, hidden_state)
  tokens_predicted = ground_truth_tokens[0, :, :]
  decoder.initHidden(hidden_state, device=device)
  for k in range(20):
    if torch.argmax(tokens_predicted[-1]) == 2:
      end_token = tokens_predicted[-1].unsqueeze(0)
      tokens_predicted = torch.cat((tokens_predicted, end_token.detach()), dim=0)
      continue
    output = decoder(tokens_predicted[-1].unsqueeze(0))
    output_gumbel = output + gumbel.sample([token_size]).to(device)
    #output_gumbel = torch.softmax(output_gumbel, dim=1)
    # end_token = torch.softmax(end_token, dim=1)
    tokens_predicted = torch.cat((tokens_predicted, output_gumbel), dim=0)

  loss = soft_hemming_loss(tokens_predicted, ground_truth_tokens.squeeze())
  if i % 100 == 0:
    print(loss)
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()

n_samples = 1000
samples = []
distance = []
for i in range(n_samples):
  sample = decoder_encoder_inferance(encoder, decoder, word, device)
  samples.append(sample)
  if i % 50 == 0:
    print(sample)
  hd = hemming_distance(sample, word)
  distance.append(hd)

mode_sample = mode_word(samples)
distance_from_mode = []
for sample in samples:
  hd = hemming_distance(sample, mode_sample)
  distance_from_mode.append(hd)

print('original word:', word)
print('mode sample:  ', mode_sample, hemming_distance(mode_sample, word))
distance_std = np.sqrt((np.array(distance)**2).mean())
print('GT std distance:', distance_std)
mode_distance_std = np.sqrt((np.array(distance_from_mode)**2).mean())
print('Mode std distance:', mode_distance_std)

