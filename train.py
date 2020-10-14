import torch
import torch.nn.functional as F
from model import EncoderRNN, DecoderRNN
from dataloader import random_binary_word, primitive_data_loader
from loss_metrics import hemming_distance, hemming_simple_loss
import numpy as np
from utils import mode_word, tokens2word, tensor2word, word2tokens

device = 'cuda'
token_size = 3
hidden_size = 130
batch_size = 4
max_sequence_length = 20

encoder = EncoderRNN(hidden_size=hidden_size, input_size=token_size)
decoder = DecoderRNN(hidden_size=hidden_size, output_size=token_size)
encoder = encoder.to(device)
decoder = decoder.to(device)

gumbel = torch.distributions.Gumbel(0, 1)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

def decoder_encoder_inferance(encoder, decoder, binary_word, device='cpu', batch_size=1, temperature=1):
  encoder.eval()
  decoder.eval()
  ground_truth_tokens = word2tokens(binary_word)
  ground_truth_tokens = torch.FloatTensor(ground_truth_tokens)
  #ground_truth_tokens = ground_truth_tokens.unsqueeze(1)
  ground_truth_tokens = ground_truth_tokens.to(device)
  hidden_state = encoder.initHidden(device)
  _, hidden_state = encoder(ground_truth_tokens, hidden_state)
  tokens_predicted = ground_truth_tokens[:, 0, :].unsqueeze(1)
  decoder.initHidden(hidden_state, device)
  for k in range(max_sequence_length):
    output = decoder(tokens_predicted[:,-1,:])
    #output_gumbel = output+ gumbel.sample([batch_size, token_size]).to(device)
    #output_gumbel = output_gumbel/temperature
    output_gumbel = F.gumbel_softmax(output, tau=temperature, hard=True)
    end_tokens = torch.argmax(tokens_predicted[:, -1, :], dim=1) == 2
    output_gumbel[end_tokens,:] = tokens_predicted[:, -1, :][end_tokens].detach()
    tokens_predicted = torch.cat((tokens_predicted, output_gumbel.unsqueeze(1)), dim=1)

  return tensor2word(tokens_predicted[0,:,:])

def train_iteration(ground_truth_tokens, encoder, decoder, encoder_optimizer, \
                    decoder_optimizer, temperature, loss_function):
  hidden_state = encoder.initHidden(device, batch_size)
  _, hidden_state = encoder(ground_truth_tokens, hidden_state)
  tokens_predicted = ground_truth_tokens[:, 0, :].unsqueeze(1)
  decoder.initHidden(hidden_state, device=device)
  for k in range(max_sequence_length):
    output = decoder(tokens_predicted[:, -1, :])
    #output_gumbel = output + gumbel.sample([batch_size, token_size]).to(device)
    output_gumbel = F.gumbel_softmax(output, tau=temperature, hard=True)
    end_tokens = torch.argmax(tokens_predicted[:, -1, :], dim=1) == 2
    #if the previous letter is the end token, the next token should be end too
    output_gumbel[end_tokens, :] = tokens_predicted[:, -1, :][end_tokens].detach()
    tokens_predicted = torch.cat((tokens_predicted, output_gumbel.unsqueeze(1)), dim=1)

  loss = loss_function(tokens_predicted, ground_truth_tokens)

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()
  return loss.item()



#ground_truth_tokens, word = random_binary_word()
word =  '010100110101001022222'
word1 = '110010001110011001122'
word2 = '000101000000011011002'
word3 = '111011100010022222222'
word4 = '100101101010010102222'
ground_truth_tokens = word2tokens([word, word1, word2, word3, word4])
ground_truth_tokens = torch.FloatTensor(ground_truth_tokens)
ground_truth_tokens = ground_truth_tokens.to(device)
train_mode = True

if train_mode:
    encoder.train()
    decoder.train()

    temperature = 1.6
    decay_coeff = 1.06
    mean_loss, min_loss = max_sequence_length, max_sequence_length
    for epoch in range(40):
      # for param in encoder_optimizer.param_groups:
      #   param['lr'] = param['lr']/(epoch+1)
      # for param in decoder_optimizer.param_groups:
      #   param['lr'] = param['lr']/(epoch+1)
      loss_list = []
      #hemming distance can't be larger than the length of a string
      for i in range(300):
        batch_tokens = primitive_data_loader(ground_truth_tokens, batch_size=batch_size)
        loss = train_iteration(batch_tokens, encoder, decoder, encoder_optimizer, \
                               decoder_optimizer, temperature, hemming_simple_loss)
        loss_list.append(loss)
      min_loss = min(mean_loss, min_loss)
      mean_loss = sum(loss_list) / len(loss_list)
      print('epoch {0}, loss: {1:2.2f}, temperature: {2:2.2f}'.format(epoch, mean_loss, temperature))
      loss_gain = min_loss-mean_loss
      relative_loss_gain = loss_gain/mean_loss
      if (loss_gain > 0.5  or loss_gain > 0.05) and mean_loss < 5.5:
        temperature = temperature/decay_coeff

    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')
else:
    encoder.load_state_dict(torch.load('encoder.pth'))
    decoder.load_state_dict(torch.load('decoder.pth'))

word=word2
n_samples = 300
samples = []
distance = []
for i in range(n_samples):
  sample = decoder_encoder_inferance(encoder, decoder, [word], device, 1, temperature=0.8)
  samples.append(sample)
  if i % 25 == 0:
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

