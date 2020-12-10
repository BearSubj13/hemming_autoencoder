import os
print(os.getcwd())

import torch
from loss_metrics import hemming_distance
import numpy as np
from utils import mode_word, tokens2word, word2tokens
from consts import hidden_size, device, token_size
from train_val_utils import decoder_encoder_inferance
from model import EncoderRNN, DecoderRNN

from data_loading import EnglishWordDataset
from consts import *

n_samples = 1000
samples = []
distance = []
word = 'homosapience########'
temperature = 1.0

encoder = EncoderRNN(hidden_size=hidden_size, input_size=token_size)
decoder = DecoderRNN(hidden_size=hidden_size, output_size=token_size)
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.load_state_dict(torch.load('weights/english_encoder.pth'))
decoder.load_state_dict(torch.load('weights/english_decoder.pth'))
encoder.eval()
decoder.eval()

english_dataset = EnglishWordDataset('words_dictionary.json', return_token=True)

input_tokens = english_dataset.word2token(word)
input_tokens = torch.FloatTensor(input_tokens)
input_tokens = input_tokens.unsqueeze(0)
input_tokens = input_tokens.repeat(n_samples,1,1)

input_tokens = input_tokens.to(device)

result_tokens = decoder_encoder_inferance(input_tokens, encoder, decoder, temperature=temperature)

for i in range(n_samples):
  sample = english_dataset.token2word(result_tokens[i,:,:].detach().cpu())
  samples.append(sample)
  if i % 25 == 0:
    print('sample:        ', sample)
  hd = hemming_distance(sample, word)
  distance.append(hd)

mode_sample = mode_word(samples)
distance_from_mode = []
for sample in samples:
  hd = hemming_distance(sample, mode_sample)
  distance_from_mode.append(hd)

print('')
print('original word:', word)
print('mode sample:  ', mode_sample, hemming_distance(mode_sample, word))
distance_std = np.sqrt((np.array(distance)**2).mean())
print('GT std distance:', distance_std)
mode_distance_std = np.sqrt((np.array(distance_from_mode)**2).mean())
print('Mode std distance:', mode_distance_std)