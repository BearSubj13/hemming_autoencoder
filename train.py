import torch
from torch.utils.data import DataLoader
from loss_metrics import hemming_simple_loss, hemming_loss_with_size_penalty
from model import EncoderRNN, DecoderRNN
from data_loading import EnglishWordDataset
from utils import word2tokens
from consts import *
from train_val_utils import train_iteration
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='', help="name for encoder and decoder weigts file")
args = parser.parse_args()
encoder_file_name = os.path.join('weights', args.name + '_encoder.pth')
decoder_file_name = os.path.join('weights', args.name + '_decoder.pth')

print("token_size: {}".format(token_size))
print("hidden_size: {}".format(hidden_size))
print("batch_size: {}".format(batch_size))
print("max sequence length: {}".format(max_sequence_length ))
print("decay rate temperature: {0:2.2f}".format(decay_coeff_temperature))
print("decay rate lr: {0:1.2f}".format(decay_coeff_lr))
print("dataset size: {}".format(dataset_size)) 
print()

encoder = EncoderRNN(hidden_size=hidden_size, input_size=token_size)
decoder = DecoderRNN(hidden_size=hidden_size, output_size=token_size)
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)


# ground_truth_tokens = word2tokens(words)
# ground_truth_tokens = torch.FloatTensor(ground_truth_tokens)
# ground_truth_tokens = ground_truth_tokens.to(device)

#encoder.load_state_dict(torch.load(encoder_file_name))
#decoder.load_state_dict(torch.load(decoder_file_name))
encoder.train()
decoder.train()

temperature = init_temperature

mean_loss, min_loss = max_sequence_length, max_sequence_length
current_lr = learning_rate

english_dataset = EnglishWordDataset('words_dictionary.json', return_token=True)
english_dataloader_train = DataLoader(english_dataset, shuffle=True, batch_size=batch_size)


for epoch in range(n_epoch):
    loss_list = []
    #hemming distance can't be larger than the length of a string
    for i, batch_tokens in enumerate(english_dataloader_train):
        current_batch = int((temperature**2)*batch_size)
        batch_tokens = batch_tokens.to(device)
        #batch_tokens = primitive_data_loader(ground_truth_tokens, batch_size=current_batch)
        loss_f = hemming_simple_loss
        loss = train_iteration(batch_tokens, encoder, decoder, encoder_optimizer, \
                               decoder_optimizer, temperature, loss_f)
        loss_list.append(loss)
    min_loss = min(mean_loss, min_loss)
    mean_loss = sum(loss_list) / len(loss_list)
    print('epoch {0}, loss: {1:2.3f}, temperature: {2:2.2f}, lr: {3:2.5f}, batch: {4}'.\
          format(epoch, mean_loss, temperature, current_lr, current_batch))
    loss_gain = min_loss - mean_loss
    relative_loss_gain = loss_gain/(min_loss+0.000000000001)
    if (loss_gain > 0.001 or relative_loss_gain > 0.001):
        if mean_loss < min_loss_to_start_change_parameters:
          temperature = max(min_temperature, temperature/decay_coeff_temperature)
          for param in encoder_optimizer.param_groups:
            param['lr'] = max(min_lr, param['lr']/decay_coeff_lr)
          for param in decoder_optimizer.param_groups:
            param['lr'] = max(min_lr, param['lr']/decay_coeff_lr)
            current_lr = param['lr']
        #save the weights for the best results
        torch.save(encoder.state_dict(), encoder_file_name)
        torch.save(decoder.state_dict(), decoder_file_name)
        #restore weights with better metrics
    elif relative_loss_gain < - max_relative_loss_difference:
        encoder.load_state_dict(torch.load(encoder_file_name))
        decoder.load_state_dict(torch.load(decoder_file_name))







