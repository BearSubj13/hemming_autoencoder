import torch
from loss_metrics import hemming_simple_loss, hemming_loss_with_size_penalty
from model import EncoderRNN, DecoderRNN
from dataloader import words, primitive_data_loader
from utils import  word2tokens
from consts import *
from train_val_utils import train_iteration



encoder = EncoderRNN(hidden_size=hidden_size, input_size=token_size)
decoder = DecoderRNN(hidden_size=hidden_size, output_size=token_size)
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)


ground_truth_tokens = word2tokens(words)
ground_truth_tokens = torch.FloatTensor(ground_truth_tokens)
ground_truth_tokens = ground_truth_tokens.to(device)

encoder.train()
decoder.train()

temperature = init_temperature

mean_loss, min_loss = max_sequence_length, max_sequence_length
current_lr = learning_rate

for epoch in range(n_epoch):
    loss_list = []
    #hemming distance can't be larger than the length of a string
    for i in range(iteration_per_epoch):
        current_batch = int((temperature**3)*batch_size)
        batch_tokens = primitive_data_loader(ground_truth_tokens, batch_size=current_batch)
        if i % 50 == 0:
            loss_f = hemming_simple_loss
        else:
            loss_f = hemming_simple_loss
        loss = train_iteration(batch_tokens, encoder, decoder, encoder_optimizer, \
                               decoder_optimizer, temperature, loss_f)
        loss_list.append(loss)
    min_loss = min(mean_loss, min_loss)
    mean_loss = sum(loss_list) / len(loss_list)
    print('epoch {0}, loss: {1:2.3f}, temperature: {2:2.2f}, lr: {3:2.5f}'.\
          format(epoch, mean_loss, temperature, current_lr))
    loss_gain = min_loss - mean_loss
    relative_loss_gain = loss_gain/(mean_loss+0.000000000001)
    if (loss_gain > 0.01 or relative_loss_gain > 0.005) and mean_loss < 4.0:
        temperature = max(0.4,temperature/decay_coeff_temperature)
        for param in encoder_optimizer.param_groups:
          param['lr'] = param['lr']/decay_coeff_lr
        for param in decoder_optimizer.param_groups:
          param['lr'] = param['lr']/decay_coeff_lr
          current_lr = param['lr']
        #save the weights for the best results
        torch.save(encoder.state_dict(), 'encoder_deleteme.pth')
        torch.save(decoder.state_dict(), 'decoder_deleteme.pth')





