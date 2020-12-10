token_size = 28
hidden_size = 100
batch_size = 1000
max_sequence_length = 19
device = 'cuda:1'
init_temperature = 1.3
decay_coeff_temperature = 1.01
decay_coeff_lr = 1.03
n_epoch = 400
iteration_per_epoch = 500
learning_rate = 0.001
min_lr = 0.0001
min_temperature = 0.3
#if the loss current_epoch_loss > max_relative_loss_difference*min_loss, then reload weights from file
max_relative_loss_difference = 0.2
min_loss_to_start_change_parameters = 0.4
end_token = str(token_size-1)
dataset_size = 20000