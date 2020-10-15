import numpy as np
import torch
from utils import one_hot


def random_binary_word(probability_of_01=(0.5, 0.5), probability_of_end=0.1, max_size=20):
    binary_word = ""
    tokens = []
    state = np.random.choice(np.arange(0, 2), p=probability_of_01)
    probability_state = np.array(probability_of_01)*(1-probability_of_end)
    probability_state = np.append(probability_state, probability_of_end)
    i = 0
    for i in range(max_size):
        binary_word += str(state)
        tokens.append(one_hot(3, state))
        if binary_word[-1] == '2':
            state = 2
        else:
            state = np.random.choice(np.arange(0, 3), p=probability_state)
    tokens.append(one_hot(3, 2))
    return tokens, binary_word

def primitive_data_loader(ground_truth_tokens, batch_size=1):
    rand_number = np.random.choice(np.arange(ground_truth_tokens.shape[0]), size=batch_size)
    mini_batch = ground_truth_tokens[rand_number,:,:]
    return mini_batch

#ground_truth_tokens
words =  ['010100110101001022222',
          '110010001110011001122',
          '000101000000011011002',
          '111011100010022222222',
          '100101101010010102222']
