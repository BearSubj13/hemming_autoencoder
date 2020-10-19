import numpy as np
from consts import token_size, end_token, max_sequence_length
from utils import one_hot


def random_binary_word(probability_of_01= (token_size-1)*[1/(token_size-1)], probability_of_end=0.1, max_size=max_sequence_length):
    binary_word = ""
    tokens = []
    state = np.random.choice(np.arange(0, token_size-1), p=probability_of_01)
    probability_state = np.array(probability_of_01)*(1-probability_of_end)
    probability_state = np.append(probability_state, probability_of_end)
    i = 0
    for i in range(max_size):
        binary_word += str(state)
        tokens.append(one_hot(token_size, state))
        if binary_word[-1] == end_token:
            state = token_size - 1
        else:
            state = np.random.choice(np.arange(0, token_size), p=probability_state)
    binary_word = binary_word + end_token
    tokens.append(one_hot(token_size, token_size-1))
    return tokens, binary_word

def primitive_data_loader(ground_truth_tokens, batch_size=1):
    rand_number = np.random.choice(np.arange(ground_truth_tokens.shape[0]), size=batch_size)
    mini_batch = ground_truth_tokens[rand_number,:,:]
    return mini_batch

#ground_truth_tokens
words =  ['0101001101010010'+5*end_token,
          '1100100011100110011'+2*end_token,
          '00010100000001101100'+end_token,
          '1110111000100'+8*end_token,
          '10010110101001010'+4*end_token]
words = words + [random_binary_word(probability_of_end=0.02)[1] for i in range(1000)]
words = 100*words