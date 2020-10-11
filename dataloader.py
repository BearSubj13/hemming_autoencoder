import numpy as np
import torch

def one_hot(total_class_numb, state):
    one_hot_emb = np.zeros(total_class_numb)
    one_hot_emb[state] = 1
    return one_hot_emb

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

def tokens2word(token_list):
    word = ''
    for token in token_list:
        state = np.argmax(token)
        word += str(state)
    return word

def word2tokens(binary_word, total_class_numb=3):
    tokens = np.zeros([len(binary_word), total_class_numb])
    for i,symbol in enumerate(binary_word):
        tokens[i,:] = one_hot(total_class_numb, int(symbol))
    return tokens

def tensor2word(token_tensor):
    word = ''
    for i in range(token_tensor.shape[0]):
        state = torch.argmax(token_tensor[i, :])
        state = int(state.item())
        word += str(state)
    return word
