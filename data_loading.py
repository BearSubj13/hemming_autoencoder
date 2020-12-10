import numpy as np
from consts import token_size, end_token, max_sequence_length, dataset_size 
from utils import one_hot
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import string
from sklearn.model_selection import train_test_split


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
# words =  ['0101001101010010'+5*end_token,
#           '1100100011100110011'+2*end_token,
#           '00010100000001101100'+end_token,
#           '1110111000100'+8*end_token,
#           '10010110101001010'+4*end_token]
# words = words + [random_binary_word(probability_of_end=0.02)[1] for i in range(dataset_size)]
# words = 10*words

class EnglishWordDataset(Dataset):
    def __init__(self, words_file_path, word_max_size, return_token=True, transform=None):
        with open(words_file_path) as json_file:
            dictionary = json.load(json_file)
        self.words_list = list(dictionary.keys())
        self.word_max_size = word_max_size
        self.alphabet = list(string.ascii_lowercase) + ['-', '#']
        self.alphabet_size = len(self.alphabet)
        self.alphabet_dict = dict(zip(self.alphabet, list(range(self.alphabet_size))))
        self.transform = transform
        self.return_token = return_token


    def word2token(self, word):
        token = torch.zeros([self.word_max_size + 1, self.alphabet_size])
        for i, symbol in enumerate(word):
            token[i, self.alphabet_dict[symbol]] = 1
        # for i in range(len(word)+1, self.word_max_size):
        #     token[i, -1] = 1
        return token


    def token2word(self, token_tensor):
        word = ''
        for i in range(token_tensor.shape[0]):
            state = torch.argmax(token_tensor[i, :])
            state = self.alphabet[state]
            word += state
        return word


    def __len__(self):
        return len(self.words_list)

    def __getitem__(self, index):
        word = self.words_list[index]
        word = word[:self.word_max_size] #if word exceeds max_size
        number_of_empty_positions_in_word = self.word_max_size - len(word) + 1
        word = word + "".join(number_of_empty_positions_in_word*['#'])
        if self.transform:
            word1 = self.transform(word)
            word2 = self.transform(word)
            if self.return_token:
                token1 = self.word2token(word1)
                token2 = self.word2token(word2)
                return token1, token2
        else:
            if self.return_token:
                token = self.word2token(word)
                return token
            else:
                return word

    def __add_transformation__(self, transfomation):
        self.transform = transfomation


class HemmingTransform(object):
    def __init__(self, alphabet):
        super().__init__()
        self.alphabet = alphabet

    def __call__(self, input_word):
        word_length = len(input_word)
        first_end_token_position = input_word.find('#')
        random_position = torch.randint(0, min(word_length, first_end_token_position), [1])
        random_letter = self.alphabet[torch.randint(0, len(self.alphabet)-1, [1])]
        output_word = list(input_word)
        output_word[random_position] = random_letter
        output_word = "".join(output_word)
        return output_word


def train_val_split(dataset, val_split=0.9, fix_split_state=None):
    if fix_split_state:
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=fix_split_state)
    else:
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets



if __name__ == "__main__":
    english_dataset = EnglishWordDataset('words_dictionary.json', return_token=True)
    transformation = HemmingTransform(english_dataset.alphabet)
    english_dataset.__add_transformation__(transformation)

    x = english_dataset[200000]
    x = english_dataset.token2word(x)
    print(x)