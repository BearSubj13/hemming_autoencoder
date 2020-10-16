import numpy as np
import torch

def gumbel_softmax(logits, temperature):
    pass

def one_hot(total_class_numb, state):
  one_hot_emb = np.zeros(total_class_numb)
  one_hot_emb[state] = 1
  return one_hot_emb

# mode of each letter in a word
def mode_word_elementwise(samples):
  binary_word = ''
  np_char = np.empty(shape=[len(samples), len(samples[0])])
  for i in range(len(samples)):
    np_char[i, :] = list(samples[i])
  for j in range(np_char.shape[1]):
    char_list = list(np_char[:,j])
    count_dict = dict((x,char_list.count(x)) for x in set(char_list))
    frequent_letter = max(count_dict, key=count_dict.get)
    binary_word += str(int(frequent_letter))
  return binary_word


def mode_word(samples):
    count_dict = dict((x, samples.count(x)) for x in set(samples))
    frequentest_word = max(count_dict, key=count_dict.get)
    return frequentest_word


def tokens2word(token_list):
    word = ''
    for token in token_list:
      state = np.argmax(token)
      word += str(state)
    return word


def word2tokens(binary_word, total_class_numb=3):
  batch_size = len(binary_word) if type(binary_word) is list else 1
  if type(binary_word) is not list:
      tokens = np.zeros([len(binary_word), total_class_numb])
      for i, symbol in enumerate(binary_word):
        tokens[i, :] = one_hot(total_class_numb, int(symbol))
      return tokens
  else:
      tokens = np.zeros([batch_size, len(binary_word[0]), total_class_numb])
      for k in range(batch_size):
        for i, symbol in enumerate(binary_word[k]):
          tokens[k, i, :] = one_hot(total_class_numb, int(symbol))
      return tokens


def tensor2word(token_tensor):
  word = ''
  for i in range(token_tensor.shape[0]):
    state = torch.argmax(token_tensor[i, :])
    state = int(state.item())
    word += str(state)
  return word






