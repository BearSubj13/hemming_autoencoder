import numpy as np

def gumbel_softmax(logits, temperature):
    pass

def mode_word(samples):
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