import torch
import torch.nn as nn

def soft_hemming_loss(input_tokens, ground_truth):
    bce = nn.BCEWithLogitsLoss()
    # if input_tokens.shape[0] > ground_truth.shape[0]:
    #     length_delta = input_tokens.shape[0] - ground_truth.shape[0]
    #     input_tokens = input_tokens[:ground_truth.shape[0],:]
    # else:
    #     length_delta = ground_truth.shape[0] - input_tokens.shape[0]
    #     ground_truth = ground_truth[:input_tokens.shape[0],:]
    loss = bce(input_tokens, ground_truth)
    return loss


def hemming_distance(word1, word2):
    assert len(word1) == len(word2)
    counter= 0
    for x,y in zip(word1, word2):
        counter = counter + (x!=y)
    return counter
