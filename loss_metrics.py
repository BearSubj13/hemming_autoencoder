import torch
import torch.nn as nn

def hemming_loss_with_size_penalty(input_tokens, ground_truth):
    device = input_tokens.device
    sequences_input = torch.argmax(input_tokens.cpu(), dim=2)
    #for the end token the last position in one hot vector == 1
    #torch.max has a bug, for cuda tensor it doesn't return the index of first occurence
    max_indices_input = torch.zeros([sequences_input.shape[0]], dtype=torch.long).to(device)
    for batch in range(sequences_input.shape[0]):
        index_list = torch.nonzero(sequences_input[batch,:] == 2, as_tuple=True)
        max_indices_input[batch] = index_list[0][0] if len(index_list[0]) > 0 else -1
    max_indices_input[max_indices_input < 0] = 20
    sequences_gt = torch.argmax(ground_truth, dim=2)
    max_indices_gt = torch.zeros([sequences_input.shape[0]], dtype=torch.long).to(device)
    for batch in range(sequences_input.shape[0]):
        index_list = torch.nonzero(sequences_gt[batch,:] == 2, as_tuple=True)
        max_indices_gt[batch] = index_list[0][0] if len(index_list[0]) > 0 else -1
    size_discrepency = max_indices_gt - max_indices_input - 1
    size_discrepency[size_discrepency<0] = 0

    #char_indexes_gt = torch.where(sequences_gt!=2)[1]

    first_end_token_input = input_tokens[:,max_indices_input,:]
    first_end_token_input = torch.diagonal(first_end_token_input, dim1=0, dim2=1)
    first_end_token_input = torch.transpose(first_end_token_input, dim0=0, dim1=1)
    the_same_token_gt = ground_truth[:,max_indices_input,:]
    the_same_token_gt = torch.diagonal(the_same_token_gt, dim1=0, dim2=1)
    the_same_token_gt = torch.transpose(the_same_token_gt, dim0=0, dim1=1)

    hemming_distance_ends = torch.abs(first_end_token_input - the_same_token_gt).sum(dim=1)/2
    weighted_distance = torch.mean(hemming_distance_ends*size_discrepency)
    hemming_dist = torch.abs(input_tokens - ground_truth).sum(dim=2)/2
    hemming_dist = hemming_dist.sum(dim=1).mean()
    return weighted_distance + hemming_dist


def hemming_simple_loss(input_tokens, ground_truth):
    hemming_dist = torch.abs(input_tokens - ground_truth).sum(dim=2)/2
    hemming_dist = hemming_dist.sum(dim=1).mean()
    return hemming_dist


def hemming_distance_batch(input_tokens, ground_truth):
    hemming_dist = torch.abs(input_tokens - ground_truth).sum(dim=2)/2
    hemming_dist = hemming_dist.sum(dim=1)
    return hemming_dist

def hemming_distance(word1, word2):
    assert len(word1) == len(word2)
    counter = 0
    for x,y in zip(word1, word2):
        counter = counter + (x!=y)
    return counter
