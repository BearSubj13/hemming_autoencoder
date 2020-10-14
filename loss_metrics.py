import torch
import torch.nn as nn

def soft_hemming_loss(input_tokens, ground_truth):
    ce = nn.CrossEntropyLoss()
    sequences_input = torch.argmax(input_tokens, dim=2)
    _, max_indices_input = torch.max(sequences_input, dim=1)
    max_indices_input[max_indices_input < 0] = 21
    sequences_gt = torch.argmax(ground_truth, dim=2)
    _, max_indices_gt = torch.max(sequences_gt, dim=1)
    size_discrepency = max_indices_gt - max_indices_input
    size_discrepency[size_discrepency<0] = 0

    sequence_not_end_positions = sequences_input != 2
    char_indexes_gt = torch.where(sequences_gt!=2)[1]
    #gt_before_end =
    for bch in range(sequence_not_end_positions.shape[0]):
        sequences_gt[bch, char_indexes_gt]
    loss = ce(input_tokens[sequence_not_end_positions], char_indexes_gt)
    first_end_token_input = input_tokens[:,max_indices_input,:]
    first_end_token_input = torch.diagonal(first_end_token_input, dim1=0, dim2=1)
    first_end_token_input = torch.transpose(first_end_token_input, dim0=0, dim1=1)
    the_same_token_gt = ground_truth[:,max_indices_input,:]
    the_same_token_gt = torch.diagonal(the_same_token_gt, dim1=0, dim2=1)
    the_same_token_gt = torch.transpose(the_same_token_gt, dim0=0, dim1=1)
    bce_weighted = nn.CrossEntropyLoss(reduce=False)
    end_tokens_loss = bce_weighted(first_end_token_input, the_same_token_gt)
    loss = loss + end_tokens_loss
    return loss


def soft_hemming_simlpe_loss(input_tokens, ground_truth):
    ce = nn.CrossEntropyLoss()
    sequences_gt = torch.argmax(ground_truth, dim=2)
    reshaped_tokens = input_tokens.transpose(1,2)
    loss = ce(reshaped_tokens, sequences_gt)
    return loss

def hemming_simple_loss(input_tokens, ground_truth):
    hemming_dist = torch.abs(input_tokens - ground_truth).sum(dim=2)/2
    hemming_dist = hemming_dist.sum(dim=1).mean()
    return hemming_dist


def hemming_distance(word1, word2):
    assert len(word1) == len(word2)
    counter= 0
    for x,y in zip(word1, word2):
        counter = counter + (x!=y)
    return counter
