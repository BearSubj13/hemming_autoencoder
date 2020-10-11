import torch
import torch.nn as nn

def soft_hemming_loss(input_tokens, ground_truth):
    bce = nn.BCEWithLogitsLoss()
    sequences_input = torch.argmax(input_tokens, dim=2)
    _, max_indices_input = torch.max(sequences_input, dim=1)
    max_indices_input[max_indices_input < 0] = 21
    sequences_gt = torch.argmax(ground_truth, dim=2)
    _, max_indices_gt = torch.max(sequences_gt, dim=1)
    size_discrepency = max_indices_gt - max_indices_input
    size_discrepency[size_discrepency<0] = 0

    sequence_not_end_positions = sequences_input != 2
    loss = bce(input_tokens[sequence_not_end_positions], ground_truth[sequence_not_end_positions])
    first_end_token_input = input_tokens[:,max_indices_input,:]
    the_same_token_gt = ground_truth[:,max_indices_input,:]
    bce_weighted = nn.BCEWithLogitsLoss(weight=size_discrepency)
    end_tokens_loss = bce_weighted(first_end_token_input, the_same_token_gt)
    loss = loss + end_tokens_loss
    return loss


def hemming_distance(word1, word2):
    assert len(word1) == len(word2)
    counter= 0
    for x,y in zip(word1, word2):
        counter = counter + (x!=y)
    return counter
