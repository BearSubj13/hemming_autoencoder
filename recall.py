import torch
import numpy as np


def outer_pairwise_distance(A, B=None):
    """
        Compute pairwise_distance of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = distance(i-th row matrix A, j-th row matrix B)
        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None: B = A

    max_size = 2 ** 26
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:

        return torch.pairwise_distance(
            A[:, None].expand(n, m, d).reshape((-1, d)),
            B.expand(n, m, d).reshape((-1, d))
        ).reshape((n, m))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_pairwise_distance(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


def outer_cosine_similarity(A, B=None):
    """
        Compute cosine_similarity of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = cosine_similarity(i-th row matrix A, j-th row matrix B)
        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None: B = A

    max_size = 2 ** 32
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:

        A_norm = torch.div(A.transpose(0, 1), A.norm(dim=1)).transpose(0, 1)
        B_norm = torch.div(B.transpose(0, 1), B.norm(dim=1)).transpose(0, 1)
        return torch.mm(A_norm, B_norm.transpose(0, 1))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_cosine_similarity(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


# 1. считаем все попарные растояния
# Далее для каждого элемента
# 2. ищем близжайшие K+1 элементов
# 3. самый близжайший не расматриваем (это он сам)
# 4. смотрим метки для K близжайших кроме первого и проверяем совпадет ли эта метка с меткой того которого мы сейчас рассматриваем (это как раз 116 строка)
# 5. считаем долю таких совпадений
def metric_Recall_top_K(X, y, K, metric='cosine'):
    """
        calculate metric R@K
        X - tensor with size n x d, where n - number of examples, d - size of embedding vectors
        y - true labels (are the same for the same object with augmentations)
        N - count of closest examples, which we consider for recall calcualtion
        metric: 'cosine' / 'euclidean'.
            !!! 'euclidean' - to slow for datasets bigger than 100K rows
    """
    res_recall = []
    res_recall10 = []

    n = X.size(0)
    d = X.size(1)
    max_size = 2 ** 32
    batch_size = max(1, max_size // (n*d))

    with torch.no_grad():

        for i in range(1 + (len(X) - 1) // batch_size):

            id_left = i*batch_size
            id_right = min((i+1)*batch_size, len(y))
            y_batch = y[id_left:id_right]

            if metric == 'cosine':
                pdist = -1 * outer_cosine_similarity(X, X[id_left:id_right])
            elif metric == 'euclidean':
                pdist = outer_pairwise_distance(X, X[id_left:id_right])
            else:
                raise AttributeError(f'wrong metric "{metric}"')

            y_rep = y_batch.repeat(K, 1)

            values, indices = pdist.topk(K + 1, 0, largest=False)
            res_recall.append((y[indices[1:]] == y_rep).sum().item())

            values, indices = pdist.topk(int(n/10) + K + 1, 0, largest=False)
            y_rep = y_batch.repeat(int(n/10)+K, 1)
            res_recall10.append((y[indices[1:]] == y_rep).sum().item())

    recall = np.sum(res_recall) / len(y) / K
    recall10 = np.sum(res_recall10) / len(y) / K
    return recall, recall10


def recall_top2(X1, X2):
    """ X - tensor with size n x d, where n - number of examples, d - size of embedding vectors
    """
    K = 2
    X = torch.cat((X1, X2), dim=0)

    y_label = torch.arange(0, X1.shape[0]).long()
    y_label = torch.cat((y_label, y_label), dim=0)
    y_label = y_label.to(X1.device)

    recall, recall10 = metric_Recall_top_K(X, y_label, K, metric='cosine')
    return recall, recall10