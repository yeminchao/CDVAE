import torch
import torch.nn as nn


def ED(A, B):
    BT = B.t()
    vecProd = A.mm(BT)
    SqA = A ** 2
    sumSqA = (torch.sum(SqA, axis=1)).view(1, A.size(0))
    sumSqA = sumSqA.t()
    sumSqAEx = sumSqA.repeat(1, vecProd.size(1))

    SqB = B ** 2
    sumSqB = (torch.sum(SqB, axis=1)).view(1, B.size(0))
    sumSqBEx = sumSqB.repeat(vecProd.size(0), 1)
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    ED = torch.sqrt(SqED)
    return ED


def discriminative_adjacency_matrix(
        source_label,
        target_label):
    W = torch.zeros((source_label.shape[0], target_label.shape[0]))
    for source_num in range(source_label.shape[0]):
        for target_num in range(target_label.shape[0]):
            if source_label[source_num] == target_label[target_num]:
                W[source_num][target_num] = 1
            else:
                W[source_num][target_num] = 0
    return W


class graph_loss(nn.Module):
    def __init__(self):
        super(graph_loss, self).__init__()

    def forward(self, source_data, target_data, W, class_num):
        loss = ED(source_data, target_data)
        n = source_data.shape[0] * target_data.shape[0] / class_num
        W_norm = (1 / n) ** 0.5 * W
        loss = torch.sum(loss * W_norm)

        return loss
