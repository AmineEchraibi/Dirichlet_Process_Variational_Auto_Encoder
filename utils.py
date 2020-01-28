from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
from sklearn.cluster import KMeans
import torch

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(torch.nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)

def initialise_phi_with_kmeans(X, K):

    mu = KMeans(K).fit(X).cluster_centers_
    phi = torch.from_numpy(np.exp( - 0.5 * np.linalg.norm(X.reshape(X.shape[0], 1, X.shape[1]) - mu.reshape(1, K, X.shape[1]),2,2)))
    return phi / torch.sum(phi, 1, True)


def initialize_responsibilities(N, K):
    """
    Initializing responsiblities or posterior class probabilities
    :param N: Number of instances
    :param K: Number of classes
    :return: R : responsibilities otf shape [N, K]
    """

    phi = torch.rand(N, K)
    phi = phi / torch.sum(phi, 1, True)

    return phi


def cluster_acc(Y_pred, Y):
    """
    Function computing cluster accuracy and confusion matrix at a permutation of the labels
    :param Y_pred: The predicted labels of shape [N, ]
    :param Y: The true labels of shape [N, ]
    :return: Clusterring_accuracy, Confusion matrix
    """
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w

def cumsum_ex(arr):
    """
    Function computing the cumulative sum exluding the last element for the first element the cumsum is 0
    :param arr: array of shape [p,]
    :return: cum_sum_arr: of shape [p,] where cum_sum_arr[0] = 0 and cum_sum_arr[i] = cumsum(arr[:i])
    """
    cum_sum_arr = torch.zeros_like(arr)
    for i in range(arr.shape[0]):
        if i == 0:
            cum_sum_arr[i] = 0
        else:
            cum_sum_arr[i] = torch.cumsum(arr[:i],0)[-1]
    return cum_sum_arr