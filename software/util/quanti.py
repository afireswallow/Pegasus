import numpy as np
from sklearn.linear_model import Ridge
import numba
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import os
import json
import argparse



@numba.njit(fastmath=True, cache=True, parallel=False)
def _cumsse_cols(X):
    # TODO: can be optimized with numpy
    N, D = X.shape
    cumsses = np.empty((N, D), X.dtype)
    cumX_column = np.empty(D, X.dtype)
    cumX2_column = np.empty(D, X.dtype)
    for j in range(D):
        cumX_column[j] = X[0, j]
        cumX2_column[j] = X[0, j] * X[0, j]
        cumsses[0, j] = 0  # no err in bucket with 1 element
    for i in range(1, N):
        one_over_count = 1.0 / (i + 1)
        for j in range(D):
            cumX_column[j] += X[i, j]
            cumX2_column[j] += X[i, j] * X[i, j]
            meanX = cumX_column[j] * one_over_count
            cumsses[i, j] = cumX2_column[j] - (cumX_column[j] * meanX)
    return cumsses


def optimal_split_val(X, dim, X_orig=None):
    X_orig = X if X_orig is None else X_orig
    if X_orig.shape != X.shape:
        assert X_orig.shape == X.shape

    if X.shape[0] == 0:
        return dim, 0, 100000000000000000000

    N, _ = X.shape
    sort_idxs = np.argsort(X_orig[:, dim])
    X_sort = X[sort_idxs]

    # cumulative SSE (sum of squared errors)
    sses_head = _cumsse_cols(X_sort)
    sses_tail = _cumsse_cols(X_sort[::-1])[::-1]
    sses = sses_head
    sses[:-1] += sses_tail[1:]
    sses = sses.sum(axis=1)

    best_idx = np.argmin(sses)
    next_idx = min(N - 1, best_idx + 1)
    col = X[:, dim]
    best_val = (col[sort_idxs[best_idx]] + col[sort_idxs[next_idx]]) / 2
    count = np.sum(X[:, dim] == best_val)

    return dim, best_val, count, sses[best_idx]


def get_sub_bucket(X):
    best_bucket = 0, 0, 100000000000000000000
    for i in range(X.shape[1]):
        bucket = optimal_split_val(X, i)
        if bucket[-1] < best_bucket[-1]:
            best_bucket = bucket
        elif bucket[-1] == best_bucket[-1]:
            if bucket[2] < best_bucket[2]:
                best_bucket = bucket
    return best_bucket


def get_ans(X):
    buckets = []
    prototypes = []
    buckets.append(X)
    beg = 0
    end = 1
    ans = []
    for i in range(15):
        best_bucket = get_sub_bucket(buckets[beg])

        X1 = buckets[beg][buckets[beg][:, best_bucket[0]] <= best_bucket[1]]
        X2 = buckets[beg][buckets[beg][:, best_bucket[0]] > best_bucket[1]]
        if X1.shape[0] == 0:
            buckets.append(X2)
            buckets.append(X2)
        elif X2.shape[0] == 0:
            buckets.append(X1)
            buckets.append(X1)
        else:
            buckets.append(X1)
            buckets.append(X2)
        end += 2
        beg += 1

        ans.append(best_bucket)
    for i in range(15, len(buckets)):
        prototypes.append(np.mean(buckets[i], axis=0))
    return ans, prototypes




def model_quanti(train_data,D,num,savepth):
    S_all = np.zeros((train_data.shape[1] // D, D, 15))
    T_all = np.zeros((train_data.shape[1] // D, 15))
    LUT_all = np.zeros((train_data.shape[1] // D, 16, D))
    for i in range(0, train_data.shape[1], D):
        # print(i)
        X = train_data[:, i:i+D]
        # print(X)
        # X = np.random.rand(100, D)
        # print(X.shape)
        #print(X.shape)
        ans, prototypes = get_ans(X)
        # print(ans)
        # print(prototypes)
        for j in range(15):
            S_all[i // D, ans[j][0], j] = 1
            T_all[i // D, j] = ans[j][1]
        for j in range(16):
            LUT_all[i // D, j] = prototypes[j]
    
    path = str(savepth)+f'/pth/num={num}'
    os.makedirs(path, exist_ok=True)

    torch.save(torch.tensor(S_all).float(), f'{path}/S_all.pth')
    torch.save(torch.tensor(T_all).float(), f'{path}/T_all.pth')
    torch.save(torch.tensor(LUT_all).float(), f'{path}/LUT_all.pth')
    
    return torch.tensor(S_all).float(),torch.tensor(T_all).float(),torch.tensor(LUT_all).float()


if __name__ == '__main__':
    #print(1)
    
    train_data = np.asarray([
        [1,  1, 1, -1,  1, -1, -1, -1],
        [1,  1, 1,  1, -1, -1, -1, -1],
        [1, -1, 1,  1,  1,  1,  1,  1],
        [1, -1, 1,  1,  1,  1,  1,  1]
    ])
    """
    train_data = np.asarray([
        [1,  1],
        [1, -1],
        [1,  1],
        [-1, -1]
    ])
    """
    S,T,LUT = model_quanti(train_data,4,1,0)
    print(S)
    print(T)
    print(LUT)
















    
