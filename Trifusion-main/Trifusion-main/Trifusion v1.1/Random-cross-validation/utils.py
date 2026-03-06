import numpy as np
import torch
from sklearn import metrics
import dgl
import networkx as nx
def caculate_metrics(real_score, pre_score):
    y_true = real_score
    y_pre = pre_score
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)

    y_score = [0 if j < 0.5 else 1 for j in y_pre]

    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, precision, recall]
    return metric_result
def integ_similarity(M1, M2):
    for i in range(len(M1)):
        for j in range(len(M1)):
            if M1[i][j] == 0:
                M1[i][j] = M2[i][j]
    return M1
def get_graph_adj(matrix, device):
    graph_adj = []
    for i in range(matrix.shape[0]):
        temp_adj = []
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                temp_adj.append(1)
            else:
                temp_adj.append(0)
        graph_adj.append(temp_adj)
    graph_adj = np.array(graph_adj).reshape(matrix.shape[0], matrix.shape[1])
    return torch.tensor(graph_adj, device=device).to(torch.float32)
def topk_filtering(args, d_d, k: int):
    d_d = d_d.numpy()
    for i in range(len(d_d)):
        sorted_idx = np.argpartition(d_d[i], -k - 1)
        d_d[i, sorted_idx[-k - 1:-1]] = 1
    return torch.tensor(np.where(d_d == 1), device=args.device)
def get_edge_index(matrix, device):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.tensor(edge_index, dtype=torch.long, device=device)
#添加设备判断
def k_matrix(matrix, k=40, device='cpu'):
    num = matrix.shape[0]

    # 兼容传入 torch.device 或字符串
    dev_type = device.type if hasattr(device, 'type') else str(device)

    if dev_type == 'cpu':
        # CPU版本
        knn_graph = np.zeros(matrix.shape)
        idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
        for i in range(num):
            knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
            knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
        return knn_graph + np.eye(num)

    else:
        # PyTorch GPU版本
        matrix = matrix.to(device)
        matrix_adj = matrix - torch.eye(num, device=device)
        idx_sort = torch.argsort(-matrix_adj, dim=1)

        knn_graph = torch.zeros_like(matrix)
        for i in range(num):
            knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
            knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]

        return (knn_graph + torch.eye(num, device=device)).cpu().numpy()
def create_matrix_from_indices(indices, shape=(853, 591)):

    matrix = np.zeros(shape, dtype=int)

    assert indices.shape[0] == 2, "Indices array must be of shape 2xN"

    for col, row in zip(indices[0], indices[1]):
        if 0 <= col < shape[0] and 0 <= row < shape[1]:
            matrix[col, row] = 1
        else:
            print(f"Warning: Index out of bounds - ({col}, {row})")

    return matrix



