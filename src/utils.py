import math
import sys

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.utils import shuffle


def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, cur_epoch):
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    labels = getattr(dataCenter, ds + '_labels')

    models = [graphSage, classification]

    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    embs = graphSage(val_nodes)
    logists = classification(embs)
    _, predicts = torch.max(logists, 1)
    labels_val = labels[val_nodes]
    assert len(labels_val) == len(predicts)
    comps = zip(labels_val, predicts.data)

    vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
    print("Validation F1:", vali_f1)

    if vali_f1 > max_vali_f1:
        max_vali_f1 = vali_f1
        embs = graphSage(test_nodes)
        logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
        print("Test F1:", test_f1)

        for param in params:
            param.requires_grad = True

        torch.save(models, 'model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))

    for param in params:
        param.requires_grad = True

    return max_vali_f1


def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds + '_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds + '_labels'))).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index * b_sz:(index + 1) * b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()


def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=800):
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)
    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = labels[nodes_batch]
            embs_batch = features[nodes_batch]

            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(nodes_batch)
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()

        max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    return classification, max_vali_f1


# b_sz=0 means to take the whole dataset in each epoch
def train_gap(dataCenter, graphSage, classification, ds, adj_list, num_classes, device, b_sz=0, epochs=800, cut_coeff=1,
              bal_coeff=1):
    print('Training GAP ...')
    c_optimizer = torch.optim.Adam(classification.parameters(), lr=7.5e-5)
    # train classification, detached from the current graph
    # classification.init_params()
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)
    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        if b_sz == 0:
            b_sz = len(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = labels[nodes_batch]
            embs_batch = features[nodes_batch]

            loss = get_gap_loss(adj_list, bal_coeff, classification, cut_coeff, embs_batch, nodes_batch, num_classes,
                                device)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch + 1, epochs, index,
                                                                                            batches, loss.item(),
                                                                                            len(visited_nodes),
                                                                                            len(train_nodes)))
            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()

        # max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    return classification


def get_gap_loss(adj_list, bal_coeff, classification, cut_coeff, embs_batch, nodes_batch, num_classes, device):
    logists = classification(embs_batch)
    batch_adj_list = {}
    for node in nodes_batch:
        batch_adj_list[int(node)] = adj_list[int(node)]
    # print("batch_adj_list: ", batch_adj_list)
    D = torch.tensor([len(v) for v in batch_adj_list.values()], dtype=torch.float).to(device)
    # print("D: ", D)
    graph = nx.Graph()
    for i, ns in batch_adj_list.items():
        for n in ns:
            if n in batch_adj_list:
                graph.add_edge(i, n)
        if not graph.has_node(i):
            graph.add_node(i)
    A = nx.adj_matrix(graph)
    A = torch.from_numpy(sparse.coo_matrix.todense(A)).to(device)
    # print("A: ", A)
    gamma = logists.T @ D
    # print("gamma: ", gamma)
    y_div_gamma = logists / gamma
    # print("y/gamma: ", y_div_gamma)
    one_minus_Y_T = (1 - logists).T
    mm = y_div_gamma @ one_minus_Y_T
    # print("MM: ", mm)
    times_adj_matrix = mm * A
    # print("TIMES A: ", times_adj_matrix)
    left_sum = times_adj_matrix.sum()
    # print("LEFT SUM: ", left_sum)
    num_nodes = len(nodes_batch)
    cluster_size = torch.sum(logists, dim=0).to(device)
    ground_truth = torch.tensor([num_nodes / float(num_classes)] * num_classes).to(device)
    mse_loss = torch.nn.modules.loss.MSELoss()
    bal = mse_loss(ground_truth, cluster_size)
    loss = cut_coeff * left_sum + bal_coeff * bal
    return loss


def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method,
                adj_list, num_classes, cut_coeff=1, bal_coeff=1):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=7.5e-5)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # print("not extended: ", len(nodes_batch))
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        embs_batch = graphSage(nodes_batch)

        if learn_method == 'sup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            loss = loss_sup
        elif learn_method == 'gap':
            loss = get_gap_loss(adj_list, bal_coeff, classification, cut_coeff, embs_batch, nodes_batch, num_classes)
        elif learn_method == 'gap_plus':
            print("GAP PLUS")
            gap_loss = get_gap_loss(adj_list, bal_coeff, classification, cut_coeff, embs_batch, nodes_batch,
                                    num_classes)
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net + gap_loss

        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net

        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                         len(visited_nodes), len(train_nodes)))
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

    return graphSage, classification
