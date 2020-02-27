import math
import sys
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.utils import shuffle


def evaluate(adj_list, val_nodes, features, graphSage, gap, bal_coeff, cut_coeff, num_classes, device, args):
    embs = graphSage(val_nodes, features, adj_list)
    logists = gap(embs)
    _, predicts = torch.max(logists, 1)
    loss = get_gap_loss(adj_list, bal_coeff, gap, cut_coeff, embs, val_nodes, num_classes, device)
    print("Validation loss: ", loss)

    filename = "model-{}_{}_mb{}_e{}_ge{}_gmb{}_gumbel-{}_cut{}_bal{}_agg{}-{}.torch".format(args.dataSet,
                                                                                             args.learn_method,
                                                                                             args.b_sz,
                                                                                             args.epochs,
                                                                                             args.gap_epochs,
                                                                                             str(args.gap_b_sz),
                                                                                             str(args.gumbel),
                                                                                             args.cut_coeff,
                                                                                             args.bal_coeff,
                                                                                             args.agg_func,
                                                                                             time.time())
    torch.save([graphSage, gap], filename)
    return loss


def get_gnn_embeddings(gnn_model, nodes_ids, features, adj_list):
    print('Loading embeddings from trained GraphSAGE model.')
    b_sz = 500
    batches = math.ceil(len(nodes_ids) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes_ids[index * b_sz:(index + 1) * b_sz]
        embs_batch = gnn_model(nodes_batch, features, adj_list)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes_ids)
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
def train_gap(nodes_ids, features, graphSage, classification, ds, adj_list, num_classes, device, b_sz=0, epochs=800,
              cut_coeff=1,
              bal_coeff=1):
    print('Training GAP ...')
    c_optimizer = torch.optim.Adam(classification.parameters(), lr=7.5e-5)
    # train classification, detached from the current graph
    # classification.init_params()
    embs = get_gnn_embeddings(graphSage, nodes_ids, features, adj_list)

    train_nodes = []
    for emd_id, node_id in enumerate(nodes_ids):
        train_nodes.append(emd_id)

    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        if b_sz == 0:
            b_sz = len(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            visited_nodes |= set(nodes_ids[nodes_batch])
            
            embs_batch = embs[nodes_batch]

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
    D.requires_grad = False
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
    A.requires_grad = False
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


def apply_model(nodes, features, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method,
                adj_list, num_classes, cut_coeff=1, bal_coeff=1):
    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(nodes)

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
        # labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        embs_batch = graphSage(nodes_batch, features, adj_list)

        if learn_method == 'sup':
            pass
            # # superivsed learning
            # logists = classification(embs_batch)
            # # loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            # loss_sup /= len(nodes_batch)
            # loss = loss_sup
        elif learn_method == 'gap':
            loss = get_gap_loss(adj_list, bal_coeff, classification, cut_coeff, embs_batch, nodes_batch, num_classes,
                                device=device)
        elif learn_method == 'gap_plus':
            gap_loss = get_gap_loss(adj_list, bal_coeff, classification, cut_coeff, embs_batch, nodes_batch,
                                    num_classes, device=device)
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net + gap_loss

        elif learn_method == 'plus_unsup':
            pass
            # # superivsed learning
            # logists = classification(embs_batch)
            # # loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            # # loss_sup /= len(nodes_batch)
            # # unsuperivsed learning
            # if unsup_loss == 'margin':
            #     loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            # elif unsup_loss == 'normal':
            #     loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            # loss = loss_sup + loss_net
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
