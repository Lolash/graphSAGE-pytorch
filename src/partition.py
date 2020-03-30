import math
import subprocess
import time
from collections import defaultdict

import networkx as nx
import torch
import numpy as np

from src.utils import get_reduced_adj_list


def partition_edge_stream(edges, training_adj_list, features, graphsage, gap, args):
    # 1st version - take an edge, add it to training adj_list, get assignments of two nodes and assign the edge to the
    # partition which had more probability
    assignments = []
    freqs = [0] * args.num_classes
    # for e in edges:
    #     training_adj_list[e[0]].add(e[1])
    #     embs = graphsage([e[0], e[1]], features, training_adj_list)
    #     training_adj_list[e[0]].remove(e[1])
    #     predicts = gap(embs)
    #     values, partitions = torch.max(predicts, 1)
    #     _, idx = torch.max(values, 0)
    #     p = partitions[idx.item()].item()
    #     assignments.append(p)
    #     freqs[p] += 1

    # 2nd version - take a window of edges, create new graph from them, get assignments of their ends and assign each
    # edge to the partition of the end's assignment with higher probability
    batches = len(edges) // args.inf_b_sz
    print("BATCHES: ", batches)
    for i in range(batches):
        batch = edges[i*args.inf_b_sz:args.inf_b_sz*(i+1)]
        for e in batch:
            training_adj_list[e[0]].add(e[1])
        for e in batch:
            embs = graphsage([e[0], e[1]], features, training_adj_list)
            predicts = gap(embs)
            values, partitions = torch.max(predicts, 1)
            _, idx = torch.max(values, 0)
            p = partitions[idx.item()].item()
            assignments.append(p)
            freqs[p] += 1
        for e in batch:
            training_adj_list[e[0]].remove(e[1])

    print(freqs)
    evaluate_edge_partitioning(edges, assignments)


def evaluate_edge_partitioning(edges, partitioning):
    nodes_assignments = defaultdict(set)
    vertex_cut = 0

    for e, p in zip(edges, partitioning):
        if e[0] not in nodes_assignments:
            nodes_assignments[e[0]].add(p)
        elif p not in nodes_assignments[e[0]]:
            nodes_assignments[e[0]].add(p)
            vertex_cut += 1
        if e[1] not in nodes_assignments:
            nodes_assignments[e[1]].add(p)
        elif p not in nodes_assignments[e[1]]:
            nodes_assignments[e[1]].add(p)
            vertex_cut += 1

    ratio = vertex_cut / len(nodes_assignments)
    print("VERTEX CUT / ALL VERTICES: ", ratio)


def partition_graph(nodes, features, adj_list, name, graphsage, gap, gnn_num_layers, gnn_emb_size, num_labels, args,
                    tensorboard=None, batch_size=-1):
    embs = None
    if batch_size == -1:
        embs = graphsage(nodes, features, adj_list)
    else:
        iters = math.ceil(len(nodes) / batch_size)
        for i in range(iters):
            batch_nodes = nodes[i * batch_size:(i + 1) * batch_size]
            batch_adj_list = get_reduced_adj_list(batch_nodes, adj_list)
            batch_embs = graphsage(batch_nodes, features, batch_adj_list)
            if embs is None:
                embs = batch_embs
            else:
                embs = torch.cat([embs, batch_embs], 0)
    assert len(embs) == len(nodes)
    if tensorboard is not None:
        tensorboard.add_embedding(embs.cpu(), tag=name)
    logists = gap(embs)
    _, predicts = torch.max(logists, 1)

    graph = nx.Graph()
    val_adj_list = {}
    colors_dict = {}
    assert len(logists) == len(nodes)
    for i, node in enumerate(nodes):
        colors_dict[int(node)] = predicts[i]
        val_adj_list[int(node)] = adj_list[int(node)]
    graph = nx.Graph()
    for i, ns in val_adj_list.items():
        for n in ns:
            if n in val_adj_list:
                graph.add_edge(i, n)
        if not graph.has_node(i):
            graph.add_node(i)
    assert (len(graph.nodes) == len(nodes))
    colors = ['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'purple']
    partitions = {}
    for i, p in colors_dict.items():
        graph.nodes[i]['color'] = colors[p]
        if colors[p] not in partitions:
            partitions[colors[p]] = [i]
        else:
            partitions[colors[p]].append(i)
    cardinalities = np.array([len(i) for i in partitions.values()])
    balanced = np.array([int(len(graph.nodes) / num_labels)] * len(cardinalities))
    print(cardinalities)
    print(balanced)
    print(cardinalities - balanced)
    balancedness = 1 - ((cardinalities - balanced) ** 2).mean()

    perf = nx.algorithms.community.performance(graph, partitions.values())
    coverage = nx.algorithms.community.coverage(graph, partitions.values())

    print("Performance of {}: {}".format(name, perf))
    print("Coverage of {}: {}".format(name, coverage))
    print("Balancedness of {}: {}".format(name, balancedness))
    print("Edge cut of {}: {}".format(name, 1 - coverage))

    filename = "ds-{}_gnn_layers-{}_gnn_emb_size-{}_{}_mb-{}_e-{}_ge-{}_gmb-{}__inf-mb-{}_gumbel-{}_cut-{}_bal-{}_agg-{}_num_classes-{}_bfs-{}_{}-{}.dot".format(
        args.dataSet,
        gnn_num_layers,
        gnn_emb_size,
        args.learn_method,
        args.b_sz,
        args.epochs,
        args.gap_epochs,
        args.gap_b_sz,
        args.inf_b_sz,
        args.gumbel,
        args.cut_coeff,
        args.bal_coeff,
        args.agg_func,
        args.num_classes,
        args.bfs,
        name,
        time.time())
    nx.nx_pydot.write_dot(graph, filename)
    subprocess.call([r"C:\Program Files (x86)\Graphviz2.38\bin\sfdp.exe", filename, "-Tpng", "-o", filename + ".png"])
