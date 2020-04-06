import math
import subprocess
import time
from collections import defaultdict

import networkx as nx
import torch
import numpy as np

from src.utils import get_reduced_adj_list


def partition_edge_stream_gap_edge(edges, training_adj_list, features, graphsage, gap, name, args):
    assignments = []
    edges_embeddings = []
    for e in edges:
        training_adj_list[e[0]].add(e[1])
        emb_src = graphsage([e[0]], features, training_adj_list)[0]
        emb_dst = graphsage([e[1]], features, training_adj_list)[0]
        training_adj_list[e[0]].remove(e[1])
        # print("SRC EMB SIZE: ", src_emb.size())
        edge_emb = torch.cat([emb_src, emb_dst], 0)
        # print("EDGE EMB SIZE: ", edge_emb.size())
        edges_embeddings.append(edge_emb)
    edges_embeddings = torch.stack(edges_embeddings)
    predicts = gap(edges_embeddings)
    _, assignments = torch.max(predicts, 1)
    evaluate_edge_partitioning(edges, assignments, name + "_edge_by_edge", args.num_classes)


def partition_edge_stream_assign_edges(edges, training_adj_list, features, graphsage, gap, name, args):
    # 1st version - take an edge, add it to training adj_list, get assignments of two nodes and assign the edge to the
    # partition which had more probability
    assignments = []
    for e in edges:
        training_adj_list[e[0]].add(e[1])
        embs = graphsage([e[0], e[1]], features, training_adj_list)
        training_adj_list[e[0]].remove(e[1])
        predicts = gap(embs)
        values, partitions = torch.max(predicts, 1)
        _, idx = torch.max(values, 0)
        p = partitions[idx.item()].item()
        assignments.append(p)
    evaluate_edge_partitioning(edges, assignments, name + "_edge_by_edge", num_classes=args.num_classes)

    # 2nd version - take a window of edges, create new graph from them, get assignments of their ends and assign each
    # edge to the partition of the end's assignment with higher probability
    if args.inf_b_sz > 0:
        assignments = []
        batches = len(edges) // args.inf_b_sz
        print("BATCHES: ", batches)
        for i in range(batches):
            batch = edges[i * args.inf_b_sz:args.inf_b_sz * (i + 1)]
            for e in batch:
                training_adj_list[e[0]].add(e[1])
            for e in batch:
                embs = graphsage([e[0], e[1]], features, training_adj_list)
                predicts = gap(embs)
                values, partitions = torch.max(predicts, 1)
                _, idx = torch.max(values, 0)
                p = partitions[idx.item()].item()
                assignments.append(p)
            for e in batch:
                training_adj_list[e[0]].remove(e[1])
        evaluate_edge_partitioning(edges, assignments, name + "_window", num_classes=args.num_classes)


def partition_edge_stream_assign_nodes(edges, training_adj_list, features, graphsage, gap, name, args):
    assignments = []
    freqs = [0] * args.num_classes
    for e in edges:
        training_adj_list[e[0]].add(e[1])
        embs = graphsage([e[0], e[1]], features, training_adj_list)
        training_adj_list[e[0]].remove(e[1])
        predicts = gap(embs)
        values, partitions = torch.max(predicts, 1)
        _, idx = torch.max(values, 0)
        p = partitions[idx.item()].item()
        assignments.append(p)
        freqs[p] += 1
    print("ASSIGN EDGE BY EDGE: \n", freqs)
    evaluate_edge_partitioning(edges, assignments, name + "_edge_by_edge", args.num_classes)


def partition_hdrf(edges, num_classes, load_imbalance):
    partial_degrees = defaultdict(lambda: 0)
    partitions = {c: set() for c in range(num_classes)}
    max_size = 0
    min_size = 0
    for src, dst in edges:
        partial_degrees[src] += 1
        src_deg = partial_degrees[src]
        dst_deg = partial_degrees[dst]
        max_hdrf = float("-inf")
        max_p = None
        for k, p in partitions.items():
            cur_hdrf = hdrf(src, dst, p, partial_degrees, max_size, min_size, load_imbalance)
            if max_hdrf < hdrf(src, dst, p, partial_degrees, max_size, min_size, load_imbalance):
                max_hdrf = cur_hdrf
                max_p = k

        partitions[max_p].add((src, dst))

        temp_max_size = temp_min_size = None
        for p in partitions.values():
            size = len(p)
            if temp_max_size is None or size > temp_max_size:
                temp_max_size = size
            if temp_min_size is None or size < temp_min_size:
                temp_min_size = size
    return partitions


def hdrf(v1, v2, p, partial_degrees, max_size, min_size, load_imbalance, eps=1):
    size = len(p)
    return hdrf_rep(v1, v2, p, partial_degrees) + hdrf_bal(load_imbalance, max_size, min_size, size, eps)


def hdrf_bal(load_imbalance, max_size, min_size, size, eps=1):
    return load_imbalance * (max_size - size) / (eps + max_size - min_size)


def hdrf_rep(v1, v2, p, partial_degrees):
    g1 = hdrf_g(v1, normalize_degree(partial_degrees[v1], partial_degrees[v2]), p)
    g2 = hdrf_g(v2, normalize_degree(partial_degrees[v2], partial_degrees[v1]), p)
    return g1 + g2


def hdrf_g(v, norm_degree_v, p):
    if v in p:
        return 1 + 1 - norm_degree_v
    else:
        return 0


def normalize_degree(d1, d2):
    return d1 / (d1 + d2)


def evaluate_edge_partitioning(edges, partitioning, name, num_classes):
    nodes_assignments = defaultdict(set)
    vertex_cut = 0
    freqs = [0] * num_classes

    for e, p in zip(edges, partitioning):
        if isinstance(p, torch.Tensor):
            p = p.item()
        freqs[p] += 1
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
    print("FREQS of {}: {}".format(name, freqs))
    print("VERTEX CUT / ALL VERTICES of {}: {}".format(name, ratio))


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
