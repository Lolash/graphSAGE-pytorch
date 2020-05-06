import math
import subprocess
import time
from collections import defaultdict
from tqdm import tqdm

import networkx as nx
import torch
import numpy as np


def partition_and_eval_edge_stream_sup_edge(edges, training_adj_list, features, graphsage, classification, name,
                                            num_classes,
                                            inference_batch_size):
    assignments = []
    if inference_batch_size > 0:
        batches = len(edges) // inference_batch_size
        print("BATCHES: ", batches)
        for i in range(batches):
            edges_embeddings = []
            batch = edges[i * inference_batch_size:inference_batch_size * (i + 1)]
            added_edges = []
            for e in batch:
                if e[0] not in training_adj_list or e[1] not in training_adj_list[e[0]]:
                    training_adj_list[e[0]].add(e[1])
                    added_edges.append((e[0], e[1]))
            for e in batch:
                emb_src = graphsage([e[0]], features, training_adj_list)[0]
                emb_dst = graphsage([e[1]], features, training_adj_list)[0]
                edge_emb = torch.cat([emb_src, emb_dst], 0)
                edges_embeddings.append(edge_emb)
            edges_embeddings = torch.stack(edges_embeddings)
            predicts = classification(edges_embeddings)
            _, batch_assignments = torch.max(predicts, 1)
            assignments += batch_assignments.flatten().tolist()
            for e in added_edges:
                training_adj_list[e[0]].remove(e[1])
                if len(training_adj_list[e[0]]) == 0:
                    del training_adj_list[e[0]]
        evaluate_edge_partitioning(edges, assignments, name + str(inference_batch_size) + "_window",
                                   num_classes=num_classes)

    assignments = []
    edges_embeddings = []
    for e in edges:
        was_added = False
        if e[0] not in training_adj_list or e[1] not in training_adj_list[e[0]]:
            training_adj_list[e[0]].add(e[1])
            was_added = True
        emb_src = graphsage([e[0]], features, training_adj_list)[0]
        emb_dst = graphsage([e[1]], features, training_adj_list)[0]
        if was_added:
            training_adj_list[e[0]].remove(e[1])
            if len(training_adj_list[e[0]]) == 0:
                del training_adj_list[e[0]]
        # print("SRC EMB SIZE: ", src_emb.size())
        edge_emb = torch.cat([emb_src, emb_dst], 0)
        # print("EDGE EMB SIZE: ", edge_emb.size())
        edges_embeddings.append(edge_emb)
    edges_embeddings = torch.stack(edges_embeddings)
    predicts = classification(edges_embeddings)
    _, assignments = torch.max(predicts, 1)
    return evaluate_edge_partitioning(edges, assignments, name + "_edge_by_edge", num_classes)


def partition_edge_stream_assign_edges(edges, training_adj_list, features, graphsage, gap, name, args):
    # 1st version - take an edge, add it to training adj_list, get assignments of two nodes and assign the edge to the
    # partition which had more probability
    print("START PARTITIONING")
    assignments = []
    freqs = [0] * args.num_classes
    print(type(edges).__name__)
    for e in tqdm(edges):
        was_added = False
        if e[0] not in training_adj_list or e[1] not in training_adj_list[e[0]]:
            training_adj_list[e[0]].add(e[1])
            was_added = True
        with torch.no_grad():
            embs = graphsage([e[0], e[1]], features, training_adj_list)
        if was_added:
            training_adj_list[e[0]].remove(e[1])
            if len(training_adj_list[e[0]]) == 0:
                del training_adj_list[e[0]]
        with torch.no_grad():
            predicts = gap(embs)
        values, partitions = torch.max(predicts, 1)
        _, idx = torch.max(values, 0)
        p = partitions[idx.item()].item()
        perfect_load = (len(assignments) + 1) / args.num_classes
        p = get_edge_partition(freqs, predicts, perfect_load, args.max_load)
        assignments.append(p)
    print("MAX LOAD: ", args.max_load)
    evaluate_edge_partitioning(edges, assignments, name + "_edge_by_edge", num_classes=args.num_classes)

    # 2nd version - take a window of edges, create new graph from them, get assignments of their ends and assign each
    # edge to the partition of the end's assignment with higher probability
    if args.inf_b_sz > 0:
        assignments = []
        freqs = [0] * args.num_classes
        batches = len(edges) // args.inf_b_sz
        print("BATCHES: ", batches)
        for i in tqdm(range(batches+1)):
            batch = edges[i * args.inf_b_sz:args.inf_b_sz * (i + 1)]
            added_edges = []
            for e in batch:
                if e[0] not in training_adj_list or e[1] not in training_adj_list[e[0]]:
                    training_adj_list[e[0]].add(e[1])
                    added_edges.append((e[0], e[1]))
            for e in batch:
                with torch.no_grad():
                    embs = graphsage([e[0], e[1]], features, training_adj_list)
                    predicts = gap(embs)
                perfect_load = (len(assignments) + 1) / args.num_classes
                p = get_edge_partition(freqs, predicts, perfect_load, args.max_load)
                assignments.append(p)
            for e in added_edges:
                training_adj_list[e[0]].remove(e[1])
                if len(training_adj_list[e[0]]) == 0:
                    del training_adj_list[e[0]]
        print("MAX LOAD: ", args.max_load)
        evaluate_edge_partitioning(edges, assignments, name + "_window_{}".format(args.inf_b_sz),
                                   num_classes=args.num_classes)


def get_edge_partition(freqs, predicts, perfect_load, max_load):
    values, partitions = torch.max(predicts, 1)
    _, idx = torch.max(values, 0)
    p = partitions[idx.item()].item()
    freqs[p] += 1
    if get_load_value(freqs, perfect_load) > max_load:
        freqs[p] -= 1
        p = partitions[(idx.item() + 1) % 2].item()
        freqs[p] += 1
        if get_load_value(freqs, perfect_load) > max_load:
            freqs[p] -= 1
            p = freqs.index(min(freqs))
            freqs[p] += 1
    return p


def get_load_value(freqs, perfect_load):
    return max(freqs) / perfect_load


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
    edge_partitions = {c: set() for c in range(num_classes)}
    vertex_partitions = {c: set() for c in range(num_classes)}
    max_size = 0
    min_size = 0
    for src, dst in edges:
        partial_degrees[src] += 1
        src_deg = partial_degrees[src]
        dst_deg = partial_degrees[dst]
        max_hdrf = float("-inf")
        max_p = None
        for ep, vp in zip(edge_partitions.items(), vertex_partitions.items()):
            cur_hdrf = hdrf(src, dst, ep[1], vp[1], partial_degrees, max_size, min_size, load_imbalance)
            if max_hdrf < hdrf(src, dst, ep[1], vp[1], partial_degrees, max_size, min_size, load_imbalance):
                max_hdrf = cur_hdrf
                max_p = ep[0]

        edge_partitions[max_p].add((src, dst))
        vertex_partitions[max_p].add(src)
        vertex_partitions[max_p].add(dst)

        temp_max_size = temp_min_size = None
        for ep in edge_partitions.values():
            size = len(ep)
            if temp_max_size is None or size > temp_max_size:
                temp_max_size = size
            if temp_min_size is None or size < temp_min_size:
                temp_min_size = size
        max_size = temp_max_size
        min_size = temp_min_size
    return edge_partitions


def hash_edge_partitioning(edges, name, num_classes):
    assignments = []
    cur = 0
    for e in edges:
        assignments.append(cur)
        cur = (cur + 1) % num_classes
    return assignments


def hdrf(v1, v2, ep, vp, partial_degrees, max_size, min_size, load_imbalance, eps=1):
    size = len(ep)
    return hdrf_rep(v1, v2, vp, partial_degrees) + hdrf_bal(load_imbalance, max_size, min_size, size, eps)


def hdrf_bal(load_imbalance, max_size, min_size, size, eps=0.01):
    # print("Load imbalance: {}".format(load_imbalance))
    result = load_imbalance * (max_size - size) / (eps + max_size - min_size)
    return result


def hdrf_rep(v1, v2, vp, partial_degrees):
    g1 = hdrf_g(v1, normalize_degree(partial_degrees[v1], partial_degrees[v2]), vp)
    g2 = hdrf_g(v2, normalize_degree(partial_degrees[v2], partial_degrees[v1]), vp)
    return g1 + g2


def hdrf_g(v, norm_degree_v, vp):
    if v in vp:
        return 1 + 1 - norm_degree_v
    else:
        return 0


def normalize_degree(d1, d2):
    return d1 / (d1 + d2)


def evaluate_edge_partitioning(edges, partitioning, name, num_classes):
    nodes_assignments = defaultdict(set)
    freqs = [0] * num_classes
    assert len(edges) == len(partitioning)
    for e, p in zip(edges, partitioning):
        if isinstance(p, torch.Tensor):
            p = p.item()
        freqs[p] += 1
        if e[0] not in nodes_assignments:
            nodes_assignments[e[0]].add(p)
        elif p not in nodes_assignments[e[0]]:
            nodes_assignments[e[0]].add(p)
        if e[1] not in nodes_assignments:
            nodes_assignments[e[1]].add(p)
        elif p not in nodes_assignments[e[1]]:
            nodes_assignments[e[1]].add(p)
    vertex_copies = sum([len(copies) for copies in nodes_assignments.values()])
    print("FREQS of {}: {}".format(name, freqs))
    normalized_load = max(freqs) / (len(partitioning) // num_classes)
    print("NORMALIZED LOAD of {}: {}".format(name, normalized_load))
    replication_factor = vertex_copies / len(nodes_assignments)
    print("REPLICATION FACTOR of {}: {}".format(name, replication_factor))
    return normalized_load, replication_factor


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
    balancedness = max(cardinalities) / int(len(graph.nodes) / num_labels)

    # perf = nx.algorithms.community.performance(graph, partitions.values())
    # coverage = nx.algorithms.community.coverage(graph, partitions.values())

    # print("Performance of {}: {}".format(name, perf))
    # print("Coverage of {}: {}".format(name, coverage))
    print("Balancedness of {}: {}".format(name, balancedness))
    print("Edge cut of {}: {}".format(name, get_edge_cut(graph)))
    #
    # filename = "ds-{}_gnn_layers-{}_gnn_emb_size-{}_{}_mb-{}_e-{}_ge-{}_gmb-{}__inf-mb-{}_gumbel-{}_cut-{}_bal-{}_agg-{}_num_classes-{}_bfs-{}_{}-{}.dot".format(
    #     args.dataSet,
    #     gnn_num_layers,
    #     gnn_emb_size,
    #     args.learn_method,
    #     args.b_sz,
    #     args.epochs,
    #     args.gap_epochs,
    #     args.gap_b_sz,
    #     args.inf_b_sz,
    #     args.gumbel,
    #     args.cut_coeff,
    #     args.bal_coeff,
    #     args.agg_func,
    #     args.num_classes,
    #     args.bfs,
    #     name,
    #     time.time())
    # nx.nx_pydot.write_dot(graph, filename)
    # subprocess.call([r"C:\Program Files (x86)\Graphviz2.38\bin\sfdp.exe", filename, "-Tpng", "-o", filename + ".png"])


def get_reduced_adj_list(nodes_batch, adj_list):
    """ Returns adj_list which contains nodes only from given nodes list. """
    reduced_adj_list = {}

    for n in nodes_batch:
        reduced_adj_list[n] = set()
        for nei in adj_list[n]:
            if nei in nodes_batch:
                reduced_adj_list[n].add(nei)
    return reduced_adj_list


def get_edge_cut(graph: nx.Graph):
    all_edges = len(graph.edges)
    edge_cuts = 0
    for n in graph.nodes:
        for nb in graph.neighbors(n):
            if graph.nodes[n]["color"] != graph.nodes[nb]["color"]:
                edge_cuts += 1
    edge_cuts = edge_cuts / 2
    return edge_cuts / all_edges
