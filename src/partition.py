import math
import subprocess
import time
from collections import defaultdict
from tqdm import tqdm

import networkx as nx
import torch
import numpy as np
import sys


def partition_edge_stream_assign_edges(edges, adj_list, features, graphsage, gap, name, args, bidirectional=False):
    # 1st version - take an edge, add it to training adj_list, get assignments of two nodes and assign the edge to the
    # partition which had more probability
    print("START PARTITIONING")
    # assignments, freqs, processing_time = _partition_edges_one_by_one(adj_list, args, bidirectional, edges, features,
    #                                                                   gap, graphsage)
    # print("MAX LOAD: ", args.max_load)
    # evaluate_edge_partitioning(edges, assignments, name + "_edge_by_edge", num_classes=args.num_classes)

    # 2nd version - take a window of edges, create new graph from them, get assignments of their ends and assign each
    # edge to the partition of the end's assignment with higher probability
    if args.inf_b_sz > 0:
        batches = len(edges) // args.inf_b_sz
        print("BATCHES: ", batches)
        assignments, freqs, processing_time = _partition_edges_in_batches(adj_list, args, bidirectional, batches, edges, features, gap,
                                                         graphsage)
        evaluate_edge_partitioning(edges, assignments, name + "_window_{}".format(args.inf_b_sz),
                                   num_classes=args.num_classes)
    return processing_time


def _partition_edges_one_by_one(adj_list, args, bidirectional, edges, features, gap, graphsage):
    processing_time = []
    assignments = []
    freqs = [0] * args.num_classes
    idx = 0
    for e in tqdm(edges):
        if idx == 0:
            start_time = time.time()
        if idx != 1 and idx % 10000 == 1:
            start_time = time.time()
        added_edges = set()
        if e[0] not in adj_list:
            adj_list[e[0]].add(e[1])
            added_edges.add((e[0], e[1]))
        if bidirectional and e[1] not in adj_list[e[0]]:
            adj_list[e[1]].add(e[0])
            added_edges.add((e[1], e[0]))
        with torch.no_grad():
            if args.learn_method == "sup_edge":
                emb_src = graphsage([e[0]], features, adj_list)[0]
                emb_dst = graphsage([e[1]], features, adj_list)[0]
                embs = torch.cat([emb_src, emb_dst], 0)
                embs = torch.stack([embs])
                predicts = gap(embs)
            else:
                embs = graphsage([e[0], e[1]], features, adj_list)
                predicts = gap(embs)

                perfect_load = (len(assignments) + 1) / args.num_classes
                if args.sorted_inference:
                    p = _get_sorted_inference(freqs, args.max_load, perfect_load, predicts)
                else:
                    max_val, max_idx = torch.max(predicts, dim=1)
                    max_val = max_val.squeeze().tolist()
                    max_idx = max_idx.squeeze().tolist()
                    p = _get_least_loaded_inference(freqs, args.max_load, perfect_load, max_val, max_idx)
                assignments.append(p)

        for e in added_edges:
            adj_list[e[0]].discard(e[1])
        added_edges.clear()

        if idx > 0 and idx % 10000 == 0:
            end_time = time.time()
            processing_time.append([idx, end_time - start_time])
        idx += 1
    return assignments, freqs, processing_time


def _partition_edges_in_batches(adj_list, args, bidirectional, batch_num, edges, features, gap, graphsage):
    assignments = []
    freqs = [0] * args.num_classes
    processing_time = []
    for i in tqdm(range(batch_num)):
        start_time = time.time()
        batch = edges[i * args.inf_b_sz:args.inf_b_sz * (i + 1)]
        added_edges = set()
        for e in batch:
            if e[0] not in adj_list:
                adj_list[e[0]].add(e[1])
                added_edges.add((e[0], e[1]))
            if bidirectional and e[1] not in adj_list[e[0]]:
                adj_list[e[1]].add(e[0])
                added_edges.add((e[1], e[0]))
        with torch.no_grad():
            if args.learn_method == "sup_edge":
                pass
            else:
                pairs = [item for sublist in map(lambda e: [e[0], e[1]], batch) for item in sublist]
                embs = graphsage(pairs, features, adj_list)
                predicts = gap(embs)
                max_val, max_idx = torch.max(predicts, dim=1)
                max_val = max_val.squeeze().tolist()
                max_idx = max_idx.squeeze().tolist()

                n = 0
                while n < len(pairs) - 1:
                    perfect_load = (len(assignments) + 1) / args.num_classes
                    if args.sorted_inference:
                        p = _get_sorted_inference(freqs, args.max_load, perfect_load, predicts)
                    else:
                        p = _get_least_loaded_inference(freqs, args.max_load, perfect_load, max_val[n:n + 2],
                                                        max_idx[n:n + 2])
                    assignments.append(p)
                    n += 2
            for e in added_edges:
                adj_list[e[0]].discard(e[1])
            added_edges.clear()
            end_time = time.time()
            processing_time.append([i, end_time - start_time])
    return assignments, freqs, processing_time


def _partition_one_batch(adj_list, args, assignments, batch, bidirectional, features, freqs, gap, graphsage):
    added_edges = set()
    for e in batch:
        if e[0] not in adj_list:
            adj_list[e[0]].add(e[1])
            added_edges.add((e[0], e[1]))
        if bidirectional and e[1] not in adj_list[e[0]]:
            adj_list[e[1]].add(e[0])
            added_edges.add((e[1], e[0]))
    with torch.no_grad():
        if args.learn_method == "sup_edge":
            pass
        else:
            pairs = [item for sublist in map(lambda e: [e[0], e[1]], batch) for item in sublist]
            embs = graphsage(pairs, features, adj_list)
            predicts = gap(embs)
            max_val, max_idx = torch.max(predicts, dim=1)
            max_val = max_val.squeeze().tolist()
            max_idx = max_idx.squeeze().tolist()

            n = 0
            while n < len(pairs) - 1:
                perfect_load = (len(assignments) + 1) / args.num_classes
                if args.sorted_inference:
                    p = _get_sorted_inference(freqs, args.max_load, perfect_load, predicts)
                else:
                    p = _get_least_loaded_inference(freqs, args.max_load, perfect_load, max_val[n:n + 2],
                                                    max_idx[n:n + 2])
                assignments.append(p)
                n += 2
    for e in added_edges:
        adj_list[e[0]].discard(e[1])
    added_edges.clear()


def _get_assigned_partition(learn_method, max_load, freqs, perfect_load, predicts, sorted_inference):
    if learn_method == "sup_edge":
        p = _get_edge_partition_from_edge(freqs, predicts, perfect_load, max_load, sorted_inference)
    else:
        p = _get_edge_partition_from_two_nodes(freqs, predicts, perfect_load, max_load, sorted_inference)
    return p


def _get_edge_partition_from_edge(freqs, predicts, perfect_load, max_load, sorted_inference):
    if sorted_inference:
        sorted_partitions = torch.argsort(predicts, descending=True)
        for p in sorted_partitions.flatten().split(1):
            p = p.item()
            freqs[p] += 1
            if get_load_value(freqs, perfect_load) < max_load:
                return p
            else:
                freqs[p] -= 1
        p = freqs.index(min(freqs))
        freqs[p] += 1
        return p
    else:
        p = torch.argmax(predicts).item()
        freqs[p] += 1
        if get_load_value(freqs, perfect_load) < max_load:
            return p
        else:
            freqs[p] -= 1
            p = freqs.index(min(freqs))
            freqs[p] += 1
            return p


def _get_edge_partition_from_two_nodes(freqs, predicts, perfect_load, max_load, sorted_inference):
    if sorted_inference:
        return _get_sorted_inference(freqs, max_load, perfect_load, predicts)
    else:
        return _get_least_loaded_inference(freqs, max_load, perfect_load, predicts)


def _get_least_loaded_inference(freqs, max_load, perfect_load, max_val, max_idx):
    v1, v2 = max_val[0], max_val[1]
    i1, i2 = max_idx[0], max_idx[1]
    if i1 != i2:
        if v1 > v2:
            # i1 = i1.item()
            p = i1
        else:
            # i2 = i2.item()
            p = i2
        freqs[p] += 1
        if get_load_value(freqs, perfect_load) < max_load:
            return p
        else:
            freqs[p] -= 1
            # p = i1.item() if p == i2 else i2.item()
            p = i1 if p == i2 else i2
            freqs[p] += 1
            if get_load_value(freqs, perfect_load) < max_load:
                return p
            freqs[p] -= 1
    else:
        p = i1
        freqs[p] += 1
        if get_load_value(freqs, perfect_load) < max_load:
            return p
        else:
            freqs[p] -= 1
    p = freqs.index(min(freqs))
    freqs[p] += 1
    return p


def _get_sorted_inference(freqs, max_load, perfect_load, predicts):
    sorted_values, sorted_partitions = torch.sort(predicts, descending=True)
    for e1, e2 in zip(sorted_partitions[0].flatten().split(1), sorted_partitions[1].flatten().split(1)):
        e1, e2 = e1.item(), e2.item()
        if predicts[0][e1] > predicts[1][e2]:
            p = e1
        else:
            p = e2
        freqs[p] += 1
        if get_load_value(freqs, perfect_load) < max_load:
            return p
        else:
            freqs[p] -= 1
            p = e1 if p == e2 else e2
            freqs[p] += 1
            if get_load_value(freqs, perfect_load) < max_load:
                return p
            freqs[p] -= 1
    p = freqs.index(min(freqs))
    freqs[p] += 1
    return p


def get_load_value(freqs, perfect_load):
    return max(freqs) / perfect_load


def partition_hdrf(edges, num_classes, load_imbalance):
    print("START HDRF PARTITIONING")
    partial_degrees = defaultdict(lambda: 0)
    edge_partitions = {c: set() for c in range(num_classes)}
    vertex_partitions = {c: set() for c in range(num_classes)}
    max_size = 0
    min_size = 0
    processing_time = []
    idx = 0
    for src, dst in tqdm(edges):
        if idx == 0:
            start_time = time.time()
        if idx != 1 and idx % 10000 == 1:
            start_time = time.time()
        partial_degrees[src] += 1
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
        if idx > 0 and idx % 10000 == 0:
            end_time = time.time()
            processing_time.append([idx, end_time - start_time])
        idx += 1

    print("SIZE OF VERTEX PARTITIONS: ", get_size(vertex_partitions))
    print("SIZE OF EDGE PARTITIONS: ", get_size(edge_partitions))

    return edge_partitions, processing_time


def hash_edge_partitioning(edges, name, num_classes):
    assignments = []
    cur = 0
    for e in tqdm(edges):
        assignments.append(cur)
        cur = (cur + 1) % num_classes
    return assignments


def hdrf(v1, v2, ep, vp, partial_degrees, max_size, min_size, load_imbalance, eps=1):
    size = len(ep)

    return hdrf_rep(v1, v2, vp, partial_degrees) + load_imbalance * hdrf_bal(max_size, min_size, size, eps)


def hdrf_bal(max_size, min_size, size, eps=0.00001):
    # print("Load imbalance: {}".format(load_imbalance))
    result = (max_size - size) / (eps + max_size - min_size)
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
    # assert len(edges) == len(partitioning)
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


# Source: https://goshippo.com/blog/measure-real-size-any-python-object/
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
