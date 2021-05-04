import argparse
import csv
import random
import torch

from torch._C import default_generator


def _generate_feats(size, node_id):
    return torch.rand(size, generator=default_generator.manual_seed(node_id)).tolist()


if __name__ == '__main__':
    print("GENERATE TRAINING GRAPH")
    parser = argparse.ArgumentParser(description='Training graph generator')

    parser.add_argument('--nodes_num', type=int)
    parser.add_argument('--edges_num', type=int)
    parser.add_argument('--feats_dim', type=int, default=64)

    args = parser.parse_args()

    edges_num = args.edges_num
    nodes_num = args.nodes_num
    feats_dim = args.feats_dim

    edges_generated = set()
    features = []
    edges = []
    node2idx = {}
    last = 0

    for i in range(edges_num):
        edge = tuple(random.sample(range(0, nodes_num), 2))
        while edge in edges_generated:
            edge = tuple(random.sample(range(0, nodes_num), 2))

        edges_generated.add(edge)
        edges.append(edge)
        if edge[0] not in node2idx:
            node2idx[edge[0]] = last
            last += 1
            src_feats = _generate_feats(feats_dim, edge[0])
            features.append(src_feats)

        if edge[1] not in node2idx:
            node2idx[edge[1]] = last
            last += 1
            dst_feats = _generate_feats(feats_dim, edge[1])
            features.append(dst_feats)

    edges_mapped = [(node2idx[e[0]], node2idx[e[1]]) for e in edges]

    with open('./generated/edges.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        for row in edges_mapped:
            csv_out.writerow(row)

    with open('./generated/feats.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        for row in features:
            csv_out.writerow(row)
