from queue import Full

from torch._C import default_generator
from torch.multiprocessing import Queue
import random
import torch


# Generates batch_num batches of size batch_size of random edges in a graph with up to nodes_num nodes and puts generated
# batches to the queue.
def generate_batches_to_queue(batch_num, batch_size, nodes_num, queue: Queue, feats_dim=64):
    num_edges = 0
    for i in range(batch_num):
        features = []
        edges = []
        node2idx = {}
        num_edges += batch_size
        print(num_edges)
        last = 0
        while len(edges) < batch_size:
            edge = tuple(random.sample(range(0, nodes_num), 2))

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

        try:
            queue.put((edges, torch.stack(features), node2idx), block=True)
        except Full:
            print("Timeout during writing to READING queue.")
            return


def _generate_feats(size, node_id):
    # Setting seed using lower-level default_generator is WAY faster!
    return torch.rand(size, generator=default_generator.manual_seed(node_id))
