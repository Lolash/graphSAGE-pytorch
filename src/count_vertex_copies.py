import argparse
import csv
from collections import defaultdict
import pandas as pd

parser = argparse.ArgumentParser(description='HDRF python implementation')

parser.add_argument('--input_file', type=str, default='')
parser.add_argument('--output_file', type=str)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--chunk_size', type=int, default=100000)
parser.add_argument('--edges_num', type=int)

args = parser.parse_args()


def parse_one_batch(batch, current_partitions, prev):
    vertex_copies = []
    for n in batch:
        n_id = n[0]
        p = n[1]
        if prev[0] is None or prev[0] == n_id:
            if p not in current_partitions:
                current_partitions.add(p)
        else:
            vertex_copies.append((prev[0], len(current_partitions)))
            current_partitions = {p}
        prev[0] = n_id
    vertex_copies.append((prev[0], len(current_partitions)))
    return vertex_copies


nodes_copies = defaultdict(int)
edges_number = args.edges_num
i = 0
current_partitions = set()
prev = [None]
with open(args.output_file, "a", newline="") as f:
    writer = csv.writer(f)
    for batch in pd.read_csv(args.input_file, chunksize=args.chunk_size, header=None):
        print(i * args.chunk_size)
        batch_node_assignments = batch.values.tolist()
        batch_vertex_copies = parse_one_batch(batch_node_assignments, current_partitions, prev)
        writer.writerows(batch_vertex_copies)
    i += 1
