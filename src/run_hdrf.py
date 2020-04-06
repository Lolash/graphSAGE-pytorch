import argparse
import pandas as pd

from src.partition import partition_hdrf

parser = argparse.ArgumentParser(description='HDRF python implementation')

parser.add_argument('--input_file', type=str, default='')
parser.add_argument('--num_classes', type=int)
parser.add_argument('--load_imbalance', type=float, default=1.5)
parser.add_argument('--output_file', type=str, default='')

args = parser.parse_args()


def iterate_edges(edges):
    for i, r in edges.iterrows():
        e = r["edge"]
        src = int(e.split(",")[0][1:])
        dst = int(e.split(",")[1][1:-1])
        yield [src, dst]


edges = pd.read_csv(args.input_file)

partitions = partition_hdrf(iterate_edges(edges), args.num_classes, args.load_imbalance)

result = []
for label, edges in partitions.items():
    for e in edges:
        result.append({"src": e[0], "dst": e[1], "label": label})

assigned_edges = pd.DataFrame(result)
if args.output_file == "":
    output_file = args.input_file + "_labeled"
else:
    output_file = args.output_file
assigned_edges.to_csv(output_file, index=False)
