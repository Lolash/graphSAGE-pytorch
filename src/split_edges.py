import argparse
import csv

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streaming partitioning with GCN')

    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--batch_size', type=int, default=100000)
    args = parser.parse_args()

    input_file_path = args.input_file
    output_file_path = args.output_file
    batch_size = args.batch_size
    i = 0

    with open(output_file_path, "a", newline="") as f:
        writer = csv.writer(f)
        for batch in pd.read_csv(input_file_path, chunksize=batch_size):
            print(i)
            i += batch_size
            edges_list = batch.values.tolist()
            nodes_list = []
            for e in edges_list:
                nodes_list.append((e[0], e[2]))
                nodes_list.append((e[1], e[2]))
            writer.writerows(nodes_list)


    print("End of file to read.")
