import multiprocessing

import pandas as pd


def read_batches_to_queue(file_path, batch_size, queue: multiprocessing.Queue):
    i = 0
    for batch in pd.read_csv(file_path, chunksize=batch_size):
        print(i)
        i += batch_size
        edges_list = batch.values.tolist()
        # reddit dataset processing
        edges = list(map(lambda x: [int(x[1].split(",")[0][1:]), int(x[1].split(",")[1][1:-1])], edges_list))
        if len(edges) > 0:
            queue.put(edges, block=True)
    print("End of file to read.")
