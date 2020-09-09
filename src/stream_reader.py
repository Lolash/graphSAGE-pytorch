from queue import Full

import pandas as pd
from torch.multiprocessing import Queue


def read_batches_to_queue(file_path, batch_size, queue: Queue):
    i = 0
    for batch in pd.read_csv(file_path, chunksize=batch_size):
        print(i)
        i += batch_size
        edges_list = batch.values.tolist()
        # reddit dataset processing
        if "reddit" in file_path:
            edges = list(map(lambda x: [int(x[1].split(",")[0][1:]), int(x[1].split(",")[1][1:-1])], edges_list))
        else:
            edges = edges_list
        if len(edges) > 0:
            try:
                queue.put(edges, block=True)
            except Full:
                print("Timeout during writing to READING queue.")
                return
    print("End of file to read.")
