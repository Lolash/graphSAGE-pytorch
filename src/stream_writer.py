import csv
from queue import Empty

from torch.multiprocessing import Queue


def write_batches_from_queue_to_file(queue: Queue, file_path):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        while True:
            try:
                batch = queue.get(block=True, timeout=60)
                writer.writerows(batch)
            except Empty:
                print("Timeout during reading from WRITING queue.")
                print(file_path)
                return
