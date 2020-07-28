import csv
import multiprocessing

from queue import Empty


def write_batches_from_queue_to_file(queue: multiprocessing.Queue, file_path):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        while True:
            try:
                batch = queue.get(block=True, timeout=10)
                writer.writerows(batch)
            except Empty:
                print("Timeout during reading from WRITING queue.")
                return
