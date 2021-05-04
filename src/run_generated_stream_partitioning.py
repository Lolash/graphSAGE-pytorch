import argparse
from collections import defaultdict
from datetime import datetime

import torch
from torch import multiprocessing
from torch.multiprocessing import Queue, Process

from src.stream_partitioner import partition_batches_from_queue
from src.stream_generator import generate_batches_to_queue
from src.stream_writer import write_batches_from_queue_to_file

if __name__ == '__main__':
    print("RUN GENERATED STREAM PARTITIONING")
    parser = argparse.ArgumentParser(description='Streaming partitioning with GCN')

    parser.add_argument('--learn_method', type=str)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--max_load', type=float)
    parser.add_argument('--model', type=str)
    parser.add_argument('--inf_b_sz', type=int)
    parser.add_argument('--batch_num', type=int)
    parser.add_argument('--nodes_num', type=int)
    parser.add_argument('--num_processes', type=int)
    parser.add_argument('--sorted_inference', action='store_true')
    parser.add_argument('--with_train_adj', action='store_true')

    args = parser.parse_args()

    print(args)

    map_location = 'cpu'
    device = "cpu"
    # device = torch.device("cpu")
    print('DEVICE:', device)

    if args.model != "":
        [graphSage, classification] = torch.load(args.model, map_location=map_location)
    elif args.graphsage_model != "" and args.classification_model != "":
        graphSage = torch.load(args.graphsage_model, map_location=map_location)
        classification = torch.load(args.classification_model, map_location=map_location)
    else:
        raise Exception("You have to specify a model to run!")

    graphSage.to(device)

    num_labels = args.num_classes
    classification.to(device)

    graphSage.share_memory()
    classification.share_memory()
    multiprocessing.set_start_method("spawn", force=True)

    pool = multiprocessing.Pool()

    reading_queue = Queue(maxsize=100)

    generator = Process(target=generate_batches_to_queue,
                        args=(args.batch_num, args.inf_b_sz, args.nodes_num, reading_queue))
    generator.start()

    processes = []
    writing_queue = Queue(maxsize=100)
    for i in range(args.num_processes):
        processes.append(Process(target=partition_batches_from_queue,
                                 args=(reading_queue, writing_queue, defaultdict(set), True, None,
                                       classification, graphSage, args)))

    for p in processes:
        p.start()

    model_path = args.model.split("/")[-1]

    filename = "ds-{}-{}_win-size-{}_{}_{}_RESULTS.csv".format(
        "generated",
        args.inf_b_sz,
        model_path,
        args.num_classes,
        datetime.now().strftime('%b%d_%H-%M-%S'))

    writer = Process(target=write_batches_from_queue_to_file, args=(writing_queue, filename))
    writer.start()

    generator.join()
    for p in processes:
        p.join()
    writer.join()
