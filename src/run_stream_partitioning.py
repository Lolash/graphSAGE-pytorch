import argparse
import copy
from collections import defaultdict
from datetime import datetime

import pyhocon
import torch
from torch import multiprocessing
from torch.multiprocessing import Queue, Process

from src.dataCenter import DataCenter
from src.stream_partitioner import partition_batches_from_queue
from src.stream_reader import read_batches_to_queue
from src.stream_writer import write_batches_from_queue_to_file

if __name__ == '__main__':
    print("RUN STREAM PARTITIONING")
    parser = argparse.ArgumentParser(description='Streaming partitioning with GCN')

    parser.add_argument('--learn_method', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--max_load', type=float)
    parser.add_argument('--model', type=str)
    parser.add_argument('--inf_b_sz', type=int)
    parser.add_argument('--num_processes', type=int)
    parser.add_argument('--edge_file_path', type=str)
    parser.add_argument('--sorted_inference', action='store_true')
    parser.add_argument('--with_train_adj', action='store_true')

    args = parser.parse_args()

    print(args)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device_id = torch.cuda.current_device()
            print('using device', device_id, torch.cuda.get_device_name(device_id))
    map_location = 'cpu' if not args.cuda else lambda storage, loc: storage.cuda()
    device = torch.device("cuda" if args.cuda else "cpu")
    # device = torch.device("cpu")
    print('DEVICE:', device)

    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)

    # load data
    ds = args.dataSet
    dataCenter = DataCenter(config)
    features_ndarray = dataCenter.get_train_features(ds)
    features = torch.from_numpy(features_ndarray).to(device)
    if args.with_train_adj:
        adj_list_train = dataCenter.get_train_adj_list(ds)
    else:
        adj_list_train = defaultdict(set)

    gnn_num_layers = config['setting.num_layers']
    gnn_emb_size = config['setting.hidden_emb_size']

    if args.model != "":
        [graphSage, classification] = torch.load(args.model, map_location=map_location)
    elif args.graphsage_model != "" and args.classification_model != "":
        graphSage = torch.load(args.graphsage_model, map_location=map_location)
        classification = torch.load(args.classification_model, map_location=map_location)
    else:
        raise Exception("You have to specify a model to run!")

    graphSage.to(device)

    if args.num_classes == 0:
        num_labels = len(set(getattr(dataCenter, ds + '_labels')))
    else:
        num_labels = args.num_classes
    classification.to(device)

    graphSage.share_memory()
    classification.share_memory()
    multiprocessing.set_start_method("spawn", force=True)

    pool = multiprocessing.Pool()

    reading_queue = Queue(maxsize=100)
    reader = Process(target=read_batches_to_queue,
                     args=(args.file_path, args.inf_b_sz, reading_queue))
    reader.start()

    processes = []
    writing_queue = Queue(maxsize=100)
    for i in range(args.num_processes):
        processes.append(Process(target=partition_batches_from_queue,
                                 args=(reading_queue, writing_queue, copy.deepcopy(adj_list_train), False, features,
                                       classification, graphSage, args)))

    for p in processes:
        p.start()

    model_path = args.model.split("/")[-1]

    filename = "ds-{}-{}_win-size-{}_{}_{}_RESULTS.csv".format(
        args.dataSet,
        args.inf_b_sz,
        model_path,
        args.num_classes,
        datetime.now().strftime('%b%d_%H-%M-%S'))

    writer = Process(target=write_batches_from_queue_to_file, args=(writing_queue, filename))
    writer.start()

    reader.join()
    for p in processes:
        p.join()
    writer.join()
