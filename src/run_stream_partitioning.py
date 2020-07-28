import time
from collections import defaultdict
from multiprocessing.context import Process

import pyhocon
import torch

from src.args import parser
from src.dataCenter import DataCenter
from src.stream_reader import read_batches_to_queue
from src.stream_writer import write_batches_from_queue_to_file
from src.stream_partitioner import partition_batches_from_queue
import multiprocessing

if __name__ == '__main__':
    print("RUN STREAM PARTITIONING")
    args = parser().parse_args()

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device_id = torch.cuda.current_device()
            print('using device', device_id, torch.cuda.get_device_name(device_id))

    device = torch.device("cuda" if args.cuda else "cpu")
    # device = torch.device("cpu")
    print('DEVICE:', device)

    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)

    # load data
    ds = args.dataSet
    dataCenter = DataCenter(config)
    features_ndarray = dataCenter.get_train_features(ds)
    features = torch.FloatTensor(features_ndarray).to(device)
    # features_train = torch.FloatTensor(getattr(dataCenter, ds + '_feats_train')).to(device)
    # features_test = torch.FloatTensor(getattr(dataCenter, ds + '_feats_test')).to(device)
    # features_val = torch.FloatTensor(getattr(dataCenter, ds + '_feats_val')).to(device)
    #
    # # self, num_layers, input_size, out_size, device, gcn=False, agg_func='MEAN')
    # adj_list_train = getattr(dataCenter, ds + "_adj_list_train")
    # adj_list_val = getattr(dataCenter, ds + '_adj_list_val')
    # adj_list_test = getattr(dataCenter, ds + '_adj_list_test')

    gnn_num_layers = config['setting.num_layers']
    gnn_emb_size = config['setting.hidden_emb_size']

    if args.model != "":
        [graphSage, classification] = torch.load(args.model)
    elif args.graphsage_model != "" and args.classification_model != "":
        graphSage = torch.load(args.graphsage_model)
        classification = torch.load(args.classification_model)
    else:
        raise Exception("You have to specify a model to run!")

    graphSage.to(device)

    if args.num_classes == 0:
        num_labels = len(set(getattr(dataCenter, ds + '_labels')))
    else:
        num_labels = args.num_classes
    classification.to(device)

    reading_queue = multiprocessing.Queue(maxsize=1000)
    reader = Process(target=read_batches_to_queue,
                     args=(config["file_path.reddit_edges"], args.inf_b_sz, reading_queue))
    # read_batches_to_queue("C:/Projects/graphSAGE-pytorch/reddit/edge_timestamp_4k.csv", 100, queue)
    reader.start()

    processes = []
    writing_queue = multiprocessing.Queue(maxsize=1000)
    for i in range(args.num_processes):
        processes.append(Process(target=partition_batches_from_queue,
                                 args=(reading_queue, writing_queue, defaultdict(set), False, features, classification,
                                       graphSage, args)))

    for p in processes:
        p.start()

    filename = "ds-{}_{}_mb-{}_e-{}_se-{}_smb-{}_inf-mb-{}_gumbel-{}_cut-{}_bal-{}_agg-{}-num_classes-{}-bfs-{}-{}-RESULTS.csv".format(
        args.dataSet,
        args.learn_method,
        args.b_sz,
        args.epochs,
        args.sup_epochs,
        args.sup_b_sz,
        args.inf_b_sz,
        args.gumbel,
        args.cut_coeff,
        args.bal_coeff,
        args.agg_func,
        args.num_classes,
        args.bfs,
        time.time())
    writer = Process(target=write_batches_from_queue_to_file, args=(writing_queue, filename))
    writer.start()

    reader.join()
    for p in processes:
        p.join()
    writer.join()
