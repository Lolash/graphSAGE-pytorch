import argparse
import time

import subprocess
import pyhocon

from src.dataCenter import *
from src.models import *
from src.utils import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--gap_epochs', type=int, default=200)
parser.add_argument('--b_sz', type=int, default=20)
parser.add_argument('--gap_b_sz', type=int, default=0)
parser.add_argument('--inf_b_sz', type=int, default=-1)
parser.add_argument('--gumbel', type=int, default=0)
parser.add_argument('--cut_coeff', type=float, default=1.0)
parser.add_argument('--bal_coeff', type=float, default=1.0)
parser.add_argument('--num_classes', type=int, default=0)
parser.add_argument('--bfs', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='./src/experiments.conf')
args = parser.parse_args()
filename = "ds-{}_{}_mb-{}_e-{}_ge-{}_gmb-{}_inf-mb-{}_gumbel-{}_cut-{}_bal-{}_agg-{}-num_classes-{}-bfs-{}-{}.dot".format(
    args.dataSet,
    args.learn_method,
    args.b_sz,
    args.epochs,
    args.gap_epochs,
    args.gap_b_sz,
    args.inf_b_sz,
    args.gumbel,
    args.cut_coeff,
    args.bal_coeff,
    args.agg_func,
    args.num_classes,
    args.bfs,
    time.time())
tensorboard = SummaryWriter("./runs/" + filename)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
# device = torch.device("cpu")
print('DEVICE:', device)


def partition_graph(nodes, features, adj_list, name, graphsage, gap, gnn_num_layers, gnn_emb_size, batch_size=-1):
    embs = None
    if batch_size == -1:
        embs = graphsage(nodes, features, adj_list)
    else:
        iters = math.ceil(len(nodes) / batch_size)
        for i in range(iters):
            batch_nodes = nodes[i * batch_size:(i + 1) * batch_size]
            batch_adj_list = get_reduced_adj_list(batch_nodes, adj_list)
            batch_embs = graphsage(batch_nodes, features, batch_adj_list)
            if embs is None:
                embs = batch_embs
            else:
                embs = torch.cat([embs, batch_embs], 0)
    print(len(embs))
    print(len(nodes))
    assert len(embs) == len(nodes)
    tensorboard.add_embedding(embs.cpu(), tag=name)
    logists = gap(embs)
    _, predicts = torch.max(logists, 1)
    graph = nx.Graph()
    val_adj_list = {}
    colors_dict = {}
    assert len(logists) == len(nodes)
    for i, node in enumerate(nodes):
        colors_dict[int(node)] = predicts[i]
        val_adj_list[int(node)] = adj_list[int(node)]
    graph = nx.Graph()
    for i, ns in val_adj_list.items():
        for n in ns:
            if n in val_adj_list:
                graph.add_edge(i, n)
        if not graph.has_node(i):
            graph.add_node(i)
    colors = ['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'purple']
    partitions = {}
    for i, p in colors_dict.items():
        graph.nodes[i]['color'] = colors[p]
        if colors[p] not in partitions:
            partitions[colors[p]] = [i]
        else:
            partitions[colors[p]].append(i)
    cardinalities = np.array([len(i) for i in partitions.values()])
    balanced = np.array([int(len(graph.nodes) / len(partitions.keys()))] * len(cardinalities))
    print(cardinalities)
    print(balanced)
    print(cardinalities - balanced)
    balancedness = 1 - ((cardinalities - balanced) ** 2).mean()

    perf = nx.algorithms.community.performance(graph, partitions.values())
    coverage = nx.algorithms.community.coverage(graph, partitions.values())

    print("Performance of {}: {}".format(name, perf))
    print("Coverage of {}: {}".format(name, coverage))
    print("Balancedness of {}: {}".format(name, balancedness))

    filename = "ds-{}_gnn_layers-{}_gnn_emb_size-{}_{}_mb-{}_e-{}_ge-{}_gmb-{}__inf-mb-{}_gumbel-{}_cut-{}_bal-{}_agg-{}_num_classes-{}_bfs-{}_{}-{}.dot".format(
        args.dataSet,
        gnn_num_layers,
        gnn_emb_size,
        args.learn_method,
        args.b_sz,
        args.epochs,
        args.gap_epochs,
        args.gap_b_sz,
        args.inf_b_sz,
        args.gumbel,
        args.cut_coeff,
        args.bal_coeff,
        args.agg_func,
        args.num_classes,
        args.bfs,
        name,
        time.time())
    nx.nx_pydot.write_dot(graph, filename)
    subprocess.call([r"C:\Program Files (x86)\Graphviz2.38\bin\sfdp.exe", filename, "-Tpng", "-o", filename + ".png"])


if __name__ == '__main__':
    if args.seed == 0:
        args.seed = random.randint(100, 999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)

    # load data
    ds = args.dataSet
    dataCenter = DataCenter(config)
    dataCenter.load_dataSet(ds)
    features = torch.FloatTensor(getattr(dataCenter, ds + '_feats')).to(device)
    features_train = torch.FloatTensor(getattr(dataCenter, ds + '_feats_train')).to(device)
    features_test = torch.FloatTensor(getattr(dataCenter, ds + '_feats_test')).to(device)
    features_val = torch.FloatTensor(getattr(dataCenter, ds + '_feats_val')).to(device)

    # self, num_layers, input_size, out_size, device, gcn=False, agg_func='MEAN')
    adj_list = getattr(dataCenter, ds + '_adj_list')
    adj_list_train = getattr(dataCenter, ds + '_adj_list_train')
    adj_list_val = getattr(dataCenter, ds + '_adj_list_val')
    adj_list_test = getattr(dataCenter, ds + '_adj_list_test')

    train_nodes = getattr(dataCenter, ds + "_train")
    val_nodes = getattr(dataCenter, ds + "_val")
    test_nodes = getattr(dataCenter, ds + '_test')

    gnn_num_layers = config['setting.num_layers']
    gnn_emb_size = config['setting.hidden_emb_size']

    graphSage = GraphSage(gnn_num_layers, features.size(1), gnn_emb_size,
                          device,
                          gcn=args.gcn, agg_func=args.agg_func)
    graphSage.to(device)

    if args.num_classes == 0:
        num_labels = len(set(getattr(dataCenter, ds + '_labels')))
    else:
        num_labels = args.num_classes
    gap = Classification(gnn_emb_size, num_labels, gumbel=args.gumbel)
    gap.to(device)

    unsupervised_loss = UnsupervisedLoss(adj_list_train, train_nodes, device)

    if args.learn_method == 'sup':
        print('GraphSage with Supervised Learning')
    elif args.learn_method == 'plus_unsup':
        print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
    else:
        print('GraphSage with Net Unsupervised Learning')

    print("cut coeff: ", args.cut_coeff)
    print("bal_coeff: ", args.bal_coeff)
    for epoch in range(args.epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        graphSage, gap = apply_model(train_nodes, features, graphSage, gap,
                                     unsupervised_loss,
                                     args.b_sz,
                                     args.unsup_loss, device, args.learn_method, adj_list_train, num_classes=num_labels,
                                     bfs=args.bfs, cut_coeff=args.cut_coeff, bal_coeff=args.bal_coeff, epoch=epoch)
        # if (epoch + 1) % 2 == 0 and args.learn_method == 'unsup':
        #     classification, args.max_vali_f1 = train_classification(dataCenter, graphSage, classification, ds, device,
        #                                                             args.max_vali_f1, args.name)
        # if args.learn_method != 'unsup':
        val_nodes = getattr(dataCenter, ds + "_val")
        args.max_vali_f1 = evaluate(adj_list_val, val_nodes, features, graphSage, gap, args.bal_coeff, args.cut_coeff,
                                    num_labels,
                                    device, args)

    # val_nodes = [i for i in range(0, 2708)]

    if args.learn_method == "unsup":
        gap = train_gap(train_nodes, features, graphSage, gap, ds, adj_list_train, epochs=args.gap_epochs,
                        b_sz=args.gap_b_sz,
                        cut_coeff=args.cut_coeff, bal_coeff=args.bal_coeff, num_classes=num_labels, device=device)

    # all_nodes = np.random.permutation(len(adj_list_train))

    partition_graph(train_nodes, features, adj_list_train, "train", graphSage, gap, gnn_num_layers, gnn_emb_size,
                    args.inf_b_sz)
    partition_graph(val_nodes, features, adj_list_val, "val", graphSage, gap, gnn_num_layers, gnn_emb_size)
    partition_graph(test_nodes, features, adj_list_test, "test", graphSage, gap, gnn_num_layers, gnn_emb_size)
    # partition_graph(all_nodes, "all", graphSage, gap)

    models = [graphSage, gap]
    # torch.save(models,
    #            "model_{}_ds{}_ep{}_mb{}-{:.0f}.torch".format(args.learn_method, args.dataSet, args.epochs, args.b_sz,
    #                                                          time.time()))
