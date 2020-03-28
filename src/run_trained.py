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


def partition_graph(nodes, features, adj_list, name, graphsage, gap, gnn_num_layers, gnn_emb_size, num_labels,
                    batch_size=-1):
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
    balanced = np.array([int(len(graph.nodes) / num_labels)] * len(cardinalities))
    print(cardinalities)
    print(balanced)
    print(cardinalities - balanced)
    balancedness = 1 - ((cardinalities - balanced) ** 2).mean()

    perf = nx.algorithms.community.performance(graph, partitions.values())
    coverage = nx.algorithms.community.coverage(graph, partitions.values())

    print("Performance of {}: {}".format(name, perf))
    print("Coverage of {}: {}".format(name, coverage))
    print("Balancedness of {}: {}".format(name, balancedness))
    print("Edge cut of {}: {}".format(name, 1.0 - coverage))

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

#
# def partition_edge_stream(graphsage, gap, edge_stream, features, training_graph, training_embeddings, args):
#
#
#
# print("Started")
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
features_test = torch.FloatTensor(getattr(dataCenter, ds + '_feats_test')).to(device)
features_val = torch.FloatTensor(getattr(dataCenter, ds + '_feats_val')).to(device)

# self, num_layers, input_size, out_size, device, gcn=False, agg_func='MEAN')
adj_list = getattr(dataCenter, ds + '_adj_list')
adj_list_train = getattr(dataCenter, ds + "_adj_list_train")
adj_list_val = getattr(dataCenter, ds + '_adj_list_val')
adj_list_test = getattr(dataCenter, ds + '_adj_list_test')

train_nodes = getattr(dataCenter, ds + "_train")
val_nodes = getattr(dataCenter, ds + "_val")
test_nodes = getattr(dataCenter, ds + '_test')

gnn_num_layers = config['setting.num_layers']
gnn_emb_size = config['setting.hidden_emb_size']

[graphSage, gap] = torch.load(
    "model-reddit_gap_edge_mb10_e20_ge200_gmb20_gumbel-0_cut1.0_bal1.0_aggMEAN-1585336080.9641972.torch")
graphSage.to(device)

if args.num_classes == 0:
    num_labels = len(set(getattr(dataCenter, ds + '_labels')))
else:
    num_labels = args.num_classes
gap.to(device)

partition_graph(train_nodes, features, adj_list_train, "train", graphSage, gap, gnn_num_layers, gnn_emb_size,
                num_labels=num_labels, batch_size=args.inf_b_sz)
# partition_graph(val_nodes, features, adj_list_val, "val", graphSage, gap, gnn_num_layers, gnn_emb_size,
#                 num_labels=num_labels, batch_size=args.inf_b_sz)
partition_graph(test_nodes, features, adj_list_test, "test", graphSage, gap, gnn_num_layers, gnn_emb_size,
                num_labels=num_labels, batch_size=args.inf_b_sz)
