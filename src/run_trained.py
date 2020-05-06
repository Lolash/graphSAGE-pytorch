import time

import pyhocon
from tensorboardX import SummaryWriter

from src.args import parser
from src.dataCenter import *
from src.models import *
from src.partition import partition_graph, partition_edge_stream_assign_edges
from src.utils import *

print("RUN TRAINED")
args = parser().parse_args()
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

if args.learn_method in ["unsup", "gap", "gap_plus"]:
    if not args.only_edges:
        partition_graph(train_nodes, features, adj_list_train, "train", graphSage, classification, gnn_num_layers,
                        gnn_emb_size,
                        num_labels=num_labels, args=args, batch_size=args.inf_b_sz)
        partition_graph(val_nodes, features, adj_list_val, "val", graphSage, classification, gnn_num_layers, gnn_emb_size,
                        num_labels=num_labels, args=args, batch_size=args.inf_b_sz)
        partition_graph(test_nodes, features, adj_list_test, "test", graphSage, classification, gnn_num_layers,
                        gnn_emb_size,
                        num_labels=num_labels, args=args, batch_size=args.inf_b_sz)
    if ds in ["fb", "reddit"]:
        train_edges = getattr(dataCenter, ds + "_train_edges")
        val_edges = getattr(dataCenter, ds + "_val_edges")
        test_edges = getattr(dataCenter, ds + "_test_edges")
        partition_edge_stream_assign_edges(train_edges, adj_list_train, features, graphSage, classification, "train", args)
        partition_edge_stream_assign_edges(val_edges, adj_list_train, features, graphSage, classification, "val", args)
        partition_edge_stream_assign_edges(test_edges, adj_list_train, features, graphSage, classification, "test", args)

elif args.learn_method == "sup_edge" and ds in ["fb", "reddit"]:
    train_edges = getattr(dataCenter, ds + "_train_edges")
    val_edges = getattr(dataCenter, ds + "_val_edges")
    test_edges = getattr(dataCenter, ds + "_test_edges")
    partition_and_eval_edge_stream_sup_edge(train_edges, adj_list_train, features, graphSage, classification, "train",
                                            args.num_classes, args.inf_b_sz)
    partition_and_eval_edge_stream_sup_edge(test_edges, adj_list_train, features, graphSage, classification, "test",
                                            args.num_classes, args.inf_b_sz)
    partition_and_eval_edge_stream_sup_edge(test_edges, defaultdict(set), features, graphSage, classification,
                                            "test-no-train-edges",
                                            args.num_classes, args.inf_b_sz)

else:
    raise Exception("Unsupported learn method.")
