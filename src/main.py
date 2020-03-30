import pyhocon
from tensorboardX import SummaryWriter

from src.dataCenter import *
from src.models import *
from src.partition import partition_graph, partition_edge_stream
from src.args import parser
from src.utils import *

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
print("ELO")
print(__name__)
if __name__ == '__main__':
    print("ELO")
    print(args)
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
    if args.num_classes == 0:
        num_labels = len(set(getattr(dataCenter, ds + '_labels')))
        args.num_classes = num_labels

    if args.model != "":
        [graphSage, gap] = torch.load(args.model)
    else:
        graphSage = GraphSage(gnn_num_layers, features.size(1), gnn_emb_size,
                              device,
                              gcn=args.gcn, agg_func=args.agg_func)

        gap = Classification(gnn_emb_size, args.num_classes, device=device, gumbel=args.gumbel)
    graphSage.to(device)
    gap.to(device)

    unsupervised_loss = UnsupervisedLoss(adj_list_train, train_nodes, device)

    if args.learn_method == 'sup':
        print('GraphSage with Supervised Learning')
    elif args.learn_method == 'plus_unsup':
        print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
    else:
        print('GraphSage with Net Unsupervised Learning')

    for epoch in range(args.epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        graphSage, gap = apply_model(train_nodes, features, graphSage, gap,
                                     unsupervised_loss,
                                     args.b_sz,
                                     args.unsup_loss, device, args.learn_method, adj_list_train,
                                     num_classes=args.num_classes,
                                     bfs=args.bfs, cut_coeff=args.cut_coeff, bal_coeff=args.bal_coeff, epoch=epoch,
                                     tensorboard=tensorboard)
        # if (epoch + 1) % 2 == 0 and args.learn_method == 'unsup':
        #     classification, args.max_vali_f1 = train_classification(dataCenter, graphSage, classification, ds, device,
        #                                                             args.max_vali_f1, args.name)
        # if args.learn_method != 'unsup':
        # val_nodes = getattr(dataCenter, ds + "_val")
        # args.max_vali_f1 = evaluate(adj_list_val, val_nodes, features, graphSage, gap, args.bal_coeff, args.cut_coeff,
        #                             num_labels,
        #                             device, epoch, args)

    # val_nodes = [i for i in range(0, 2708)]

    if args.learn_method == "unsup":
        gap = train_gap(train_nodes, features, graphSage, gap, ds, adj_list_train, epochs=args.gap_epochs,
                        b_sz=args.gap_b_sz, cut_coeff=args.cut_coeff, bal_coeff=args.bal_coeff,
                        num_classes=args.num_classes,
                        device=device, tensorboard=tensorboard)
    models = [graphSage, gap]
    torch.save(models, filename + ".torch")
    # all_nodes = np.random.permutation(len(adj_list_train))

    partition_graph(train_nodes, features, adj_list_train, "train", graphSage, gap, gnn_num_layers, gnn_emb_size,
                    num_labels=args.num_classes, args=args, tensorboard=tensorboard, batch_size=args.inf_b_sz)
    partition_graph(val_nodes, features, adj_list_val, "val", graphSage, gap, gnn_num_layers, gnn_emb_size,
                    num_labels=args.num_classes, args=args, tensorboard=tensorboard, batch_size=args.inf_b_sz)
    partition_graph(test_nodes, features, adj_list_test, "test", graphSage, gap, gnn_num_layers, gnn_emb_size,
                    num_labels=args.num_classes, args=args, tensorboard=tensorboard, batch_size=args.inf_b_sz)
    # # partition_graph(all_nodes, "all", graphSage, gap)

    train_edges = getattr(dataCenter, ds + "train_edges")
    test_edges = getattr(dataCenter, ds + "test_edges")
    partition_edge_stream(train_edges, adj_list_train, features, graphSage, gap, args)
    partition_edge_stream(test_edges, adj_list_train, features, graphSage, gap, args)
