import time
from datetime import datetime

import pyhocon
from tensorboardX import SummaryWriter

from src.args import parser
from src.dataCenter import *
from src.models import *
from src.utils import *

args = parser().parse_args()
filename = "ds-{}_{}_mb-{}_e-{}_se-{}_smb-{}_inf-mb-{}_gumbel-{}_cut-{}_bal-{}_agg-{}-num_classes-{}-bfs-{}-lr-{}-{}.dot".format(
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
    args.lr,
    datetime.now().strftime('%b%d_%H-%M-%S'))
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
if __name__ == '__main__':
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
    train_features = torch.from_numpy(getattr(dataCenter, ds + '_feats_train')).to(device)
    train_adj_list = getattr(dataCenter, ds + '_adj_list_train')

    train_nodes = getattr(dataCenter, ds + "_train")

    if ds in ["fb", "reddit"]:
        train_edges = getattr(dataCenter, ds + "_train_edges")
        val_edges = getattr(dataCenter, ds + "_val_edges")
        test_edges = getattr(dataCenter, ds + "_test_edges")

    gnn_num_layers = config['setting.num_layers']
    gnn_emb_size = config['setting.hidden_emb_size']
    if args.num_classes == 0:
        num_labels = len(set(getattr(dataCenter, ds + '_labels')))
        args.num_classes = num_labels

    classification = None
    if args.model != "":
        [graphSage, classification] = torch.load(args.model)
    else:
        if args.graphsage_model != "":
            graphSage = torch.load(args.graphsage_model)
        else:
            graphSage = GraphSage(gnn_num_layers, train_features.size(1), gnn_emb_size,
                                  device,
                                  gcn=args.gcn, agg_func=args.agg_func)
        if args.classification_model != "":
            classification = torch.load(args.classification_model)
        else:
            classification = None

    if classification is None:
        if args.learn_method == "sup_edge":
            print("DOUBLE EMB SIZE")
            classification = Classification(gnn_emb_size * 2, args.num_classes, device=device, gumbel=args.gumbel)
        else:
            classification = Classification(gnn_emb_size, args.num_classes, device=device, gumbel=args.gumbel)

    graphSage.to(device)
    classification.to(device)

    unsupervised_loss = UnsupervisedLoss(train_adj_list, train_nodes, device)
    best_val_loss = None
    best_val_models = [graphSage, classification]
    for epoch in range(args.epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        graphSage, classification = apply_model(nodes=train_nodes, features=train_features, graphSage=graphSage,
                                                classification=classification,
                                                unsupervised_loss=unsupervised_loss, adj_list=train_adj_list, args=args,
                                                epoch=epoch, tensorboard=tensorboard, device=device)
        # if (epoch + 1) % 2 == 0 and args.learn_method == 'unsup':
        #     classification, args.max_vali_f1 = train_classification(dataCenter, graphSage, classification, ds, device,
        #                                                             args.max_vali_f1, args.name)
        # if args.learn_method != 'unsup':
        # val_nodes = getattr(dataCenter, ds + "_val")
        # val_gap_loss = evaluate(adj_list_val, val_nodes, features, graphSage, gap, device, args)
        # if best_val_loss is None or best_val_loss > val_gap_loss:
        #     best_val_models = [graphSage, gap]
        # if args.b_sz > 0:
        #     tensorboard.add_scalar("val_gap_loss", val_gap_loss.item(), global_step=epoch*args.b_sz)
        # else:
        #     tensorboard.add_scalar("val_gap_loss", val_gap_loss.item(), global_step=epoch)

    # val_nodes = [i for i in range(0, 2708)]
    if args.learn_method == "sup_edge":
        if ds not in ["fb", "reddit", "twitch"]:
            raise Exception("You have to specify edge-based dataset.")
        print("TRAIN SUP EDGE")
        train_edges = getattr(dataCenter, ds + "_train_edges")
        edge_labels = getattr(dataCenter, ds + "_edge_labels")
        classification, graphSage = train_supervised_edge_partitioning(train_edges, train_features, graphSage,
                                                                       classification,
                                                                       train_adj_list,
                                                                       args.num_classes, edge_labels, args.cut_coeff,
                                                                       args.bal_coeff,
                                                                       device, tensorboard, args.sup_b_sz,
                                                                       args.sup_epochs, args.lr)

    models = [graphSage, classification]
    torch.save(graphSage, filename + ".GRAPHSAGE.torch")
    torch.save(models, filename + ".torch")
    torch.save(best_val_models, filename + "BEST-VAL-MODELS.torch")
