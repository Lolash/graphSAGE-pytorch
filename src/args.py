import argparse


def parser():
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
    parser.add_argument('--gcn_coeff', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--bfs', type=int, default=0)
    parser.add_argument('--graphsage_model', type=str, default="")
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--gcn', action='store_true')
    parser.add_argument('--learn_method', type=str, default='sup')
    parser.add_argument('--unsup_loss', type=str, default='normal')
    parser.add_argument('--max_vali_f1', type=float, default=0)
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--config', type=str, default='./src/experiments.conf')

    return parser
