import argparse
import torch

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epoch1', type=int, default=300,  # 500
                        help='Number of epochs to train.')
    parser.add_argument('--epoch2', type=int, default=50,  # 50
                        help='Number of epochs to train.')
    parser.add_argument('--epoch3', type=int, default=300,  # 300
                        help='Number of epochs to train.')
    parser.add_argument('--Alpha', type=int, default=50,
                        help='Alpha parameter.')
    parser.add_argument('--Beta', type=int, default=5,
                        help='Beta parameter.')
    parser.add_argument('--Garma', type=int, default=1,
                        help='Alpha parameter.')
    parser.add_argument('--Topk', type=int, default=10,
                        help='Alpha parameter.')
    parser.add_argument('--lr', type=float, default=0.001,  # 0.01
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,  # 5e-6
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.3,  # 0.3
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="credit",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="GCN",
                        choices=["GCN", "GIN","GAT"],
                        help='model to use.')
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
