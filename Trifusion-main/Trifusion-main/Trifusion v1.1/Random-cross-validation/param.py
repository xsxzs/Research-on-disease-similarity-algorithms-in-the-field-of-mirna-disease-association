import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--path', type=str, default='./new_data')
    parser.add_argument('--epoch', type=int, default=110)
    parser.add_argument('--SEED', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kfolds', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--numnodes', type=int, default=4250)
    # parser.add_argument('--hidden', type=int, default=512)
    # parser.add_argument('--numnodes', type=int, default=1444)
    parser.add_argument('--MLPDropout', type=float, default=0.4)
    parser.add_argument('--status', type=int, default=1)
    parser.add_argument('--numrna', type=int, default=1890)
    parser.add_argument('--numdis', type=int, default=2360)
    # parser.add_argument('--numrna', type=int, default=853)
    # parser.add_argument('--numdis', type=int, default=591)
    parser.add_argument('--numlayer', type=int, default=3)
    parser.add_argument('--numhead', type=int, default=8)
    parser.add_argument('--numneighbor', type=int, default=40)

    return parser.parse_args()