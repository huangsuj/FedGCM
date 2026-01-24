import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dirichlet_alpha', help='dirichlet distribution parameter', type=float,
                    default=0.5)

parser.add_argument('--gpu_id', help='', type=int,
                    default=0)

parser.add_argument('--threshold', help='', type=float,
                    default=0.9)  #

parser.add_argument('--alpha', help='', type=float,
                    default=0.5)

##
parser.add_argument('--seed', help='seed everything', type=int,
                    default=2023)

parser.add_argument('--partition', help='data simulation method', type=str,
                    default="Louvain")

parser.add_argument('--data_name', help='dataset name', type=str,
                    default="Chameleon")

parser.add_argument('--num_clients', help='number of clients', type=int,
                    default=10)

parser.add_argument('--gmodel_name', help='global model name', type=str,
                    default="ACM")

parser.add_argument('--num_rounds', help='number of global model training rounds', type=int,
                    default=300)

parser.add_argument('--num_epochs', help='number of global model local training epochs', type=int,
                    default=5)

parser.add_argument('--lr', help='global model learning rate', type=float,
                    default=1e-2)
parser.add_argument(
    "--structure_info",
    type=int,
    default=0,
    help="1 for using structure information in acmgcnp, 0 for not",
)

parser.add_argument('--weight_decay', help='global model weight decay', type=float,
                    default=0.0)

parser.add_argument('--drop', help='global model drop out prob', type=float,
                    default=0.3)

parser.add_argument('--normalize_train', help='number of personal local model training times', type=int,
                    default=5)

parser.add_argument('--hidden_dim', help='multi layer model hidden units', type=int,
                        default=64)

parser.add_argument('--K', help='group size', type=int,
                    default=4)
args = parser.parse_args()
print(args)

