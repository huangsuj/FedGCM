import torch
import warnings
from config import args
from fedKnowledge_main import step_main
warnings.filterwarnings('ignore')
torch.manual_seed(42)


if __name__ == "__main__":
    table_rows = []
    if args.data_name in ['Cora', 'penn94', 'amazon-ratings']:
        args.alpha = 0.2
    step_main(table_rows)