import torch
import warnings
from utils import set_seed
from config import args
from roles.server import ServerManager
from roles.client import ClientsManager
from datasets.graph_fl_dataset import GraphFLDataset
warnings.filterwarnings('ignore')
torch.manual_seed(42)

def step_main(table_rows):
    gpu_id = args.gpu_id
    num_clients = args.num_clients
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    datasets = GraphFLDataset(
        args,
        device,
        root='./datasets/',
        name=args.data_name,
        sampling=args.partition,
        num_clients=num_clients,
        analysis_local_subgraph=False,
        analysis_global_graph=False
    )

    model_name = args.gmodel_name

    Server = ServerManager(
        args=args,
        model_name=model_name,
        datasets=datasets,
        num_clients=num_clients,
        device=device,
        num_rounds=args.num_rounds,
        client_sample_ratio=1
    )

    client_manager = ClientsManager(
        model_name=model_name,
        datasets=datasets,
        num_clients=num_clients,
        device=device,
        eval_single_client=False,
    )

    print("| ★  Data simulation: {}, Client: {}, Model name: {}".format(args.partition, args.num_clients,
                                                                        args.gmodel_name))
    print("| ")


    Server.collaborative_training_model(
        table_rows,
        client_manager.clients,
        datasets.name,
        datasets.num_clients,
        datasets.sampling,
        model_name=model_name,
        normalize_trains=args.normalize_train)

    








    



