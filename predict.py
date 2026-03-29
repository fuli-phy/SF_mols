import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from typing import List
from model import CGCNN, RBFExpansion, ELEMENT_TABLE, prepare_atom_vector

class Predictor:
    def __init__(self, model_path: str, params: dict, device: torch.device):
        self.device = device
        self.model = self._load_model(model_path, params)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str, params: dict):
        try:
            model = torch.load(model_path, map_location=self.device)
            if isinstance(model, dict):
                net = CGCNN(
                    atom_dim=params['atom_dim'], 
                    edge_dim=params['edge_dim'], 
                    hidden_dim=params['hidden_dim']
                )
                net.load_state_dict(model)
                return net
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @torch.no_grad()
    def run(self, loader: DataLoader) -> List[float]:
        results = []
        print("Starting prediction...")
        for data in tqdm(loader, desc="Predicting"):
            data = data.to(self.device)
            output = self.model(data)
            results.extend(output.cpu().numpy().flatten().tolist())
        return results


def load_prediction_data(data_dir: str, nsamp: int = -1) -> List:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Loading raw data from {data_dir}...")
    atom_features = prepare_atom_vector("atom_init.json")

    edge_index = torch.load(os.path.join(data_dir, "edge_index.pt"), weights_only=False)
    edge_vector = torch.load(os.path.join(data_dir, "edge_vector.pt"), weights_only=False)
    elements = torch.load(os.path.join(data_dir, "ele.pt"), weights_only=False)

    num_samples = len(edge_index) if nsamp < 0 else min(nsamp, len(edge_index))
    rbf = RBFExpansion(vmin=0, vmax=5, bins=20) # 保持与 predict.py 原逻辑一致的 vmax=5
    
    graph_list = []
    for i in range(num_samples):
        x = torch.tensor([atom_features[e] for e in elements[i]], dtype=torch.float)
        bond_length = torch.norm(edge_vector[i].float(), dim=1)
        edge_attr = rbf(bond_length)
        
        from torch_geometric.data import Data
        graph = Data(x=x, edge_index=edge_index[i], edge_attr=edge_attr)
        graph_list.append(graph)
    return graph_list

if __name__ == "__main__":
    CONFIG = {
        "model_file": 'best_model.pth',
        "data_dir": '/data/home/fuli/work/GPU5_fuli/24_Singlet_fission/singlet_fission_dataset_cutoff3/',
        "output_file": 'prediction_results.csv',
        "params": {
            "atom_dim": 92,
            "edge_dim": 20,
            "hidden_dim": 32,
            "batch_size": 128
        },
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }

    try:
        graphs = load_prediction_data(CONFIG["data_dir"])
        loader = DataLoader(graphs, batch_size=CONFIG["params"]["batch_size"], shuffle=False)
    except Exception as e:
        print(f"Data loading failed: {e}")
        exit()

    predictor = Predictor(
        model_path=CONFIG["model_file"],
        params=CONFIG["params"],
        device=CONFIG["device"]
    )

    predictions = predictor.run(loader)
    print("Success prediction, Xixi")
    df = pd.DataFrame({"Predicted_Value": predictions})
    df.to_csv(CONFIG["output_file"], index=False)
    print(f"Results saved to {CONFIG['output_file']}")