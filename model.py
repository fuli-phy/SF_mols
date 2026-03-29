import json
import os
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, GCNConv
from torch_geometric.utils import scatter

ELEMENT_TABLE = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
    101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',
    110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
}

def prepare_atom_vector(filename: str = "atom_init.json") -> dict:
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found in current directory.")
        return {}
    with open(filename, 'r') as f:
        elem_embedding = json.load(f)
        return {ELEMENT_TABLE[int(key)]: value for key, value in elem_embedding.items()}


class RBFExpansion(nn.Module):
    def __init__(self, vmin: float = 0, vmax: float = 5, bins: int = 20, lengthscale: Optional[float] = None):
        super().__init__()
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        if lengthscale is None:
            self.gamma = 1.0 / (self.centers[1] - self.centers[0])
        else:
            self.gamma = 1.0 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)

class CGCNN(nn.Module):
    """Crystal Graph Convolutional Neural Network"""
    def __init__(self, atom_dim: int, edge_dim: int, hidden_dim: int, 
                 num_layers: int = 10, aggr: str = 'add', bn: bool = True, bias: bool = True):
        super().__init__()
        self.convs = nn.ModuleList([
            CGConv(atom_dim, edge_dim, aggr=aggr, batch_norm=bn, bias=bias) 
            for _ in range(num_layers)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.leaky_relu(x)
            
        x = self.mlp(x)
        return scatter(x, data.batch, dim=0, reduce='mean')


def load_molecular_data(data_dir: str, nsamp: int = -1, label_type: str = 's1') -> Tuple[List[Data], torch.Tensor]:
    """加载并构建用于训练的图列表和对应标签"""
    print(f"Loading data from {data_dir} for label: {label_type}...")
    atom_features = prepare_atom_vector()
    
    label_files = {
        'egap': 'gap.pt', 'ehomo': 'homo.pt', 'elumo': 'lumo.pt',
        's1': 's1.pt', 't1': 't1.pt', 'fs1': 'fs1.pt', 'score': 'score.pt'
    }
    
    try:
        edge_index = torch.load(os.path.join(data_dir, "edge_index.pt"), weights_only=False)
        edge_vector = torch.load(os.path.join(data_dir, "edge_vector.pt"), weights_only=False)
        elements = torch.load(os.path.join(data_dir, "ele.pt"), weights_only=False)
        target_y = torch.load(os.path.join(data_dir, label_files[label_type]), weights_only=False)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing required .pt files in {data_dir}: {e}")

    num_samples = len(edge_index) if nsamp < 0 else nsamp
    print(f"Total samples to process: {num_samples}")

    rbf = RBFExpansion()
    graph_list = []

    for i in range(num_samples):
        x = torch.tensor([atom_features[e] for e in elements[i]], dtype=torch.float)
        bond_length = torch.norm(edge_vector[i].float(), dim=1)
        edge_attr = torch.squeeze(rbf(bond_length), dim=1)
        
        graph = Data(x=x, edge_index=edge_index[i], edge_attr=edge_attr)
        graph_list.append(graph)

    return graph_list, torch.tensor(target_y[:num_samples], dtype=torch.float).view(-1, 1)


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    for data, y in loader:
        data, y = data.to(device), y.to(device)
        out = model(data)
        total_loss += loss_fn(out, y).item()
    return total_loss / len(loader)

def train_model(model, dataset, labels, batch_size, lr, epochs, device):
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_train = int(len(dataset) * 0.8)
    combined = list(zip(dataset, labels))
    train_loader = DataLoader(combined[:n_train], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(combined[n_train:], batch_size=batch_size)

    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for data, y in train_loader:
            data, y = data.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        train_loss = evaluate_loss(model, train_loader, loss_fn, device)
        test_loss = evaluate_loss(model, test_loader, loss_fn, device)
        
        print(f"Epoch {epoch:04d} | Train MAE: {train_loss:.5f} | Test MAE: {test_loss:.5f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f"best_model.pth")


if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DATA_PATH = '/data/home/fuli/work/GPU5_fuli/24_Singlet_fission/singlet_fission_dataset_cutoff5/'
    
    params = {
        "atom_dim": 92,
        "edge_dim": 20,
        "hidden_dim": 32,
        "batch_size": 96,
        "lr": 0.0005,
        "epochs": 3000
    }

    graphs, labels = load_molecular_data(DATA_PATH, label_type='s1')
    
    net = CGCNN(
        atom_dim=params["atom_dim"], 
        edge_dim=params["edge_dim"], 
        hidden_dim=params["hidden_dim"]
    ).to(dev)

    print(f"Starting training on {dev}...")
    train_model(
        net, graphs, labels, 
        batch_size=params["batch_size"], 
        lr=params["lr"], 
        epochs=params["epochs"], 
        device=dev
    )

