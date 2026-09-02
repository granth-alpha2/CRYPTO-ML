"""
model.py
========
GNN model definitions for plaintext training.

Two architectures available (set via config.yaml model.type):
  - GCN  : Graph Convolutional Network (Kipf & Welling 2017)
  - GIN  : Graph Isomorphism Network  (Xu et al. 2019)

Both support:
  - Configurable depth (num_layers)
  - ReLU activation (default) OR polynomial activation (for HE compatibility)
  - Dropout regularization

Usage:
    from src.model import build_model
    model = build_model(cfg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv


# -------------------------------------------------------------------
# Polynomial Activation (HE-compatible ReLU approximation)
# -------------------------------------------------------------------

class PolyActivation(nn.Module):
    """
    Degree-2 polynomial approximation of ReLU:
        f(x) = a0 + a1*x + a2*x^2

    Coefficients are learnable, initialized to approximate ReLU.
    Under CKKS homomorphic encryption, only additions and multiplications
    are supported — this replaces the non-linear ReLU which cannot be
    computed directly on ciphertext.
    """
    def __init__(self):
        super().__init__()
        # Initialized to ReLU approximation: 0.5 + 0.5x + ~0 x^2
        self.a0 = nn.Parameter(torch.tensor(0.5))
        self.a1 = nn.Parameter(torch.tensor(0.5))
        self.a2 = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a0 + self.a1 * x + self.a2 * (x ** 2)


def get_activation(name: str) -> nn.Module:
    """Return activation module by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "poly":
        return PolyActivation()
    else:
        raise ValueError(f"Unknown activation: {name}. Choose 'relu' or 'poly'.")


# -------------------------------------------------------------------
# GCN Model
# -------------------------------------------------------------------

class GCNModel(nn.Module):
    """
    Graph Convolutional Network for node classification.

    Architecture:
        Input (166 features)
            -> GCNConv + Activation + Dropout   (x num_layers-1)
            -> GCNConv
            -> Softmax (2 classes: licit / illicit)

    Args:
        in_channels  (int)  : Number of input node features (166)
        hidden_dim   (int)  : Hidden layer dimension
        num_classes  (int)  : Number of output classes (2)
        num_layers   (int)  : Number of GCN layers (keep <= 2 for HE)
        dropout      (float): Dropout probability
        activation   (str)  : 'relu' or 'poly'
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.acts  = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.acts.append(get_activation(activation))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.acts.append(get_activation(activation))

        # Output layer
        self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, act) in enumerate(zip(self.convs[:-1], self.acts)):
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer — no activation, no dropout
        x = self.convs[-1](x, edge_index)
        return x  # raw logits

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings from the second-to-last layer (for analysis)."""
        for conv, act in zip(self.convs[:-1], self.acts):
            x = conv(x, edge_index)
            x = act(x)
        return x


# -------------------------------------------------------------------
# GIN Model
# -------------------------------------------------------------------

class GINModel(nn.Module):
    """
    Graph Isomorphism Network for node classification.

    More expressive than GCN (Weisfeiler-Lehman test equivalent).
    Slightly more expensive under HE due to MLP per layer.

    Args:
        in_channels  (int)  : Number of input node features
        hidden_dim   (int)  : Hidden layer dimension
        num_classes  (int)  : Number of output classes
        num_layers   (int)  : Number of GIN layers
        dropout      (float): Dropout probability
        activation   (str)  : 'relu' or 'poly'
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()

        dims = [in_channels] + [hidden_dim] * (num_layers - 1) + [num_classes]

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            mlp = nn.Sequential(
                nn.Linear(in_d, out_d),
                get_activation(activation),
                nn.Linear(out_d, out_d),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(out_d))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # raw logits


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """
    Build and return a GNN model from config.

    Args:
        cfg (dict): Loaded config.yaml as dict

    Returns:
        model (nn.Module): Untrained GNN model
    """
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    kwargs = dict(
        in_channels = dataset_cfg["node_features"],   # 166
        hidden_dim  = model_cfg["hidden_dim"],
        num_classes = dataset_cfg["num_classes"],      # 2
        num_layers  = model_cfg["num_layers"],
        dropout     = model_cfg["dropout"],
        activation  = model_cfg["activation"],
    )

    model_type = model_cfg["type"].upper()
    if model_type == "GCN":
        return GCNModel(**kwargs)
    elif model_type == "GIN":
        return GINModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose GCN or GIN.")


if __name__ == "__main__":
    # Quick sanity check
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
