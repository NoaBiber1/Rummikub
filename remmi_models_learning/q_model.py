import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=159, hidden_dim=256, seed=42, lr=0.001):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self._init_weights(input_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _init_weights(self, input_dim, hidden_dim):
        std = 0.01
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=std)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        nn.init.normal_(self.fc2.bias, mean=0.0, std=std)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=std)
        nn.init.normal_(self.fc3.bias, mean=0.0, std=std)
        nn.init.normal_(self.fc4.weight, mean=0.0, std=std)
        nn.init.normal_(self.fc4.bias, mean=0.0, std=std)

    def forward(self, x):
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        a3 = F.relu(self.fc3(a2))
        q_value = self.fc4(a3)
        return q_value

    def print_weights(self):
        for name, layer in [
            ("fc1", self.fc1),
            ("fc2", self.fc2),
            ("fc3", self.fc3),
            ("fc4", self.fc4),
        ]:
            w = layer.weight.data
            b = layer.bias.data
            print(f"--- {name} ---")
            print(f"  weight shape: {tuple(w.shape)}, "
                  f"mean={w.mean().item():.6f}, std={w.std().item():.6f}, "
                  f"min={w.min().item():.6f}, max={w.max().item():.6f}")
            print(f"  bias shape:   {tuple(b.shape)}, "
                  f"mean={b.mean().item():.6f}, std={b.std().item():.6f}, "
                  f"min={b.min().item():.6f}, max={b.max().item():.6f}")
            print(f"  weight values:\n{w}")
            print(f"  bias values:\n{b}")
            print()