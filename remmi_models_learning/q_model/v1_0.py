import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=159, hidden_dim=256, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self._init_weights(input_dim, hidden_dim)

    def _init_weights(self):
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