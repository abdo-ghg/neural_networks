import micrograd.nn as nn

class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(5, 4 * 16, bias=bias)
        self.relu    = nn.ReLU()
        self.c_proj  = nn.Linear(4 * 16, 3, bias=bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x