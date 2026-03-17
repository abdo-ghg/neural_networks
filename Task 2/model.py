import micrograd.nn as nn

class MLP(nn.Module):
    def __init__(self, bias=True, features=5, classes=3, tanh_hidden=True, num_of_hidden_layers=1, num_hidden_neurons=4 * 16):
        super().__init__()
        self.num_of_hidden_layers = num_of_hidden_layers
        
        self.c_fc    = nn.Linear(features, num_hidden_neurons, bias=bias)
        self.hidden    = nn.Tanh() if tanh_hidden else nn.Sigmoid()

        for i in range(num_of_hidden_layers - 1):
            setattr(self, f'hidden_{i}', nn.Linear(num_hidden_neurons, num_hidden_neurons, bias=bias))

        self.c_proj  = nn.Linear(num_hidden_neurons, classes, bias=bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.hidden(x)
        for i in range(self.num_of_hidden_layers - 1):
            x = getattr(self, f'hidden_{i}')(x)
        x = self.c_proj(x)
        return x