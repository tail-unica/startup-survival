import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        # Se viene passato un intero singolo, trasformazione in lista
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
            
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Layer intermedi (Hidden -> Hidden)
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            
        # Ultimo layer (Ultimo Hidden -> Output)
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x