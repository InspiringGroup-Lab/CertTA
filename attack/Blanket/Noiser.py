import torch
import torch.nn as nn


class AddNoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AddNoiser, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.independent_size_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.independent_delay_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        z = torch.FloatTensor(size=(x.shape[0], self.input_dim)).to(x.device)
        z.uniform_(-0,0.5)
        size_noise = self.independent_size_mlp(z)
        delay_noise_ms = self.independent_delay_mlp(z)
        return size_noise, delay_noise_ms 


class InsertNoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InsertNoiser, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.independent_where_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.independent_size_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.independent_delay_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        z = torch.FloatTensor(size=(x.shape[0], self.input_dim)).to(x.device)
        z.uniform_(-0,0.5)
        where_noise = self.independent_where_mlp(z)
        size_noise = self.independent_size_mlp(z)
        size_noise = size_noise / torch.max(size_noise, dim=1, keepdim=True)[0].repeat(1, size_noise.shape[1]) * 1460
        delay_noise_ms = self.independent_delay_mlp(z)
        delay_noise_ms = delay_noise_ms / torch.max(delay_noise_ms, dim=1, keepdim=True)[0].repeat(1, delay_noise_ms.shape[1]) * 256
        return where_noise, size_noise, delay_noise_ms