import torch.nn as nn
from configs import config


class DNN(nn.Module):
    def __init__(
            self, 
            input, 
            num_layers=config.training.num_layers,
            output=config.training.output_size, 
            p_dropout=config.training.p_dropout
            ):

        super().__init__()

        layers = []
        inp = input

        for _ in range(num_layers):
            out = inp // 2
            layers.extend([
                    nn.Linear(inp, out),
                    nn.BatchNorm1d(out),
                    nn.ReLU(),
                    nn.Dropout(p_dropout)
            ])
            inp = out
        
        layers.append(nn.Linear(inp, output))
        self.model = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.model(x)