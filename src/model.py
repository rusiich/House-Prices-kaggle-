import torch.nn as nn
from configs import config


class DNN(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers=config.training.num_layers,
        output_size=config.training.output_size,
        p_dropout=config.training.p_dropout
    ):
        super().__init__()

        layers = []
        inp = input_size

        for n in range(num_layers):
            out = 2 ** (num_layers - n) * output_size
            layers.extend([
                nn.Linear(inp, out),
                nn.BatchNorm1d(out),
                nn.ReLU(),
                nn.Dropout(p_dropout)
            ])
            inp = out

        layers.append(nn.Linear(inp, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)