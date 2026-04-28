import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input, output=2, p_dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(p_dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Dropout(p_dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Dropout(p_dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),


            nn.Dropout(p_dropout),
            nn.Linear(32, output),
        )


    def forward(self, x):
        return self.model(x)