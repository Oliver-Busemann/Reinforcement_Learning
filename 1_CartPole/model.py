import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


action_space = 2
obs_space = 4

hidden_size = 128
drop_rate = 0.2

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(obs_space, hidden_size)
        self.drop_1 = nn.Dropout(p=drop_rate)
        self.fc_2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = self.drop_1(x)
        x = self.fc_2(x)
        return x

class NN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-2
        self.loss = nn.CrossEntropyLoss()
        self.model = model()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
 
        pred = self.model(x)

        loss = self.loss(pred, y)

        self.log('CrossEntropyLoss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)