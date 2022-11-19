import torch
import torch.nn as nn

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.criterion = nn.BCELoss()
       
    def forward(self, Sf):
        device = Sf.device
        labels = torch.ones(Sf.shape[0]).to(device)
        return self.criterion(Sf, labels)

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, Sr, Sw, Sf):
        device = Sf.device
        real_labels = torch.ones(Sr.shape[0]).to(device)
        fake_labels = torch.zeros(Sr.shape[0]).to(device)
        return self.criterion(Sr, real_labels) + 0.5*(self.criterion(Sw, fake_labels) + self.criterion(Sf, fake_labels))