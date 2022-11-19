import torch
import torch.nn as nn

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.l1_coef = 50
        self.l2_coef = 100
       
    def forward(self, Sf, activation_fake=None, activation_real=None, fake_img=None, real_img=None):
        device = Sf.device
        labels = torch.ones(Sf.shape[0]).to(device)
        loss = self.criterion(Sf, labels)

        if activation_fake is not None:
            loss = loss + self.l2_coef*self.l2_loss(activation_fake, activation_real) + self.l1_coef*self.l1_loss(fake_img, real_img)
        
        return loss

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, Sr, Sw, Sf):
        device = Sf.device
        real_labels = torch.ones(Sr.shape[0]).to(device)
        fake_labels = torch.zeros(Sr.shape[0]).to(device)
        return self.criterion(Sr, real_labels) + 0.5*(self.criterion(Sw, fake_labels) + self.criterion(Sf, fake_labels))