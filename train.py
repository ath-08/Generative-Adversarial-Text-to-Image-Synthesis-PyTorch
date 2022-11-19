import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import warnings

from dataset import ImageGenDataset
from model import Generator, Discriminator
from loss import GeneratorLoss, DiscriminatorLoss
from utils import weights_init, plot_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", module="matplotlib\..*")

batch_size = 128
lr = 0.0002
betas = (0.5, 0.999)
print_interval = 25
datasetFile = 'flowers.hdf5'

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

train_dataset = ImageGenDataset(filename=datasetFile, split='train', transform=transform)
val_dataset = ImageGenDataset(filename=datasetFile, split='valid', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

generator_loss = GeneratorLoss()
discriminator_loss = DiscriminatorLoss()



def train_epoch(epoch):
    generator.train()
    discriminator.train()

    train_loss_g_acc = 0
    train_loss_d_acc = 0
    d_x_acc = 0
    d_g_x_acc = 0
    plotted = False

    for batch_idx, data in enumerate(train_loader):
        right_txt_embeds = data['right_txt_embeds'].to(device)
        wrong_txt_embeds = data['wrong_txt_embeds'].to(device)
        real_img = data['real_img'].to(device)

        noise = torch.randn(real_img.shape[0], 100, 1, 1).to(device)

        discriminator_optimizer.zero_grad()
        fake_img = generator(noise, right_txt_embeds)
        Sr = discriminator(real_img, right_txt_embeds)  # real image, right text
        Sw = discriminator(real_img, wrong_txt_embeds)  # real image, wrong text
        Sf1 = discriminator(fake_img, right_txt_embeds)  # fake image, right text
        loss_d = discriminator_loss(Sr, Sw, Sf1)
        loss_d.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()
        fake_img = generator(noise, right_txt_embeds)
        Sf2 = discriminator(fake_img, right_txt_embeds) 
        loss_g = generator_loss(Sf2)
        loss_g.backward()
        generator_optimizer.step()

        if batch_idx%print_interval == 0:
            print('Epoch', epoch, 'batch_idx', batch_idx, ': loss_d =', loss_d.item(), 'loss_g =', loss_g.item(), 'D(X) =', Sr.mean().item(), 'D(G(X)) =', Sf1.mean().item(), flush=True)

        if not plotted:
            plotted = True
            plot_images(fake_img, generated=True, train=True, epoch=epoch, batch_idx=batch_idx)
            plot_images(real_img, generated=False, train=True, epoch=epoch, batch_idx=batch_idx)
            
            train_loss_g_acc += loss_g.item()*right_txt_embeds.shape[0]
            train_loss_d_acc += loss_d.item()*right_txt_embeds.shape[0]
            d_x_acc += Sr.sum().item()
            d_g_x_acc += Sf1.sum().item()

    train_loss_g_acc /= len(train_loader.dataset)
    train_loss_d_acc /= len(train_loader.dataset)
    d_x_acc /= len(train_loader.dataset)
    d_g_x_acc /= len(train_loader.dataset)

    return train_loss_g_acc, train_loss_d_acc, d_x_acc, d_g_x_acc

def validate(epoch):
    generator.eval()
    discriminator.eval()

    val_loss_g_acc = 0
    val_loss_d_acc = 0
    d_x_acc = 0
    d_g_x_acc = 0
    plotted = False

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            right_txt_embeds = data['right_txt_embeds'].to(device)
            wrong_txt_embeds = data['wrong_txt_embeds'].to(device)
            real_img = data['real_img'].to(device)

            noise = torch.randn(real_img.shape[0], 100, 1, 1).to(device)
            
            fake_img = generator(noise, right_txt_embeds)
            Sr = discriminator(real_img, right_txt_embeds)  # real image, right text
            Sw = discriminator(real_img, wrong_txt_embeds)  # real image, wrong text
            Sf = discriminator(fake_img, right_txt_embeds)  # fake image, right text
            loss_d = discriminator_loss(Sr, Sw, Sf)
            loss_g = generator_loss(Sf)

            val_loss_g_acc += loss_g.item()*right_txt_embeds.shape[0]
            val_loss_d_acc += loss_d.item()*right_txt_embeds.shape[0]
            d_x_acc += Sr.sum().item()
            d_g_x_acc += Sf.sum().item()

            if not plotted:
                plotted = True
                plot_images(fake_img, generated=True, train=False, epoch=epoch, batch_idx=batch_idx)
                plot_images(real_img, generated=False, train=False, epoch=epoch, batch_idx=batch_idx)

    val_loss_g_acc /= len(train_loader.dataset)
    val_loss_d_acc /= len(train_loader.dataset)
    d_x_acc /= len(train_loader.dataset)
    d_g_x_acc /= len(train_loader.dataset)

    return val_loss_g_acc, val_loss_d_acc, d_x_acc, d_g_x_acc

def train(epochs):

    train_stats = pd.DataFrame(columns=['epoch', 'loss_d', 'loss_g', 'D(X)', 'D(G(X))'])
    val_stats = pd.DataFrame(columns=['epoch', 'loss_d', 'loss_g', 'D(X)', 'D(G(X))'])

    for epoch in range(1, epochs+1):
        train_loss_g_acc, train_loss_d_acc, train_d_x_acc, train_d_g_x_acc = train_epoch(epoch)
        val_loss_g_acc, val_loss_d_acc, val_d_x_acc, val_d_g_x_acc = validate(epoch)
        
        print('\nEpoch', epoch, ': Train loss_d =', train_loss_d_acc, 'loss_g =', train_loss_g_acc, 'D(X) =', train_d_x_acc, 'D(G(X)) =', train_d_g_x_acc, flush=True)
        print('Epoch', epoch, ': Val loss_d =', val_loss_d_acc, 'loss_g =', val_loss_g_acc, 'D(X) =', val_d_x_acc, 'D(G(X)) =', val_d_g_x_acc, '\n', flush=True)

        train_stats = train_stats.append({'epoch': epoch, 'loss_d': train_loss_d_acc, 'loss_g': train_loss_g_acc, 'D(X)': train_d_x_acc, 'D(G(X))': train_d_g_x_acc}, ignore_index=True)
        val_stats = val_stats.append({'epoch': epoch, 'loss_d': val_loss_d_acc, 'loss_g': val_loss_g_acc, 'D(X)': val_d_x_acc, 'D(G(X))': val_d_g_x_acc}, ignore_index=True)

        train_stats.to_csv('train_stats.csv')
        val_stats.to_csv('val_stats.csv')

        if epoch%10 == 0 or epoch == 1:
            checkpoint = {"generator": generator.state_dict(),
                            "discriminator": discriminator.state_dict(),
                            "generator_optimizer": generator_optimizer.state_dict(),
                            "discriminator_optimizer": discriminator_optimizer.state_dict(),
                            "epoch": epoch}
        
            torch.save(checkpoint, "checkpoints/checkpoint_epoch_" + str(epoch) + ".pth")



if __name__ == '__main__':
    train(200)



