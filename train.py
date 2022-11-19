import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import argparse

from dataset import ImageGenDataset
from model import Generator, Discriminator
from loss import GeneratorLoss, DiscriminatorLoss
from utils import weights_init, plot_images

class GAN_Trainer():
    def __init__(self, epochs, batch_size, lr, dataset_file, checkpoint_interval, print_interval, checkpoints_folder, results_folder):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.print_interval = print_interval
        self.dataset_file = dataset_file
        self.checkpoints_folder = checkpoints_folder
        self.checkpoint_interval = checkpoint_interval
        self.results_folder = results_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.betas = (0.5, 0.999)

        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.train_dataset = ImageGenDataset(filename=self.dataset_file, split='train', transform=self.transform)
        self.val_dataset = ImageGenDataset(filename=self.dataset_file, split='valid', transform=self.transform)

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True)

        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

        self.generator_loss = GeneratorLoss()
        self.discriminator_loss = DiscriminatorLoss()

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        train_loss_g_acc = 0
        train_loss_d_acc = 0
        d_x_acc = 0
        d_g_x_acc = 0
        plotted = False

        for batch_idx, data in enumerate(self.train_loader):
            right_txt_embeds = data['right_txt_embeds'].to(self.device)
            wrong_txt_embeds = data['wrong_txt_embeds'].to(self.device)
            real_img = data['real_img'].to(self.device)

            noise = torch.randn(real_img.shape[0], 100, 1, 1).to(self.device)

            self.discriminator_optimizer.zero_grad()
            fake_img = self.generator(noise, right_txt_embeds)
            Sr, _ = self.discriminator(real_img, right_txt_embeds)  # real image, right text
            Sw, _ = self.discriminator(real_img, wrong_txt_embeds)  # real image, wrong text
            Sf1, _ = self.discriminator(fake_img, right_txt_embeds)  # fake image, right text
            loss_d = self.discriminator_loss(Sr, Sw, Sf1)
            loss_d.backward()
            self.discriminator_optimizer.step()


            self.generator_optimizer.zero_grad()
            fake_img = self.generator(noise, right_txt_embeds)
            Sf2, activation_fake = self.discriminator(fake_img, right_txt_embeds)
            _, activation_real =  self.discriminator(real_img, right_txt_embeds)

            activation_fake = torch.mean(activation_fake, 0)
            activation_real = torch.mean(activation_real, 0)

            loss_g = self.generator_loss(Sf2, activation_fake, activation_real, fake_img, real_img)
            loss_g.backward()
            self.generator_optimizer.step()

            if batch_idx % self.print_interval == 0:
                print('Epoch', epoch, 'batch_idx', batch_idx, ': loss_d =', loss_d.item(), 'loss_g =', loss_g.item(), 'D(X) =', Sr.mean().item(), 'D(G(X)) =', Sf1.mean().item(), flush=True)

            if not plotted:
                plotted = True
                plot_images(results_folder=self.results_folder, images=fake_img, generated=True, train=True, epoch=epoch, batch_idx=batch_idx)
                plot_images(results_folder=self.results_folder, images=real_img, generated=False, train=True, epoch=epoch, batch_idx=batch_idx)
                
                train_loss_g_acc += loss_g.item()*right_txt_embeds.shape[0]
                train_loss_d_acc += loss_d.item()*right_txt_embeds.shape[0]
                d_x_acc += Sr.sum().item()
                d_g_x_acc += Sf1.sum().item()

        train_loss_g_acc /= len(self.train_loader.dataset)
        train_loss_d_acc /= len(self.train_loader.dataset)
        d_x_acc /= len(self.train_loader.dataset)
        d_g_x_acc /= len(self.train_loader.dataset)

        return train_loss_g_acc, train_loss_d_acc, d_x_acc, d_g_x_acc

    def validate(self, epoch):
        self.generator.eval()
        self.discriminator.eval()

        val_loss_g_acc = 0
        val_loss_d_acc = 0
        d_x_acc = 0
        d_g_x_acc = 0
        plotted = False

        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                right_txt_embeds = data['right_txt_embeds'].to(self.device)
                wrong_txt_embeds = data['wrong_txt_embeds'].to(self.device)
                real_img = data['real_img'].to(self.device)

                noise = torch.randn(real_img.shape[0], 100, 1, 1).to(self.device)
                
                fake_img = self.generator(noise, right_txt_embeds)
                Sr, activation_real = self.discriminator(real_img, right_txt_embeds)  # real image, right text
                Sw, _ = self.discriminator(real_img, wrong_txt_embeds)  # real image, wrong text
                Sf, activation_fake = self.discriminator(fake_img, right_txt_embeds)  # fake image, right text
                loss_d = self.discriminator_loss(Sr, Sw, Sf)
                loss_g = self.generator_loss(Sf, activation_fake, activation_real, fake_img, real_img)

                val_loss_g_acc += loss_g.item()*right_txt_embeds.shape[0]
                val_loss_d_acc += loss_d.item()*right_txt_embeds.shape[0]
                d_x_acc += Sr.sum().item()
                d_g_x_acc += Sf.sum().item()

                if not plotted:
                    plotted = True
                    plot_images(results_folder=self.results_folder, images=fake_img, generated=True, train=False, epoch=epoch, batch_idx=batch_idx)
                    plot_images(results_folder=self.results_folder, images=real_img, generated=False, train=False, epoch=epoch, batch_idx=batch_idx)

        val_loss_g_acc /= len(self.train_loader.dataset)
        val_loss_d_acc /= len(self.train_loader.dataset)
        d_x_acc /= len(self.train_loader.dataset)
        d_g_x_acc /= len(self.train_loader.dataset)

        return val_loss_g_acc, val_loss_d_acc, d_x_acc, d_g_x_acc

    def train(self):

        train_stats = pd.DataFrame(columns=['epoch', 'loss_d', 'loss_g', 'D(X)', 'D(G(X))'])
        val_stats = pd.DataFrame(columns=['epoch', 'loss_d', 'loss_g', 'D(X)', 'D(G(X))'])

        for epoch in range(1, self.epochs+1):
            train_loss_g_acc, train_loss_d_acc, train_d_x_acc, train_d_g_x_acc = self.train_epoch(epoch)
            val_loss_g_acc, val_loss_d_acc, val_d_x_acc, val_d_g_x_acc = self.validate(epoch)
            
            print('\nEpoch', epoch, ': Train loss_d =', train_loss_d_acc, 'loss_g =', train_loss_g_acc, 'D(X) =', train_d_x_acc, 'D(G(X)) =', train_d_g_x_acc, flush=True)
            print('Epoch', epoch, ': Val loss_d =', val_loss_d_acc, 'loss_g =', val_loss_g_acc, 'D(X) =', val_d_x_acc, 'D(G(X)) =', val_d_g_x_acc, '\n', flush=True)

            train_stats = train_stats.append({'epoch': epoch, 'loss_d': train_loss_d_acc, 'loss_g': train_loss_g_acc, 'D(X)': train_d_x_acc, 'D(G(X))': train_d_g_x_acc}, ignore_index=True)
            val_stats = val_stats.append({'epoch': epoch, 'loss_d': val_loss_d_acc, 'loss_g': val_loss_g_acc, 'D(X)': val_d_x_acc, 'D(G(X))': val_d_g_x_acc}, ignore_index=True)

            train_stats.to_csv('train_stats.csv')
            val_stats.to_csv('val_stats.csv')

            if epoch % self.checkpoint_interval == 0 or epoch == 1:
                checkpoint = {"generator": self.generator.state_dict(),
                                "discriminator": self.discriminator.state_dict(),
                                "generator_optimizer": self.generator_optimizer.state_dict(),
                                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                                "epoch": epoch}
            
                torch.save(checkpoint, self.checkpoints_folder + "/checkpoint_epoch_" + str(epoch) + ".pth")


parser = argparse.ArgumentParser(description='Train GAN')
parser.add_argument('--epochs', default=200)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--learning_rate', default=0.0002)
parser.add_argument('--dataset_file', default='flowers.hdf5')
parser.add_argument('--print_interval', default=25)
parser.add_argument('--checkpoints_folder', default='checkpoints')
parser.add_argument('--checkpoint_interval', default=10)
parser.add_argument('--results_folder', default='results')

if __name__ == '__main__':
    args = parser.parse_args()

    trainer = GAN_Trainer(args.epochs, args.batch_size, args.learning_rate, args.dataset_file, args.checkpoint_interval, args.print_interval, args.checkpoints_folder, args.results_folder)
    trainer.train()

    print('Training completed!')



