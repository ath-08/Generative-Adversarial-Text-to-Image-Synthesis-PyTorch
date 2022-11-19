import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_images(results_folder, images, generated, train, epoch, batch_idx):
    images = images.cpu()
    im = make_grid(images, nrow=8, pad_value=1)  # the default nrow is 8
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.title("Generated Images" if generated else "Real Images")

    filename = results_folder
    filename += ('/train/' if train else 'val/')
    filename += ('/generated/' if generated else 'real/')
    filename += str(epoch) + '_' + str(batch_idx) + '.jpg'

    # We need to transpose the images from CWH to WHC
    im = np.transpose(im.numpy(), (1, 2, 0)).clip(0, 1)
    plt.imshow(im)
    plt.savefig(filename)
