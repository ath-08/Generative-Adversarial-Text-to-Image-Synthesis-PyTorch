import torch
from torchvision import transforms
import h5py
import numpy as np
from PIL import Image
import io

class ImageGenDataset(torch.utils.data.Dataset):
    def __init__(self, filename, split, transform=transforms.ToTensor()):
        super(ImageGenDataset, self).__init__()
        self.filename = filename
        self.split = split
        self.transform = transform
        
        self.f = h5py.File(self.filename, mode='r')
        self.dataset = self.f[self.split]
        self.keys = list(self.dataset.keys())

    def __len__(self):
        return len(self.keys)
  
    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.dataset[key]

        real_img = self.transform(Image.open(io.BytesIO(bytes(np.array(sample['img'])))))
        right_txt_embeds = torch.Tensor(sample['embeddings'])
        wrong_txt_embeds = torch.Tensor(self.find_wrong_txt_embeds(str(np.array(sample['class']).astype(str))))

        return {"real_img": real_img,
                "right_txt_embeds": right_txt_embeds,
                "wrong_txt_embeds": wrong_txt_embeds,
                "right_txt": str(np.array(sample['txt']).astype(str))}
    
    def find_wrong_txt_embeds(self, curr_class):
        r = torch.randint(low=0, high=len(self.keys), size=(1,)).item()
        wrong_class = str(np.array(self.dataset[self.keys[r]]['class']).astype(str))

        if curr_class == wrong_class:
            return self.find_wrong_txt_embeds(curr_class)

        return self.dataset[self.keys[r]]['embeddings']