import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
  print(torch.cuda.current_device())
  print(torch.cuda.device(0))
  print(torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))
else:
  print("No NVIDIA driver found. Using CPU")

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.RandomHorizontalFlip() if split == 'train' else lambda x: x,
        ])
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(self.transforms(img))
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)
    
def lab_to_rgb(L, ab):
        L, ab = (L + 1) * 50, ab * 110
        Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        return np.stack([lab2rgb(img) for img in Lab], axis=0)

def visualize_images(L, fake_color, real_color):
        fake_imgs, real_imgs = lab_to_rgb(L, fake_color), lab_to_rgb(L, real_color)
    
    # Determine the number of images to display
        num_images = min(len(fake_imgs), len(real_imgs), L.shape[0])
        
        fig, axes = plt.subplots(3, num_images, figsize=(15, 8))
        
        if num_images == 1:
            axes = np.expand_dims(axes, axis=1)
        
        for i in range(num_images):
            grayscale_img = L[i][0].cpu()
            
            axes[0, i].imshow(grayscale_img, cmap='gray')
            axes[1, i].imshow(fake_imgs[i])
            axes[2, i].imshow(real_imgs[i])
            
            for ax in axes[:, i]:
                ax.axis("off")
        plt.show()

def return_as_color_image(L, fake_color):
    fake_imgs = lab_to_rgb(L, fake_color)
    
    # Display the first image in the batch
    plt.imshow(fake_imgs[0])
    plt.axis('off')  # Hide the axis
    plt.show()
    
    # Return the colorized image
    return fake_imgs[0]
    

# # Load the entire model
# model = torch.load('trained_model.pth')
# model.eval()  # Set the model to evaluation mode

# Function to test the loaded model
def colorize_and_test(model, img_path):
    data = ColorizationDataset([img_path], split='val')[0]
    L = data['L'].to(device).unsqueeze(0)
    fake_color = model(L).detach()  # Run the input through the loaded model
    #visualize_images(L, fake_color, data['ab'].unsqueeze(0))
    img_color = return_as_color_image(L, fake_color)
    print(img_color.shape)
    return img_color

# Test the model with an image path
# colored_image  = colorize_and_test(model, '436d1409c0fc5abe0c5ff2b04cbd828b.jpg')