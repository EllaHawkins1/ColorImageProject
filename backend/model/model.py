import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#import cv2
import time
import torch.nn as nn
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
    def __init__(self, images, split='train'):
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.RandomHorizontalFlip() if split == 'train' else lambda x: x,
        ])
        self.images = images  # Changed from paths to images

    def __getitem__(self, idx):
        img = self.images[idx] # Get the PIL image directly
        img = self.transforms(img)  # Apply transformations
        img_lab = transforms.ToTensor()(img)  # Convert to tensor
        # Uncomment and modify the following lines as needed for your model
        # img_lab = rgb2lab(img).astype("float32")
        # L = img_lab[[0], ...] / 50. - 1.
        # ab = img_lab[[1, 2], ...] / 110.
        return {'L': img_lab, 'ab': "ab"}  # Return the tensor and placeholder for 'ab'

    def __len__(self):
        return len(self.paths)
    

def lab_to_rgb(L, ab):
    # Ensure L and ab have the correct shapes
    assert L.shape[1] == 1, f"Expected L to have 1 channel, got {L.shape[1]}"
    assert ab.shape[1] == 2, f"Expected ab to have 2 channels, got {ab.shape[1]}"
    
    # Scale L and ab to the expected range
    L, ab = (L + 1) * 50, ab * 110
    
    # Concatenate L and ab to form the Lab image
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    
    # Convert each image in the batch from Lab to RGB
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
    #fake_color = lab_to_rgb(L, fake_color)
    # Assuming fake_color is a tensor with shape [1, 3, 256, 256]
    fake_imgs = fake_color.squeeze(0)  # Remove the batch dimension, now shape is [3, 256, 256]

    # Convert to NumPy and transpose dimensions
    img_np = fake_imgs.permute(1, 2, 0).cpu().numpy()  # Now shape is [256, 256, 3]

    # If the image was normalized, denormalize it (example for normalization with mean=0.5, std=0.5)
    # img_np = (img_np * 0.5) + 0.5  # Adjust this based on your normalization

    # # Clip values to [0, 1] range if necessary
    # img_np = img_np.clip(0, 1)

    # Display the image
    # plt.imshow(img_np)
    # plt.axis('off')  # Hide the axis
    # plt.show()

    # Return the colorized image
    return img_np
    
    # Return the colorized image
    return fake_imgs[0]
def rgb_to_gray(img):
    return img.mean(dim=1, keepdim=True)


# Define the colorization model
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x
    

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    # Convert from Tensor image and display
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    if len(img.shape) == 2:  # grayscale image
        plt.imshow(npimg, cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualize_all_three(original_images, grayscale_images, colorized_images, n=5):
    """
    Display grayscale, colorized, and original images side by side.
    n: number of images to display from the batch
    """
    fig = plt.figure(figsize=(3*n, 4))
    for i in range(n):
        # Display original image
        ax = plt.subplot(1, 3*n, 3*i + 1)
        imshow(original_images[i])
        ax.set_title("Original")
        ax.axis("off")

        # Display original grayscale image
        ax = plt.subplot(1, 3*n, 3*i + 2)
        imshow(grayscale_images[i])
        ax.set_title("Grayscale")
        ax.axis("off")

        # Display colorized image
        ax = plt.subplot(1, 3*n, 3*i + 3)
        imshow(colorized_images[i])
        ax.set_title("Colorized")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def torch_rgb_to_hsv(rgb):
    """
    Convert an RGB image tensor to HSV.

    Parameters:
    - rgb: tensor of shape (batch_size, 3, height, width) in RGB format in the range [0, 1].

    Returns:
    - hsv: tensor of same shape in HSV format in the range [0, 1].
    """
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    max_val, _ = torch.max(rgb, dim=1)
    min_val, _ = torch.min(rgb, dim=1)
    diff = max_val - min_val

    # Compute H
    h = torch.zeros_like(r)
    mask = (max_val == r) & (g >= b)
    h[mask] = (g[mask] - b[mask]) / diff[mask]
    mask = (max_val == r) & (g < b)
    h[mask] = (g[mask] - b[mask]) / diff[mask] + 6.0
    mask = max_val == g
    h[mask] = (b[mask] - r[mask]) / diff[mask] + 2.0
    mask = max_val == b
    h[mask] = (r[mask] - g[mask]) / diff[mask] + 4.0
    h = h / 6.0
    h[diff == 0.0] = 0.0

    # Compute S
    s = torch.zeros_like(r)
    s[diff != 0.0] = diff[diff != 0.0] / max_val[diff != 0.0]

    # V is just max_val
    v = max_val

    return torch.stack([h, s, v], dim=1)


def torch_hsv_to_rgb(hsv):
    """
    Convert an HSV image tensor to RGB.

    Parameters:
    - hsv: tensor of shape (batch_size, 3, height, width) in HSV format in the range [0, 1].

    Returns:
    - rgb: tensor of same shape in RGB format in the range [0, 1].
    """
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    r[i_mod == 0.0] = v[i_mod == 0.0]
    g[i_mod == 0.0] = t[i_mod == 0.0]
    b[i_mod == 0.0] = p[i_mod == 0.0]

    r[i_mod == 1.0] = q[i_mod == 1.0]
    g[i_mod == 1.0] = v[i_mod == 1.0]
    b[i_mod == 1.0] = p[i_mod == 1.0]

    r[i_mod == 2.0] = p[i_mod == 2.0]
    g[i_mod == 2.0] = v[i_mod == 2.0]
    b[i_mod == 2.0] = t[i_mod == 2.0]

    r[i_mod == 3.0] = p[i_mod == 3.0]
    g[i_mod == 3.0] = q[i_mod == 3.0]
    b[i_mod == 3.0] = v[i_mod == 3.0]

    r[i_mod == 4.0] = t[i_mod == 4.0]
    g[i_mod == 4.0] = p[i_mod == 4.0]
    b[i_mod == 4.0] = v[i_mod == 4.0]

    r[i_mod == 5.0] = v[i_mod == 5.0]
    g[i_mod == 5.0] = p[i_mod == 5.0]
    b[i_mod == 5.0] = q[i_mod == 5.0]

    return torch.stack([r, g, b], dim=1)

def exaggerate_colors(images, saturation_factor=1.5, value_factor=1.2):
    """
    Exaggerate the colors of RGB images.

    Parameters:
    - images: tensor of shape (batch_size, 3, height, width) in RGB format.
    - saturation_factor: factor by which to increase the saturation. Default is 1.5.
    - value_factor: factor by which to increase the value/brightness. Default is 1.2.

    Returns:
    - color_exaggerated_images: tensor of same shape as input, with exaggerated colors.
    """
    # Convert images to the range [0, 1]
    images = (images + 1) / 2.0

    # Convert RGB images to HSV
    images_hsv = torch_rgb_to_hsv(images)

    # Increase the saturation and value components
    images_hsv[:, 1, :, :] = torch.clamp(images_hsv[:, 1, :, :] * saturation_factor, 0, 1)
    images_hsv[:, 2, :, :] = torch.clamp(images_hsv[:, 2, :, :] * value_factor, 0, 1)

    # Convert the modified HSV images back to RGB
    color_exaggerated_images = torch_hsv_to_rgb(images_hsv)

    # Convert images back to the range [-1, 1]
    color_exaggerated_images = color_exaggerated_images * 2.0 - 1.0

    return color_exaggerated_images


import torch

# Load the entire mode

# Function to test the loaded model
# def colorize_and_test(model, img_path):
#     data = ColorizationDataset([img_path], split='val')[0]
#     L = data['L'].to(device).unsqueeze(0)
#     print(L.shape)
#     A = rgb_to_gray(L)
#     fake_color = model(A).detach()  # Run the input through the loaded model
#     original_images_cpu = fake_color.cpu()
#     A = A.cpu().squeeze(1)

#         #colorized_images_cpu=scale_predicted_colors(colorized_images_cpu)
#     colorized_images_cpu=exaggerate_colors(original_images_cpu)
#     #visualize_images(L, fake_color, data['ab'].unsqueeze(0))
#     img_color = return_as_color_image(A, colorized_images_cpu)
#     print(img_color.shape)
#     return img_color

def colorize_and_test(model, img):
    data = ColorizationDataset([img], split='val')[0]
    try :

        L = data['L'].to(device).unsqueeze(0)
        #print("ger ,", L)
        
        A = rgb_to_gray(L)
        print("converted to gray")
        fake_color = model(A).detach()
        print("colorized")
        original_images_cpu = fake_color.cpu()
        
        A = A.cpu().squeeze(1)
    except  Exception as e:
        print(e)
    colorized_images_cpu = exaggerate_colors(original_images_cpu)
    img_color = return_as_color_image(A, colorized_images_cpu)
    #print("returning image",img_color)

    return img_color
