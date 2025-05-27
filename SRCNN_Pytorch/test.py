import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from SRCNN_model import SRCNN
from dataset import SRCNNTestDataset
from PIL import Image
import matplotlib.pyplot as plt

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1 / mse)
def ssim(img1, img2):
    img1_np = img1.cpu().numpy().squeeze()  # remove channel dimension, shape (H, W)
    img2_np = img2.cpu().numpy().squeeze()
    return compare_ssim(img1_np, img2_np, win_size=3, data_range=1.0)

# Paths
test_dir = 'dataset_img/test'
model_path = 'checkpoints/model.pth'

# Load model
model = SRCNN().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

# Load test data
test_dataset = SRCNNTestDataset(test_dir)
test_loader = DataLoader(test_dataset, batch_size=1)

if len(test_loader) == 0:
    print("⚠️ No test images found. Check directory paths and formats.")
    exit()

avg_psnr = 0
avg_ssim = 0

for data in test_loader:
    lr = data['lr'].cuda()
    hr = data['hr'].cuda()
    filename = data['filename'][0]

    with torch.no_grad():
        output = model(lr)

    # Ensure shape match
    _, _, h, w = hr.shape
    output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)

    batch_psnr = psnr(hr, output).item()
    batch_ssim = ssim(hr[0], output[0])

    avg_psnr += batch_psnr
    avg_ssim += batch_ssim

    print(f"{filename} | PSNR: {batch_psnr:.2f} dB | SSIM: {batch_ssim:.4f}")

avg_psnr /= len(test_loader)
avg_ssim /= len(test_loader)

print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
