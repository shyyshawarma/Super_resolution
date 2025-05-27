import os
import numpy as np
import matplotlib.pyplot as plt

import torch 
from torch import optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from SRCNN_model import SRCNN
from dataset import SRCNNValDataset

if __name__ == '__main__':
    Batch_size = 10
    test_data = SRCNNValDataset('dataset_img/test/low_res/69020.png')
    test_loader = DataLoader(dataset=test_data, batch_size=Batch_size)

    for batch, data in enumerate(test_loader):
        test_img = data['lr'][0]
        test_truth = data['truth'][0]
        break

    model = SRCNN()
    model.load_state_dict(torch.load('checkpoints/model.pth'))
    model.cuda()
    model.eval()

    with torch.no_grad():
        test_img_temp = test_img.unsqueeze(0)
        test_img_temp = test_img_temp.cuda()
        output = model(test_img_temp)
        output = output.squeeze(0)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.imshow(to_pil_image(test_img))
plt.title('input')
plt.subplot(1,3,2)
plt.imshow(to_pil_image(output))
plt.title('output')
plt.subplot(1,3,3)
plt.imshow(to_pil_image(test_truth))
plt.title('ground_truth')





