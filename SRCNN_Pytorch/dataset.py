import cv2
import torch
from torch.utils.data import Dataset
import os
FIXED_SIZE = (256, 256)  # width, height

class SRCNNTrainDataset(Dataset):
    def __init__(self, image_dir):
        self.truth_path = os.path.join(image_dir, 'train', 'high_res')
        self.lr_path = os.path.join(image_dir, 'train', 'low_res')
        self.truth_images_names = [os.path.join(self.truth_path, f) for f in os.listdir(self.truth_path)]
        self.lr_images_names = [os.path.join(self.lr_path, f) for f in os.listdir(self.lr_path)]

    def __getitem__(self, index):
        truth_image = cv2.imread(self.truth_images_names[index])
        lr_image = cv2.imread(self.lr_images_names[index])

        # Resize both images
        truth_image = cv2.resize(truth_image, FIXED_SIZE, interpolation=cv2.INTER_CUBIC)
        lr_image = cv2.resize(lr_image, FIXED_SIZE, interpolation=cv2.INTER_CUBIC)

        ycbcr_truth = cv2.cvtColor(truth_image, cv2.COLOR_BGR2YCR_CB)
        y_truth, _, _ = cv2.split(ycbcr_truth)

        ycbcr_lr = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCR_CB)
        y_lr, _, _ = cv2.split(ycbcr_lr)

        y_truth = torch.from_numpy(y_truth.astype('float32')) / 255.0
        y_lr = torch.from_numpy(y_lr.astype('float32')) / 255.0

        return {
            "lr": y_lr.unsqueeze(0),
            "truth": y_truth.unsqueeze(0)
        }

    def __len__(self):
        return len(self.truth_images_names)

# Do the same for SRCNNValDataset — just update the __init__ path to 'val'
class SRCNNValDataset(Dataset):
    def __init__(self, image_dir):
        self.truth_path = os.path.join(image_dir, 'val', 'high_res')
        self.lr_path = os.path.join(image_dir, 'val', 'low_res')
        self.truth_images_names = [os.path.join(self.truth_path, f) for f in os.listdir(self.truth_path)]
        self.lr_images_names = [os.path.join(self.lr_path, f) for f in os.listdir(self.lr_path)]

    def __getitem__(self, index):
        truth_image = cv2.imread(self.truth_images_names[index])
        lr_image = cv2.imread(self.lr_images_names[index])

        # Resize both images
        truth_image = cv2.resize(truth_image, FIXED_SIZE, interpolation=cv2.INTER_CUBIC)
        lr_image = cv2.resize(lr_image, FIXED_SIZE, interpolation=cv2.INTER_CUBIC)

        ycbcr_truth = cv2.cvtColor(truth_image, cv2.COLOR_BGR2YCR_CB)
        y_truth, _, _ = cv2.split(ycbcr_truth)

        ycbcr_lr = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCR_CB)
        y_lr, _, _ = cv2.split(ycbcr_lr)

        y_truth = torch.from_numpy(y_truth.astype('float32')) / 255.0
        y_lr = torch.from_numpy(y_lr.astype('float32')) / 255.0

        return {
            "lr": y_lr.unsqueeze(0),
            "truth": y_truth.unsqueeze(0)
        }

    def __len__(self):
        return len(self.truth_images_names)

from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
class SRCNNTestDataset(Dataset):
    def __init__(self, root_dir):
        self.lr_dir = os.path.join(root_dir, 'low_res')
        self.hr_dir = os.path.join(root_dir, 'high_res')
        self.filenames = sorted(os.listdir(self.lr_dir))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.filenames[idx])
        hr_path = os.path.join(self.hr_dir, self.filenames[idx])

        lr_img = Image.open(lr_path).convert('L')  # <-- grayscale
        hr_img = Image.open(hr_path).convert('L')  # <-- grayscale

        return {
            'lr': self.to_tensor(lr_img),
            'hr': self.to_tensor(hr_img),
            'filename': self.filenames[idx]
        }
