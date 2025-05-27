# SRCNN from scratch with custom dataset
# [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092.pdf)
![srcnn architecture](https://user-images.githubusercontent.com/82307352/168474637-8b1cece5-4ea0-4ae9-9d61-757941c240b3.png)


## Introduction 
SRCNN architecture uses 3 CNN layers for implementing super resolution, and is one of the very first papers to use deep neural network for the image super resolution task. By preprocessing an original image (i.e. the ground truth) into a low resolution image (LR), SRCNN upsamples the LR image into high resolution image (HR) and compares it with the ground truth during the training process.  


## Dataset
Custom training dataset was obtained from the below link from kaggle:
https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution
It is convenient to use since the dataset contains both LR and HR of the same image, which we don't need to run through cumbersome data preprocessing. However, since the upscale factor is unknown, it is inappropriate to precisely measure the performance of the trained architecture.

In *dataset.py*, there are two classes for both training and validation dataset. Both have almost identical. Notice that I only extracted 'Y' channels of input images for the training, which is mentioned in the paper. 

## Train
For the training, I used Adam optimizer instead of SGD. I used both MSE loss and PSNR (peak signal-to-noise ratio) for performance metric. PSNR naively represents how similar the model output is compared to the ground truth, measured in dB scale. 

## Results
With 20 Epoch for the training, below is the MSE error and PSNR with respect to each epoch, for both training and validation
![results ](https://user-images.githubusercontent.com/82307352/168474780-e16292ed-c785-41c5-9494-c07e90fcd909.jpg)

Below is the output result with one of the validation images. Notice the initial blurred image turns into a more sharp, high-resolution image.  
![그림2](https://user-images.githubusercontent.com/82307352/168475111-aa2a2024-9bd8-4882-bdbf-18d02b6bed32.jpg)
![그림4](https://user-images.githubusercontent.com/82307352/168475243-626caa54-f472-4049-aef8-c08a0fde9cf5.png)


