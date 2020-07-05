# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Cover It !

# %%
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['figure.figsize'] = (16, 9)

import argparse
import os
import sys
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from torchvision.utils import save_image

from skimage import io
from PIL import Image, ImageOps

# %% [markdown]
# **Configs**

# %%
PROJECT_DIR = os.path.abspath(os.curdir)
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCH = 100

# %% [markdown]
# # Data Preperation

# %%
# os.listdir(f'{PROJECT_DIR}/data/raw/coco-test2017')[0]


# %%
class CoverItDataset(Dataset):
    """Cover It dataset."""

    def __init__(self, root_dir, transform=None, seed=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.hide_index_dict = {idx: image for idx,
                                image in enumerate(os.listdir(self.root_dir))}

        if seed:
            np.random.seed(seed)

        hide_index_arr = np.array(list(self.hide_index_dict.keys()))
        cover_index_arr = hide_index_arr.copy()
        np.random.shuffle(cover_index_arr)
#         print(((hide_index_arr-cover_index_arr)==0).sum())
        self.cover_index_arr2 = np.where(
            (hide_index_arr - cover_index_arr) == 0, np.flip(hide_index_arr), cover_index_arr)
        print("# of collisions:",
              ((hide_index_arr - self.cover_index_arr2) == 0).sum())

        del hide_index_arr, cover_index_arr

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hide_img_path = os.path.join(self.root_dir,
                                     str(self.hide_index_dict[idx]))

        cover_img_path = os.path.join(self.root_dir,
                                      str(self.hide_index_dict[self.cover_index_arr2[idx]]))

        hide_image = io.imread(hide_img_path)
        cover_image = io.imread(cover_img_path)

        sample = [hide_image, cover_image]
        for i, image in enumerate(sample):

            if image.shape[-1] != 3:  # black white photo
                #             print(f"image:{img_path} || must have 3 color channels!")
                #             print(image.shape)
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
                sample[i] = image
    #             print(image.shape)

        if self.transform:
            sample = self.transform(sample)

        return sample


# %%
# DEBUG

# img_path = "/home/kazim/Desktop/project/cover_it/data/raw/coco-test2017/000000059177.jpg"
# # img_path = "/home/kazim/Desktop/project/cover_it/data/raw/coco-test2017/000000000001.jpg"

# image = io.imread(img_path)
# print(image.shape)
# h,w = image.shape
# image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
# # image = (image * np.ones((1,1,1), dtype=np.int8)).t
# print(image.shape)


# %%
# h,w = image.shape
# image = (image * np.ones((1,1,1), dtype=np.int8)).transpose(1,2,0)

# plt.imshow(image.squeeze() ,cmap=plt.get_cmap('gray'))

# %% [markdown]
# ## Raw Dataset

# %%

coverit_dataset = CoverItDataset(
    root_dir=os.path.join(PROJECT_DIR, 'data', 'raw', 'coco-test2017'))


# %%
print('length of dataset:', len(coverit_dataset))

fig, axis = plt.subplots(2, 4, figsize=(20, 10))

for i in range(4):
    cover, hide = coverit_dataset[i]
    print(i, cover.shape, hide.shape)
    axis[0, i].imshow(cover)
    axis[1, i].imshow(hide)

    for ax in axis[:, i]:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

# %% [markdown]
# ## Transformations

# %%


class ToPILImage(object):
    """Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL.Image while preserving value range.
    """

    def __call__(self, sample):
        res = []

        for pic in sample:
            npimg = pic
            mode = None
            if isinstance(pic, torch.FloatTensor):
                pic = pic.mul(255).byte()
            if torch.is_tensor(pic):
                npimg = np.transpose(pic.numpy(), (1, 2, 0))
            assert isinstance(
                npimg, np.ndarray), 'pic should be Tensor or ndarray'
            if npimg.shape[2] == 1:
                npimg = npimg[:, :, 0]

                if npimg.dtype == np.uint8:
                    mode = 'L'
                if npimg.dtype == np.int16:
                    mode = 'I;16'
                if npimg.dtype == np.int32:
                    mode = 'I'
                elif npimg.dtype == np.float32:
                    mode = 'F'
            else:
                if npimg.dtype == np.uint8:
                    mode = 'RGB'
            assert mode is not None, '{} is not supported'.format(npimg.dtype)
            res.append(Image.fromarray(npimg, mode=mode))
        return res


# %%
class PadCenterCrop(object):
    """
    Get the center/random crop of an image with given size. If image is too small pad it accordingly.
    """

    def __init__(self, out_size, is_random_crop=False):
        if isinstance(out_size, int):
            self.out_size = (out_size, out_size)
        elif isinstance(out_size, tuple):
            self.out_size = out_size
        else:
            raise Exception('Expect int or tuple only!')

        self.random_crop = is_random_crop

    def __call__(self, sample):
        res = []

        for img in sample:
            #             breakpoint()
            w, h = img.size
            th, tw = self.out_size

            diff = min(img.size) - max(self.out_size)

            if diff < 0:
                img = ImageOps.expand(img, border=-diff, fill=0)
                w, h = img.size

            if w == tw and h == th:
                res.append(img)
                break

            if self.random_crop:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
            else:
                x1 = (w - tw) / 2
                y1 = (h - th) / 2

            res.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return res


# %%
class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, is_gray=False, do_normalize=True):
        self.is_gray = is_gray

    def __call__(self, sample):
        res = []

        for pic in sample:
            if isinstance(pic, np.ndarray):
                # handle numpy array
                img = torch.from_numpy(pic.transpose((2, 0, 1)))
                # backard compability
                res.append(img.float().div(255))
                break
            # handle PIL Image
            if pic.mode == 'I':
                img = torch.from_numpy(np.array(pic, np.int32, copy=False))
            elif pic.mode == 'I;16':
                img = torch.from_numpy(np.array(pic, np.int16, copy=False))
            else:
                img = torch.ByteTensor(
                    torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            elif pic.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)

            if self.is_gray:
                coef = torch.tensor([0.299, 0.587, 0.114])
                img = (img * coef.view(1, 1, -1)).sum(axis=-1, keepdims=True)

            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img, torch.ByteTensor):
                res.append(img.float().div(255))
            else:
                res.append(img.div(255))

        return res


# %%
# toPIL = ToPILImage()
# pcc = PadCenterCrop(out_size=IMAGE_SIZE)
# tten = ToTensor()


# %%
# toPIL(sample)


# %%
# tten(pcc(toPIL(sample)))[1].shape


# %%
# fig,axis = plt.subplots(4,4, figsize=(20,20))
# axis = axis.reshape(-1)
# for i,ax in enumerate(axis):
#     sample = crop(toPIL(coverit_dataset[i]))
#     print(i, sample.size)
#     ax.imshow(sample)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])


# %%
# from inspect import getsource
# print(getsource(transforms.ToPILImage))

# %% [markdown]
# ## Transformed Dataset

# %%
coverit_dataset_transformed = CoverItDataset(root_dir=os.path.join(PROJECT_DIR, 'data', 'raw', 'coco-test2017'),
                                             transform=transforms.Compose([
                                                 ToPILImage(),
                                                 PadCenterCrop(
                                                     out_size=IMAGE_SIZE),
                                                 ToTensor(is_gray=False)
                                             ]))


# %%
print('length of dataset:', len(coverit_dataset_transformed))


# %%
def show_image(image, fig_size=(5, 5), log_dims=False, is_return=False):
    """
    Args:
        image: tensor
    """
    if len(image.shape) == 2:
        image = image[None, :, :]

    is_colored = not (image.shape[0] == 1)
    if is_colored:
        image_np = image.numpy().transpose(1, 2, 0)
    else:
        image_np = image.numpy()

    if log_dims:
        print(i, image_np.shape)

    if is_colored:
        plt.imshow(image_np)
    else:
        plt.imshow(image_np[0], cmap=plt.get_cmap('gray'))

    ax, fig = plt.gca(), plt.gcf()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if fig_size:
        fig.set_size_inches(fig_size)

    if is_return:
        return fig


# %%
hide, cover = coverit_dataset_transformed[0]
print(hide.shape, cover.shape)


# %%
show_image(hide, is_return=False)
plt.show()
show_image(cover, is_return=False)
plt.show()

# %% [markdown]
# ## Data Loader
# %% [markdown]
# Dataloader gives batch of hide and cover images in a list for each iteration

# %%
dataloader = DataLoader(coverit_dataset_transformed,
                        batch_size=BATCH_SIZE, num_workers=4, shuffle=False)


# %%
hide, cover = next(iter(dataloader))


# %%
print(hide.shape, cover.shape)


# %%
def show_batch(batch, figsize=(10, 10), log_dims=False, is_return=False, title=None):
    """
    Args:
        batch: tensor
    """
    subplot_size = int(batch.shape[0]**(1 / 2))
    is_colored = not (batch.shape[1] == 1)

    batch_np = batch.numpy().transpose(0, 2, 3, 1)

    fig, axis = plt.subplots(subplot_size, subplot_size, figsize=figsize)
    axis = axis.reshape(-1)
    for i, ax in enumerate(axis):
        sample = batch_np[i]
        if log_dims:
            print(i, sample.shape)

        if is_colored:
            ax.imshow(sample)
        else:
            ax.imshow(sample.squeeze(), cmap=plt.get_cmap('gray'))

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if is_return:
        return fig


# %%

show_batch(hide, title='hide')
plt.show()

show_batch(cover, title='cover')
plt.show()

# %% [markdown]
# # Model

# %%


def calc_cnn(
        input_size=256,
        padding=1,
        kernel_size=3,
        stride=1):
    output_size = (input_size + 2 * padding - kernel_size) // stride + 1
    return output_size


calc_cnn()

# %% [markdown]
# ### HidingNet

# %%


class HidingUNet(nn.Module):
    """
    Takes list of 2 input images -> convert them to 1 hidden image
    input   : [(col) x 256 x 256, (col) x 256 x 256] #hide and cover
    output  : (col) x 256 x 256 #hidden
    """

    def __init__(self):
        super(HidingUNet, self).__init__()

        self.conv_1 = nn.Conv2d(6, 16, 3, 1, 1)
#         self.bnorm_1 = nn.BatchNorm2d(12)
        self.drop_1 = nn.Dropout2d(p=0.2)
        self.conv_1_2 = nn.Conv2d(16, 16, 3, 1, 1)

        self.maxp_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(16, 32, 3, 1, 1)
#         self.bnorm_2 = nn.BatchNorm2d(12)
        self.drop_2 = nn.Dropout2d(p=0.2)
        self.conv_2_2 = nn.Conv2d(32, 32, 3, 1, 1)

        self.maxp_2 = nn.MaxPool2d(2, 2)
        self.conv_3 = nn.Conv2d(32, 64, 3, 1, 1)
#         self.bnorm_3 = nn.BatchNorm2d(12)
        self.drop_3 = nn.Dropout2d(p=0.2)
        self.conv_3_2 = nn.Conv2d(64, 64, 3, 1, 1)
############
        self.maxp_3 = nn.MaxPool2d(2, 2)
        self.conv_4 = nn.Conv2d(64, 128, 3, 1, 1)
#         self.bnorm_4 = nn.BatchNorm2d(12)
        self.drop_4 = nn.Dropout2d(p=0.2)
        self.conv_4_2 = nn.Conv2d(128, 64, 3, 1, 1)
#############
        self.upsm_1 = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.bnorm_3 = nn.BatchNorm2d(12)
        self.conv_5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.drop_5 = nn.Dropout2d(p=0.2)
        self.conv_5_2 = nn.Conv2d(64, 32, 3, 1, 1)

        self.upsm_2 = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.bnorm_4 = nn.BatchNorm2d(12)
        self.conv_6 = nn.Conv2d(32, 32, 3, 1, 1)
        self.drop_6 = nn.Dropout2d(p=0.2)
        self.conv_6_2 = nn.Conv2d(32, 16, 3, 1, 1)

        self.upsm_3 = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.bnorm_5 = nn.BatchNorm2d(12)
        self.conv_7 = nn.Conv2d(16, 16, 3, 1, 1)
        self.drop_7 = nn.Dropout2d(p=0.2)
        self.conv_7_2 = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, input, bool_drop=False):
        """
        input(list): [hide,cover]
        return(tensor): hidden
        """
        concated = torch.cat(
            input, dim=1)  # [6, 256, 256] concat along color axis

        # down
        d1 = F.leaky_relu(self.conv_1(concated))  # [16, 256, 256]
        if bool_drop:
            d1 = self.drop_1(d1)
        d1 = F.leaky_relu(self.conv_1_2(d1))  # [16, 256, 256]

        d2 = self.maxp_1(d1)    # [16, 128, 128]
        d2 = F.leaky_relu(self.conv_2(d2))    # [32, 128, 128]
        if bool_drop:
            d2 = self.drop_2(d2)
        d2 = F.leaky_relu(self.conv_2_2(d2))  # [32, 128, 128]

        d3 = self.maxp_2(d2)    # [32, 64, 64]
        d3 = F.leaky_relu(self.conv_3(d3))    # [64, 64, 64]
        if bool_drop:
            d3 = self.drop_3(d3)
        d3 = F.leaky_relu(self.conv_3_2(d3))  # [64, 64, 64]

        # bottom
        b = self.maxp_3(d3)     # [64, 32, 32]
        b = F.leaky_relu(self.conv_4(b))      # [128, 32, 32]
        if bool_drop:
            b = self.drop_4(b)
        b = F.leaky_relu(self.conv_4_2(b))    # [64, 32, 32]

        # up
        u3 = self.upsm_1(b)     # [64, 64, 64]
        u3 = d3 + u3              # [64, 64, 64]
        u3 = F.leaky_relu(self.conv_5(u3))    # [64, 64, 64]
        if bool_drop:
            u3 = self.drop_5(u3)
        u3 = F.leaky_relu(self.conv_5_2(u3))  # [32, 64, 64]

        u2 = self.upsm_2(u3)    # [32, 128, 128]
        u2 = d2 + u2              # [32, 128, 128]
        u2 = F.leaky_relu(self.conv_6(u2))    # [32, 128, 128]
        if bool_drop:
            u2 = self.drop_6(u2)
        u2 = F.leaky_relu(self.conv_6_2(u2))  # [16, 128, 128]

        u1 = self.upsm_3(u2)    # [16, 256, 256]
        u1 = d1 + u1              # [16, 256, 256]
        u1 = F.leaky_relu(self.conv_7(u1))    # [16, 256, 256]
        if bool_drop:
            u1 = self.drop_7(u1)
        out = F.leaky_relu(self.conv_7_2(u1))  # [3, 256, 256]

        return out


# %%
# HidingUNet()(batch).shape


# %%
hidden = HidingUNet()([hide, cover], bool_drop=True).detach()
show_batch(hidden, figsize=(10, 10), title='hidden')


# %%
hidden

# %% [markdown]
# **Look at transformerdecoder!!!**
# %% [markdown]
# ### RevealNet

# %%


class RevalingNet(nn.Module):
    """
    Takes list of 2 input images -> output revealed hide image
    input   : [(col) x 256 x 256, (col) x 256 x 256] #hidden and cover
    output  : (col) x 256 x 256 #revealed hide
    """

    def __init__(self):
        super(RevalingNet, self).__init__()

        self.conv1 = nn.Conv2d(6, 3, 3, 1, 1)
        self.bnorm_1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bnorm_2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, input):

        # [6, 256, 256] concat along color axis
        concated = torch.cat(input, dim=1)

        x = F.leaky_relu(self.conv1(concated))  # [3, 256, 256]
        x = self.bnorm_1(x)
        x = F.leaky_relu(self.conv2(x))  # [3, 256, 256]
        x = self.bnorm_2(x)
        x = F.leaky_relu(self.conv3(x))  # [3, 256, 256]

        return x  # return hide


# %%
rev_hide = RevalingNet()([hidden, cover]).detach()
show_batch(hidden, figsize=(10, 10), title='revealed')
rev_hide.shape

# %% [markdown]
# ### CoverIt

# %%


class CoverIt(nn.Module):
    def __init__(self, hidenet, revealnet):
        super(CoverIt, self).__init__()
        self.hidenet = hidenet
        self.revealnet = revealnet

    def forward(self, hide_cover):
        hide, cover = hide_cover
        hidden = self.hidenet(hide_cover)
        rev_hide = self.revealnet([hidden, cover])
        return hidden, rev_hide


# %%
hidenet = HidingUNet()  # .cuda()
revealnet = RevalingNet()  # .cuda()
coverit = CoverIt(hidenet, revealnet).cuda()

# %% [markdown]
# ## Results Before Traning

# %%
hide = hide.cuda()
cover = cover.cuda()
batch = [hide, cover]

hidden, rev_hide = coverit(batch)
hidden = hidden.detach().cpu()
rev_hide = rev_hide.detach().cpu()

show_batch(hidden, figsize=(10, 10), title='hidden')
plt.show()
show_batch(rev_hide, figsize=(10, 10), title='revealed')
plt.show()


# %%
# removing early experiments
# !rm -rf logs


# %%
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# Sets up a timestamped log directory.
logdir = os.path.join("logs", "train_data", "hidingnet_v3_",
                      datetime.now().strftime("%Y%m%d-%H%M"))
print(logdir)
# Creates a file writer for the log directory.
writer = SummaryWriter(logdir)


# %%
def show_input_output(hide, cover, hidden, rev_hide, figsize=(4, 16), log_dims=False, is_return=False):
    """
    16*4

    Args:
        hide:     tensor_image
        cover:    tensor_image
        hidden:   tensor_image
        rev_hide: tensor_image
    """
    is_colored = not (hide.shape[1] == 1)

    hide_np = hide.numpy().transpose(0, 2, 3, 1)
    cover_np = cover.numpy().transpose(0, 2, 3, 1)
    hidden_np = hidden.numpy().transpose(0, 2, 3, 1)
    rev_hide_np = rev_hide.numpy().transpose(0, 2, 3, 1)

    fig, axis = plt.subplots(4, hide.shape[0], figsize=figsize)

    for col in range(hide.shape[0]):
        for row, np_img in enumerate([hide_np, cover_np, hidden_np, rev_hide_np]):
            ax = axis[row, col]

            sample = np_img[col]

            if log_dims:
                print(row, col, sample.shape)

            if is_colored:
                ax.imshow(sample)
            else:
                ax.imshow(sample.squeeze(), cmap=plt.get_cmap('gray'))

            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    if is_return:
        return fig


# %%
show_input_output(hide.cpu(), cover.cpu(), hidden.detach().cpu(
), rev_hide.detach().cpu(), figsize=(40, 10), is_return=False)

# %% [markdown]
# #### start tensorboard

# %%
# !tensorboard --logdir=./logs

# %% [markdown]
# ## Load Model

# %%
# NOTE below row may not be working in windows os
model_id = logdir.split('/')[-1]
print(model_id)

model_dir = os.path.join(PROJECT_DIR, 'models', model_id)
print(model_dir)


# %%
# Create Dirs if doesn't exist


os.makedirs(os.path.join(PROJECT_DIR, 'models',
                         model_id, 'encoder'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, 'models',
                         model_id, 'decoder'), exist_ok=True)


# %%
# load_specific = True

# NOTE below function works for unix based file system, use os lib for windows
def load_model(encoder, decoder, model_dir, load_specific=None):

    start_epoch = 0
    if load_specific:
        encoder.load_state_dict(torch.load(load_specific[0]))
        decoder.load_state_dict(torch.load(load_specific[1]))
        print('Loaded model parameters!')
    else:
        try:
            epoch = max([int(i.split('.')[0].split('_')[1])
                         for i in os.listdir(f'{model_dir}/encoder')])
            print(f"Found early model with parameters of epoch {epoch}")
            encoder_path = f"{model_dir}/encoder/encoder_{epoch}.pth"
            decoder_path = f"{model_dir}/decoder/decoder_{epoch}.pth"
            print("encoder_path:", encoder_path)

            encoder.load_state_dict(torch.load(encoder_path))
            decoder.load_state_dict(torch.load(decoder_path))
            print('Loaded model parameters!')

            start_epoch = int(start_epoch) + 1

        except ValueError:  # No saved models
            print("Could't find early model parameters!")

    print(f"Starting epoch: {start_epoch}")
    return start_epoch


# %%
# for param in hidenet.parameters():
#     print(param[0])
#     break


# %%
start_epoch = load_model(hidenet, revealnet, model_dir=model_dir,
                         load_specific=None)
start_epoch


# %%
# for param in hidenet.parameters():
#     print(param[0])
#     break

# %% [markdown]
# # Training

# %%
coverit = CoverIt(hidenet, revealnet).cuda()


# %%
mse = nn.MSELoss()
optimizer = optim.Adam(coverit.parameters(), lr=0.001)
alpha = 1 / 2


# %%
running_total_loss = 0.0
running_hiding_loss = 0.0
running_revealing_loss = 0.0

# EPOCH = 30
# loop over the dataset multiple times
for epoch in range(start_epoch, start_epoch + EPOCH):

    for step, data in enumerate(dataloader, 0):

        #         data = data.type(torch.cuda.FloatTensor)

        # get the inputs
        #         inputs = data.cuda()
        hide, cover = data
        hide = hide.cuda()
        cover = cover.cuda()
        batch = [hide, cover]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        hidden, rev_hide = coverit(batch)
#         outputs = autoencoder(inputs)
        hiding_loss = mse(cover, hidden)
        revealing_loss = mse(hide, rev_hide)

        total_loss = alpha * hiding_loss + (1 - alpha) * revealing_loss
        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item()
        running_hiding_loss += hiding_loss.item()
        running_revealing_loss += revealing_loss.item()

        if step % 100 == 0:    # every 100 mini-batches...

            print("Epoch: {} | Step: {}  =========  Loss: {:.3}".format(
                epoch, step, total_loss.item()))

            # ...log the running total_loss
            writer.add_scalar('training total_loss',
                              running_total_loss / 100,
                              global_step=epoch * len(dataloader) + step)
            writer.add_scalar('training hiding_loss',
                              running_total_loss / 100,
                              global_step=epoch * len(dataloader) + step)
            writer.add_scalar('training revealing_loss',
                              running_total_loss / 100,
                              global_step=epoch * len(dataloader) + step)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch

            if step % 1000 == 0:  # every 1000 mini-batches...

                hidden = hidden.detach().cpu()
                rev_hide = rev_hide.detach().cpu()

                # create grids of images
                figure = show_input_output(hide.cpu(), cover.cpu(
                ), hidden, rev_hide, figsize=(40, 10), is_return=True)

                # write to tensorboard
                writer.add_figure('hide cover hidden revealed',
                                  figure,
                                  global_step=epoch * len(dataloader) + step)

                if epoch % 10 == 0:  # every 10 epoch...
                    torch.save(hidenet.state_dict(),
                               f"./models/{model_id}/encoder/encoder_{epoch}.pth")
                    torch.save(revealnet.state_dict(),
                               f"./models/{model_id}/decoder/decoder_{epoch}.pth")
                    print('Saved model params to the disk!')

            running_total_loss = 0.0
            running_hiding_loss = 0.0
            running_revealing_loss = 0.0
print('Finished Training')
