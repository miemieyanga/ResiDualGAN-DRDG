import imp


import os
import sys
import PIL
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from .utils import *
import albumentations as A
from ..datasets.seg_dataset import SegDataset
import segmentation_models_pytorch as smp
from ..models.residualgan import *
# from ..datasets.seg_dataset import TransferDataset
from torchvision.utils import save_image


def load_models(models, model_type, dsa, dsb):
    res = []
    for model in models:
        model_path = "./train/res/{}2{}/{}".format(dsa, dsb, model)
        model = get_model(model_type)
        model.load_state_dict(torch.load("{}/model.pt".format(model_path), map_location='cuda:0'))
        res.append(model)
    return res

def load_images(images, img_size, dataset):
    trans = A.Compose([
        A.RandomCrop(448, 448)
    ])
    dataset = SegDatatset("dataset/{}".format(dataset), train=True, transform=trans)
    imgs = []
    lbls = []
    for image in images:
        cur_img, cur_lbl = dataset[image]
        imgs.append(cur_img)
        lbls.append(cur_lbl)
    return imgs, lbls



def show_imgs_and_lbls(models, images, dsa="PotsdamIRRG" ,dsb="Vaihingen", \
    figsize=(12,8), img_size=(448, 448), model_type="DeepLabV3+"):
    """
    models: ["AdaptSegNet", "MUCSS"......]
    images: [1, 2, 3, ...]
    """
    plt.subplots_adjust(hspace=0.01, wspace=0.01)

    models = load_models(models, model_type, dsa, dsb)
    images, labels = load_images(images, img_size, dsb)
    rows = len(images)
    lines = len(models)+2

    for i, image in enumerate(images):
        print(i)
        plt.subplot(rows, lines, i*lines+1)  
        img_pil = transforms.ToPILImage()(image).convert("RGB")
        plt.imshow(img_pil)
        plt.axis('off')

        plt.subplot(rows, lines, i*lines+2)
        plt.imshow(lbl_img(Image.fromarray(np.uint8(labels[i]))))
        plt.axis('off')

        for j, model in enumerate(models):
            model.eval()
            plt.subplot(rows, lines, i*lines+j+3)
            cur_res = torch.argmax(model(image.unsqueeze(dim=0).cuda()), dim=1)
            res_img = lbl_img_from_tensor(cur_res)
            plt.imshow(res_img)
            plt.axis('off')

    plt.savefig("./res.pdf", bbox_inches="tight", dpi=450)



def lbl_img(lbl_img, palette):
    lbl_img.putpalette(palette)
    return lbl_img


def lbl_img_from_tensor(lbl_torch, palette):
    img = Image.fromarray(np.uint8(lbl_torch.squeeze().detach().cpu().numpy()))
    return lbl_img(img, palette)


def img_from_tensor(img_torch):
    return transforms.ToPILImage()(img_torch.squeeze())


def show_one_image(img):
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def show_two_images(img1, img2):
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.axis("off")
    plt.imshow(img1)
    plt.subplot(122)
    plt.axis("off")
    plt.imshow(img2)
    plt.show()
