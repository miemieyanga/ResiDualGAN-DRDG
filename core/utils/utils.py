import imp
from turtle import forward
import numpy as np
from torch import FloatTensor
from torch.autograd import Variable
import torch.autograd as autograd
import torch
import math
import segmentation_models_pytorch as smp
import logging
import sys
import os
import torch.nn as nn

def get_model(model_type, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=6):
    model = None
    if model_type == "UNet":
        model = smp.Unet(
                encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=classes,  # model output channels (number of classes in your dataset)
            ).cuda()
    elif model_type == "DeepLabV3":
        model = smp.DeepLabV3(
                encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=classes,  # model output channels (number of classes in your dataset)
            ).cuda()
    elif model_type == "DeepLabV3+":
        model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=classes,  # model output channels (number of classes in your dataset)
            ).cuda()
    return model

def weights_init_normal(m):
    try:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    except:
        return 


def adjust_param(cur_epoch, total_epoch):
    t = float(cur_epoch)/total_epoch
    return math.exp(-5*((1-t)**2))


def iou(output, target, n_classes=6):
    smooth = 1e-5
    ious = []
    output = output.argmax(dim=1)
    for cls in range(n_classes):
        pred_inds = output == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        ious.append((float(intersection)+smooth)/ (float(union) + smooth))
    ious.append(sum(ious)/6)
    return np.array(ious)*100


def tp(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output == cls
        target_inds = target == cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(np.float)


def fp(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output == cls
        target_inds = target != cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(np.float)


def fn(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output != cls
        target_inds = target == cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(np.float)


def tf(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output != cls
        target_inds = target != cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(np.float)


def f1(output, target, n_classes=6):
    smooth = 1e-5
    output = output.argmax(dim=1)
    f1 = (2*tp(output, target, n_classes) + smooth)/\
        (2*tp(output, target, n_classes)+fp(output, target, n_classes)+fn(output, target, n_classes) + smooth)
    f1 = np.append(f1, np.sum(f1)/6)
    return f1*100


def log_loss(epoch, time, loss, iou, f1, lr, file_path="./log.txt"):
    with open(file_path, 'a+') as f:
        f.write("epoch={}\ttime={:.3f}\tloss={:.3f}\tiou={:.3f}\tf1={:.3f}\tlr={:.6f}\n"
        .format(epoch, time, loss, iou, f1, lr))


def up_lower_limit_str(data):
    min_n = min(data)
    max_n = max(data)
    return "{:.2f}±{:.2f}".format(float(min_n+max_n)/2, float(max_n-min_n)/2)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        for i, m, s in zip(range(tensor.size(0)), self.mean, self.std):
            t = tensor[i]
            t.mul_(s).add_(m)
        return tensor
    
    
def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def update_avg(new_data, data, cnt):
    assert cnt != 0
    return float(new_data + (cnt - 1) * data) / cnt


class BerHu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 10e-5

    def forward(self, pred, target):
        abs_1 = torch.abs((pred - target))
        L = 0.2 * torch.max(abs_1)
        mask = (abs_1 <= L)
        abs_2 = (torch.square(abs_1)+ L**2)/(2*L+self.eps)
        return (abs_1[mask].sum() + abs_2[~mask].sum())/torch.numel(pred)

def _data_part(str):
    if str == "all":
        return True, False
    elif str == "train":
        return False, True
    elif str == "test":
        return False, False
    else:
        raise KeyError