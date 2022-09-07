import argparse
import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
import time
import copy
import logging
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.autograd import Variable
from core.configs.default_seg import _C as cfg
from core.utils.utils import weights_init_normal, UnNormalize, adjust_param, setup_logger, _data_part
from core.models.focal_loss import FocalLoss2d
from core.models.build import build_seg_model
from core.models.output_discriminator import OutputDiscriminator
from core.datasets.seg_dataset import SegDataset
from core.utils.utils import *
from evaluate import evl

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
    
def train(cfg):
    logger = logging.getLogger("segmentation.trainer")
    logger.info("Start training")
    device = cfg.MODELS.DEVICE
    model = build_seg_model(cfg).to(device)
    save_path = cfg.OUTPUT_DIR + "/model"
    if cfg.TRAIN.EPOCH >0:
        model.load_state_dict(torch.load(f"{save_path}/model.pt"))

    discriminator = OutputDiscriminator(cfg.MODELS.CLASSES).to(device)
    if cfg.TRAIN.LOSS == "ce": 
        criterion = CrossEntropyLoss(ignore_index=255).to(device)
    elif cfg.TRAIN.LOSS == "focal":
        criterion = FocalLoss2d(ignore_index=255).to(device)
    else:
        raise Exception("not impletment yet")
    bce_loss = BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), cfg.TRAIN.LR)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), cfg.TRAIN.LR_D)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=0.00002, patience=20)

    trans = A.Compose([
        A.RandomCrop(cfg.DATASETS.SIZE, cfg.DATASETS.SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ])
    
    _all, _train = _data_part(cfg.DATASETS.SOURCE_PART)
    ds = SegDataset(cfg.DATASETS.SOURCE_DATASET_PATH, all=_all, train=_train, transform=trans)
    # ds = SegDatatset(dataset_path, all=True, transform=trans)
    dataloader = DataLoader(ds, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.N_CPU, shuffle=True)
    
    _all, _train = _data_part(cfg.DATASETS.VAL_PART)
    val_ds = SegDataset(cfg.DATASETS.VAL_DATASET_PATH, all=_all, train=_train, transform=trans)
    val_dataloader = DataLoader(val_ds, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.N_CPU, shuffle=True)
    
    _all, _train = _data_part(cfg.DATASETS.TARGET_PART)
    target_ds = SegDataset(cfg.DATASETS.TARGET_DATASET_PATH, all=_all, train=_train, transform=trans, iter_len=len(ds), get_label=False)
    target_dataloader = DataLoader(target_ds, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.N_CPU, shuffle=True)

    source_label = 0
    target_label = 1
    
    save_iou = 0
    save_model = None
    os.makedirs(save_path , exist_ok=True)
    start_t = time.time()
    for epoch in range(cfg.TRAIN.EPOCH, cfg.TRAIN.TOTAL_EPOCH):
        model.train()
        discriminator.train()
        adv_loss_sum = 0
        epoch_loss = 0
        d_loss_sum = 0
        source_iter = enumerate(dataloader)
        target_iter = enumerate(target_dataloader)
        t1 = time.time()
        # train
        for i in range(len(dataloader)):
            optimizer.zero_grad()
            optimizer_D.zero_grad()

            # train segmentation module

            # don't accumulate gradients of Driscriminator
            for param in discriminator.parameters():
                param.requires_grad = False

            _, (input, target) = source_iter.__next__()
            input = input.to(device)
            target = target.to(device).long()
            y_pred = model.forward(input)
            
            if cfg.MODELS.BACKBONE == "MiT":
                y_pred = F.interpolate(y_pred, scale_factor=4)
                
            loss = criterion(y_pred, target)
            loss.backward()

            if cfg.MODELS.OUT_ADV:
                # train with target
                _, (input_target, _) = target_iter.__next__()
                input_target = input_target.to(device)
                y_pred_target = model.forward(input_target)
                
                if cfg.MODELS.BACKBONE == "MiT":
                    y_pred_target = F.interpolate(y_pred_target, scale_factor=4)
                
                d_out = discriminator.forward(F.softmax(y_pred_target, dim=1))                
                adv_loss = bce_loss(d_out, Variable(torch.FloatTensor(d_out.data.size()).fill_(source_label)).to(device))
                adv_loss = adv_loss*cfg.LOSS.ADV
                # print("adv_loss:{}".format(adv_loss))
                adv_loss.backward()

                # train discriminator

                for param in discriminator.parameters():
                    param.requires_grad = True

                y_pred = y_pred.detach()
                y_pred_target = y_pred_target.detach()

                d_out = discriminator(F.softmax(y_pred, dim=1))
                d_loss_1 = bce_loss(d_out, Variable(torch.FloatTensor(d_out.data.size()).fill_(source_label)).to(device))

                d_out = discriminator(F.softmax(y_pred_target, dim=1))
                d_loss_2 = bce_loss(d_out, Variable(torch.FloatTensor(d_out.data.size()).fill_(target_label)).to(device))

                d_loss = d_loss_1 + d_loss_2
                # print("d_loss:{}".format(d_loss))
                d_loss.backward()
                optimizer_D.step()

                adv_loss_sum += adv_loss.data
                d_loss_sum += d_loss.data

            optimizer.step()

            # log info while training

            epoch_loss += loss.data

        t2 = time.time()
        logger.info('epoch={}\ti={}\tloss:{:.3f}\tadv_loss:{:.5f}\td_loss:{:.3f}\ttime:{:.3f} s\tlr:{:.6f}'.format
                      (epoch, i, epoch_loss / len(dataloader), adv_loss_sum/ len(dataloader), 
                      d_loss_sum/ len(dataloader), t2 - t1, optimizer.param_groups[0]['lr']))
        scheduler.step(epoch_loss)

        # validation using images of target domain

        val_loss = 0
        iou_sum = np.zeros((7,))
        f1_sum = np.zeros((7,))
        model.eval()
        discriminator.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_dataloader):
                input = input.to(device)
                target = target.to(device).long()
                y_pred = model.predict(input)
                if cfg.MODELS.BACKBONE == "MiT":
                    y_pred = F.interpolate(y_pred, scale_factor=4)
                val_loss += criterion(y_pred, target).data
                iou_sum += iou(y_pred, target)
                f1_sum += f1(y_pred, target)
            avg_loss = val_loss / len(val_dataloader)
            c_iou = iou_sum/len(val_dataloader)
            c_f1 = f1_sum/len(val_dataloader)
            logger.info(f"epoch={epoch}\tavg_loss={avg_loss}\tiou={c_iou}\tf1={c_f1}\n")

        if c_iou[-1] >= save_iou:
            save_iou = c_iou[-1]
            logger.info(f"save model with iou=\n{c_iou}")
            model.eval()
            save_model = copy.deepcopy(model)
            torch.save(model.state_dict(), f'{save_path}/model.pt')
            
    return save_model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg",
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "-opts",
        help="Modify config options using the command-line",
        default="",
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("segmentation", cfg.OUTPUT_DIR)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    model = train(cfg)
    
    _all, _train = _data_part(cfg.DATASETS.EVL_PART)
    trans = A.Compose([A.RandomCrop(cfg.DATASETS.SIZE, cfg.DATASETS.SIZE)])
    val_ds = SegDataset(cfg.DATASETS.EVL_DATASET_PATH, all=_all, train=_train, transform=trans)   
    
    evl(model, val_ds, logger, cfg.OUTPUT_DIR+"/res", cfg.DATASETS.EVL_BATCH, cfg.DATASETS.EVL_GENERATE)

if __name__ == "__main__":
    main()