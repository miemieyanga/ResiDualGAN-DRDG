import sys
import PIL
import torchvision
from yacs.config import CfgNode as CN
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import logging

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import FloatTensor
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from core.models.residualgan import *
from core.models.resize_block import ResizeBlock
from core.models.build import *
from core.utils.utils import weights_init_normal, UnNormalize, adjust_param, setup_logger, BerHu
from core.utils.data_display import *
from core.datasets.dual_dataset import DualDataset
from core.configs.default import _C as cfg
from transfer import transfer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)), device=device) if device[:4] == "cuda" else \
        FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    validity = D(interpolates)
    fake = Variable(torch.cuda.FloatTensor(np.ones(validity.shape), device=device), requires_grad=False) if device[:4] == "cuda" else \
        Variable(torch.cuda.FloatTensor(np.ones(validity.shape), device=device))
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=validity,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(cfg):
    logger = logging.getLogger("RDG.trainer")
    logger.info("Start training")
    
    save_path = cfg.OUTPUT_DIR + "/models"
    model_path = f"{save_path}"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(f"{save_path}/temp", exist_ok=True)
    
    device = cfg.MODELS.DEVICE
    G_AB, G_BA = build_generators(cfg)
    G_AB.to(device)
    G_BA.to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)  
    
    if cfg.TRAIN.EPOCH != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load(f"{model_path}/G_AB_{cfg.TRAIN.EPOCH}.pth"))
        G_BA.load_state_dict(torch.load(f"{model_path}/G_BA_{cfg.TRAIN.EPOCH}.pth"))
        D_A.load_state_dict(torch.load(f"{model_path}/D_A_{cfg.TRAIN.EPOCH}.pth"))
        D_B.load_state_dict(torch.load(f"{model_path}/D_B_{cfg.TRAIN.EPOCH}.pth"))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    transform_imgs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    transform_dsms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    dual_dataset = DualDataset(f"{cfg.DATASETS.SOURCE_PATH}", f"{cfg.DATASETS.TARGET_PATH}", transform_imgs, transform_dsms)
    dataloader = DataLoader(dual_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.N_CPU)
    
    cycle_loss = torch.nn.L1Loss().to(device)
    depth_loss_f = BerHu() if cfg.TRAIN.DEPTH_LOSS == "berhu" else torch.nn.L1Loss()
    depth_loss_f.to(device)
    
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=cfg.TRAIN.LR, betas=(cfg.TRAIN.B1, cfg.TRAIN.B2)
    )
    optimizer_D_A = torch.optim.RMSprop(D_A.parameters(), lr=cfg.TRAIN.LR)
    optimizer_D_B = torch.optim.RMSprop(D_B.parameters(), lr=cfg.TRAIN.LR)
    un_normalize = UnNormalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, min_lr=0.000001, patience=25)
    scheduler_D_A = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D_A, mode='min', factor=0.5, min_lr=0.000001, patience=25)
    scheduler_D_B = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D_B, mode='min', factor=0.5, min_lr=0.000001, patience=25)
    
    lambda_adv = cfg.LOSS.ADV
    lambda_cycle = cfg.LOSS.CYCLE
    lambda_gp = cfg.LOSS.GP
    lambda_depth = cfg.LOSS.DEPTH
    lambda_depth_cycle = cfg.LOSS.DEPTH_CYCLE
    
    require_depth = True
    if not (lambda_depth or lambda_depth_cycle):
        require_depth = False
    
    batches_done = 0
    prev_time = time.time()
    for epoch in range(cfg.TRAIN.EPOCH, cfg.TRAIN.TOTAL_EPOCH):
        cnt = 0
        g_loss_avg = d_loss_avg = cycle_loss_avg = depth_avg = depth_cyc_avg = 0
        state_str = ""
        for i, (imgs_a, imgs_b, dsm_a, dsm_b) in enumerate(dataloader):
            imgs_a = imgs_a.to(device)
            imgs_b = imgs_b.to(device)
            if require_depth:
                dsm_a = dsm_a.to(device)
                dsm_b = dsm_b.to(device)

            # ----------------------
            #  Train Discriminators
            # ----------------------

            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            # Generate a batch of images
            fake_A = G_BA(imgs_b).detach()
            fake_B = G_AB(imgs_a).detach()

            # ----------
            # Domain A
            # ----------
            # Compute gradient penalty for improved wasserstein training
            gp_A = compute_gradient_penalty(D_A, imgs_a.data, fake_A.data, device)
            # Adversarial loss
            D_A_loss = -torch.mean(D_A(imgs_a)) + torch.mean(D_A(fake_A)) + lambda_gp * gp_A

            # ----------
            # Domain B
            # ----------

            # Compute gradient penalty for improved wasserstein training
            gp_B = compute_gradient_penalty(D_B, imgs_b.data, fake_B.data, device)
            # Adversarial loss
            D_B_loss = -torch.mean(D_B(imgs_b)) + torch.mean(D_B(fake_B)) + lambda_gp * gp_B

            # Total loss
            D_loss = D_A_loss + D_B_loss

            D_loss.backward()
            optimizer_D_A.step()
            optimizer_D_B.step()

            if i % cfg.TRAIN.N_CRITIC == 0:


                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                if require_depth:
                    # Translate images to opposite domain
                    fake_A, depth_b = G_BA(imgs_b, require_depth=True)
                    fake_B, depth_a = G_AB(imgs_a, require_depth=True)

                    # Reconstruct images
                    recov_A, depth_fake_b = G_BA(fake_B, require_depth=True)
                    recov_B, depth_fake_a = G_AB(fake_A, require_depth=True)
                else:
                    fake_A = G_BA(imgs_b, require_depth=False)
                    fake_B = G_AB(imgs_a, require_depth=False)

                    # Reconstruct images
                    recov_A = G_BA(fake_B, require_depth=False)
                    recov_B = G_AB(fake_A, require_depth=False)

                # Adversarial loss
                G_adv = -torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
                # Cycle loss
                G_cycle = cycle_loss(recov_A, imgs_a) + cycle_loss(recov_B, imgs_b)
                #BerHu Loss
                # depth_loss = berhu_loss(depth_b, dsm_b) + berhu_loss(depth_a, dsm_a) + \
                #           berhu_loss(depth_fake_b, F.interpolate(dsm_a, depth_fake_b.shape[2:])) +\
                #           berhu_loss(depth_fake_a, F.interpolate(dsm_b, depth_fake_a.shape[2:]))
                depth_loss = (depth_loss_f(depth_b, dsm_b) + depth_loss_f(depth_a, dsm_a))\
                    if lambda_depth !=0 else 0
                depth_cycle_loss = (depth_loss_f(depth_fake_b, F.interpolate(dsm_a, depth_fake_b.shape[2:])) +\
                        depth_loss_f(depth_fake_a, F.interpolate(dsm_b, depth_fake_a.shape[2:])))\
                            if lambda_depth_cycle !=0 else 0

                # Total loss
                G_loss = lambda_adv * G_adv + lambda_cycle * G_cycle + \
                        lambda_depth * depth_loss + lambda_depth_cycle * depth_cycle_loss

                G_loss.backward()
                optimizer_G.step()
                
                # --------------
                # Log Progress
                # --------------
                if i % 200 == 0:
                    save_image(un_normalize(imgs_a[0]), f"{save_path}/temp/ori.png")
                    save_image(un_normalize(fake_B[0].detach()), f"{save_path}/temp/res.png")
                    save_image(un_normalize(recov_A[0].detach()), f"{save_path}/temp/recov.png")

                # Determine approximate time left
                batches_left = (cfg.TRAIN.TOTAL_EPOCH - cfg.TRAIN.EPOCH) * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / cfg.TRAIN.N_CRITIC)
                prev_time = time.time()
                
                cnt += 1
                g_loss_avg = update_avg(G_adv.item(), g_loss_avg, cnt)
                d_loss_avg = update_avg(D_loss.item(), d_loss_avg, cnt)
                cycle_loss_avg = update_avg(G_cycle.item(), cycle_loss_avg, cnt)
                depth_avg = update_avg(depth_loss.item(), depth_avg, cnt)\
                if lambda_depth !=0 else 0
                depth_cyc_avg = update_avg(depth_cycle_loss.item(), depth_cyc_avg, cnt)\
                if lambda_depth_cycle !=0 else 0
                
                state_str = "epoch={}/{}\tbatch={}/{}\tD loss={:.6f}\tG loss={:.6f}\tcycle={:.6f}\tdepth_avg={:.6f}\tdepth_cyc_avg={:.6f}\tk_AB={:.4f}\tk_BA={:.4f}\tETA:{}\n".format(
                    epoch, cfg.TRAIN.TOTAL_EPOCH, i, len(dataloader), d_loss_avg, g_loss_avg,
                    cycle_loss_avg, depth_avg, depth_cyc_avg, G_AB.generator.k.item(), G_BA.generator.k.item(), time_left
                )
                logger.debug(state_str)
            batches_done += 1
        
        scheduler_G.step(G_loss)
        scheduler_D_A.step(D_A_loss)
        scheduler_D_B.step(D_B_loss)
        logger.info(state_str)
            
        if cfg.TRAIN.CHECKPOINT != -1 and epoch % cfg.TRAIN.CHECKPOINT == 0 and epoch!=0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), f"{model_path}/G_AB_{cfg.TRAIN.EPOCH}.pth")
            torch.save(G_BA.state_dict(), f"{model_path}/G_BA_{cfg.TRAIN.EPOCH}.pth")
            torch.save(D_A.state_dict(), f"{model_path}/D_A_{cfg.TRAIN.EPOCH}.pth")
            torch.save(D_B.state_dict(), f"{model_path}/D_B_{cfg.TRAIN.EPOCH}.pth")
    
    return G_AB, G_BA 


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
    logger = setup_logger("RDG", cfg.OUTPUT_DIR)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    G_AB, _ = train(cfg)
    transfer(G_AB, f"{cfg.DATASETS.SOURCE_PATH}", cfg.DATASETS.TARGET_SIZE,
             cfg.OUTPUT_DIR+"/data", torch.device(cfg.MODELS.DEVICE))
    
    
if __name__ == "__main__":
    main()
    
