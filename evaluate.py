import os
from asyncio.log import logger
from distutils.command.build import build
from statistics import mode
from matplotlib.pyplot import axis
import torch
import numpy as np
from core.datasets.seg_dataset import SegDataset
from core.utils.utils import _data_part,setup_logger,iou,f1
from core.utils.data_display import lbl_img_from_tensor
from core.models.build import build_seg_model
import torch.functional as F
import pandas as pd
import argparse
import albumentations as A
from core.configs.default_seg import _C as cfg


def get_test_loader(dataset, batch_size, num_workers=1, shuffle=False):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def get_palette(dataset):
    palette = []
    for item in dataset.palette:
        palette += item
    palette += [0, 0, 0] * 249
    return palette


def seg_eval_once(model, dataset, logger, save_path, batch_size = 1, generate_res=True):
    dataset.get_file_name = True
    dataloader = get_test_loader(dataset, batch_size)
    model.eval()
    iou_sum = np.zeros((7,))
    f1_sum = np.zeros((7,))
    palette = get_palette(dataset)
    os.makedirs(f"{save_path}/res", exist_ok=True)
    for i, (file_name, input, target) in enumerate(dataloader):
        logger.info(f"evaluating {i}\n")
        input = input.cuda()
        target = target.cuda()
        y_pred = model.predict(input)
        iou_sum += iou(y_pred, target)
        f1_sum += f1(y_pred, target)
        if generate_res:
            for j in range(y_pred.shape[0]):
                lbl_img = lbl_img_from_tensor(torch.argmax(y_pred[j], dim=0), palette)
                lbl_img.save(f"{save_path}/res/{file_name[j]}")
    c_iou = iou_sum / len(dataloader)
    c_f1 = f1_sum / len(dataloader)
    return c_iou, c_f1

def evl(model, dataset, logger, save_path, batch_size, generate_res):
    iou, f1 = seg_eval_once(model, dataset, logger, save_path, batch_size, generate_res)
    columns = dataset.label + ["overall"]
    data = [[],[]]
    for i in range(len(iou)):
        data[0].append(iou[i])
        data[1].append(f1[i])
    logger.info(f"finish, iou={iou[-1]}, f1={f1[-1]}")
    
    pf = pd.DataFrame(data=data, columns=columns, index=["iou", "f1"], dtype=float)
    pf.to_excel(f"{save_path}/evl_res.xlsx", index=True)
    
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
    parser.add_argument(
        "--model_path",
        default="",
        type=str
    )
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("segmentation", cfg.OUTPUT_DIR)
    
    model = build_seg_model(cfg)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    
    _all, _train = _data_part(cfg.DATASETS.EVL_PART)
    trans = A.Compose([A.RandomCrop(cfg.DATASETS.SIZE, cfg.DATASETS.SIZE)])
    val_ds = SegDataset(cfg.DATASETS.EVL_DATASET_PATH, all=_all, train=_train, transform=trans)   
    
    evl(model, val_ds, logger, cfg.OUTPUT_DIR+"/res", cfg.DATASETS.EVL_BATCH, True)

if __name__ == "__main__":
    main()