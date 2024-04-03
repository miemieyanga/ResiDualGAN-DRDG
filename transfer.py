from core.datasets.dual_dataset import TransferDataset
from core.models.residualgan import ResizeGenerator
from core.utils.utils import UnNormalize, setup_logger
from core.models.build import build_generators
import torchvision.transforms as transforms
import torch.utils.data as D
import albumentations as A
import os
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
import argparse
from shutil import copyfile
import logging
from core.configs.default import _C as cfg
from tqdm import tqdm


def transfer(model, data_path, target_size, tar_path, device="cuda:0", batch_size=8, require_depth=False):
    logger = logging.getLogger("RDG.transfer")
    logger.info("Start transferring")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    ds = TransferDataset(data_path, transform)
    dataloader = D.DataLoader(ds, batch_size=batch_size, num_workers=8)
    model.eval()
    un_normalize = UnNormalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    un_normalize_depth = UnNormalize((0.5,),(0.5,))
    os.makedirs(f"{tar_path}/images/", exist_ok=True)
    os.makedirs(f"{tar_path}/labels/", exist_ok=True)
    os.makedirs(f"{tar_path}/dsms/", exist_ok=True)
    with torch.no_grad():
        for i,(file_name, img, lbl) in tqdm(enumerate(dataloader), desc='tranfer datasets...'):
            logger.info(f"images shape: {img.shape}, labels shape: {lbl.shape}")
            logger.info(f"file name: {file_name}")
            img = img.to(device)

            if require_depth:
                res, depth = model(img, require_depth=True)
            else:
                res = model(img, require_depth=False)

            for j in range(len(img)):
                img_path = "{}/images/{}".format(tar_path, file_name[j])
                lbl_path = "{}/labels/{}".format(tar_path, file_name[j])
                dsm_path = "{}/dsms/{}".format(tar_path, file_name[j])
                save_image(un_normalize(res[j]), img_path)
                
                if require_depth:
                    save_image(un_normalize_depth(depth[j]), dsm_path)

                cur_lbl = transforms.ToPILImage()(lbl[j])
                cur_lbl = cur_lbl.resize((target_size, target_size), Image.NEAREST)
                cur_lbl.save(lbl_path)
                
    copyfile(f"{data_path}/all.txt", f"{tar_path}/all.txt")
    copyfile(f"{data_path}/train.txt", f"{tar_path}/train.txt")
    copyfile(f"{data_path}/test.txt", f"{tar_path}/test.txt")
            
            
def transfer_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg",
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
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
    
    model_path = f"{cfg.OUTPUT_DIR}/models/G_AB_{cfg.TRAIN.TOTAL_EPOCH-1}.pth"
    data_path = f"{cfg.DATASETS.SOURCE_PATH}"
    target_size = cfg.DATASETS.TARGET_SIZE
    target_path = cfg.GENERATE_PATH
    model, _ = build_generators(cfg)
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda:0")
    transfer(model, data_path, target_size, target_path)
    
if __name__ == "__main__":
    transfer_main()