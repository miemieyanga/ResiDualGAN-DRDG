import imp
from statistics import mode
from tkinter.messagebox import NO
from .residualgan import *
from .resize_block import *
import segmentation_models_pytorch as smp


def build_generators(cfg):
    G_AB = ResizeGenerator(
        cfg.MODELS.IN_CHANNELS,
        (cfg.DATASETS.TARGET_SIZE, cfg.DATASETS.TARGET_SIZE),
        generator=cfg.MODELS.GENERATOR,
        resize_block=cfg.MODELS.RESIZE_BLOCK,
        interpolation=cfg.MODELS.INTERPOLATION,
        k = cfg.MODELS.K,
        k_grad=cfg.MODELS.K_GRAD,
        post_resize=cfg.MODELS.POST_RESIZE
    )
    # G_AB.generator.k = torch.nn.Parameter(torch.Tensor([float(cfg.MODELS.K)]))
    G_BA = ResizeGenerator(
        cfg.MODELS.IN_CHANNELS,
        ( cfg.DATASETS.SOURCE_SIZE, cfg.DATASETS.SOURCE_SIZE),
        generator=cfg.MODELS.GENERATOR,
        resize_block=cfg.MODELS.RESIZE_BLOCK,
        interpolation=cfg.MODELS.INTERPOLATION,
        k = cfg.MODELS.K,
        k_grad=cfg.MODELS.K_GRAD,
        post_resize=cfg.MODELS.POST_RESIZE
    )
    # G_BA.generator.k = torch.nn.Parameter(torch.Tensor([float(cfg.MODELS.K)]))
    return G_AB, G_BA

def build_seg_model(cfg):
    model = None
    if cfg.MODELS.BACKBONE == "UNet":
        model = smp.Unet(
                encoder_name=cfg.MODELS.ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=cfg.MODELS.PRETRAIN,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=cfg.MODELS.IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=cfg.MODELS.CLASSES,  # model output channels (number of classes in your dataset)
            )
    elif cfg.MODELS.BACKBONE == "DeepLabV3":
        model = smp.DeepLabV3(
                encoder_name=cfg.MODELS.ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=cfg.MODELS.PRETRAIN,
                in_channels=cfg.MODELS.IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=cfg.MODELS.CLASSES,  # model output channels (number of classes in your dataset)
            )
    elif cfg.MODELS.BACKBONE == "DeepLabV3+":
        model = smp.DeepLabV3Plus(
                encoder_name=cfg.MODELS.ENCODER,
                encoder_weights=cfg.MODELS.PRETRAIN,
                in_channels=cfg.MODELS.IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=cfg.MODELS.CLASSES,  # model output channels (number of classes in your dataset)
            )
    elif cfg.MODELS.BACKBONE == "MiT":
        raise NotImplementedError
        # model = Segformer(num_classes=6, decoder_dim=256, dims=(64,128,320,448)).cuda()
    else:
        raise NotImplementedError
    
    
    return model