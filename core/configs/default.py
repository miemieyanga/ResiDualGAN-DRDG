from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = "./"

_C.MODELS = CN()
_C.MODELS.IN_CHANNELS = 3
_C.MODELS.GENERATOR = "UNet"
_C.MODELS.RESIZE_BLOCK = False
_C.MODELS.INTERPOLATION = "bilinear"
_C.MODELS.DEVICE = "cuda"
_C.MODELS.K = 1.0
_C.MODELS.K_GRAD = False
_C.MODELS.POST_RESIZE = True

_C.DATASETS = CN()
_C.DATASETS.SOURCE_PATH = "./PotsdamIRRG"
_C.DATASETS.TRANS_SOURCE_PATH = _C.DATASETS.SOURCE_PATH
_C.DATASETS.TARGET_PATH = "./Vaihingen_dsm"
_C.DATASETS.SOURCE_SIZE = 896
_C.DATASETS.TARGET_SIZE = 512

_C.LOSS = CN()
_C.LOSS.ADV = 5
_C.LOSS.CYCLE = 10
_C.LOSS.GP = 1
_C.LOSS.DEPTH = 1.5
_C.LOSS.DEPTH_CYCLE = 1.0

_C.TRAIN = CN()
_C.TRAIN.EPOCH = 0
_C.TRAIN.TOTAL_EPOCH = 101
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.LR = 0.0005
_C.TRAIN.B1 = 0.5
_C.TRAIN.B2 = 0.999
_C.TRAIN.N_CPU = 8
_C.TRAIN.N_CRITIC = 5
_C.TRAIN.CHECKPOINT = 100
_C.TRAIN.DEPTH_LOSS = "berhu"


