export CUDA_VISIBLE_DEVICES=2
# RDG_PIRRG2V
export output_dir="./res"
export datasets_path="./datasets"
nohup python train_residualgan.py -cfg ./configs/residualgan.yaml -opts \
LOSS.DEPTH 0.0 LOSS.DEPTH_CYCLE 0.0 LOSS.ADV 1 LOSS.CYCLE 10 \
OUTPUT_DIR $output_dir MODELS.GENERATOR "UNet" MODELS.K_GRAD False \
DATASETS.SOURCE_PATH $datasets_path/PotsdamIRRG \
DATASETS.TARGET_PATH $datasets_path/Vaihingen_dsm \
DATASETS.TRANS_SOURCE_PATH $datasets_path/PotsdamIRRG \
DATASETS.SOURCE_SIZE 896 \
DATASETS.TARGET_SIZE 512

# DRDG_PIRRG2V
export output_dir="./res"
export datasets_path="./datasets"
nohup python train_residualgan.py -cfg ./configs/residualgan.yaml -opts \
LOSS.DEPTH 2.0 LOSS.DEPTH_CYCLE 1.0 LOSS.ADV 5 LOSS.CYCLE 10 \
OUTPUT_DIR $output_dir MODELS.GENERATOR "UNet" MODELS.K_GRAD False \
DATASETS.SOURCE_PATH $datasets_path/PotsdamIRRG \
DATASETS.TARGET_PATH $datasets_path/Vaihingen_dsm \
DATASETS.TRANS_SOURCE_PATH $datasets_path/PotsdamIRRG \
DATASETS.SOURCE_SIZE 896 \
DATASETS.TARGET_SIZE 512

# RDG_V2PRGB
export output_dir="./res"
export datasets_path="./datasets"
nohup python train_residualgan.py -cfg ./configs/residualgan.yaml -opts \
LOSS.DEPTH 0.0 LOSS.DEPTH_CYCLE 0.0 LOSS.ADV 1 LOSS.CYCLE 10 \
OUTPUT_DIR $output_dir MODELS.GENERATOR "UNet" MODELS.K_GRAD False \
DATASETS.SOURCE_PATH $datasets_path/Vaihingen_dsm \
DATASETS.TARGET_PATH $datasets_path/PotsdamRGB \
DATASETS.TRANS_SOURCE_PATH $datasets_path/Vaihingen \
DATASETS.SOURCE_SIZE 512 \
DATASETS.TARGET_SIZE 896

# DRDG_V2PRGB
export output_dir="./res"
export datasets_path="./datasets"
nohup python train_residualgan.py -cfg ./configs/residualgan.yaml -opts \
LOSS.DEPTH 2.0 LOSS.DEPTH_CYCLE 1.0 LOSS.ADV 5 LOSS.CYCLE 10 \
OUTPUT_DIR $output_dir MODELS.GENERATOR "UNet" MODELS.K_GRAD False \
DATASETS.SOURCE_PATH $datasets_path/Vaihingen_dsm \
DATASETS.TARGET_PATH $datasets_path/PotsdamRGB \
DATASETS.TRANS_SOURCE_PATH $datasets_path/Vaihingen \
DATASETS.SOURCE_SIZE 512 \
DATASETS.TARGET_SIZE 896

# &&python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV True TRAIN.BATCH_SIZE 32 DATASETS.EVL_BATCH 32 DATASETS.SOURCE_DATASET_PATH $output_dir/data OUTPUT_DIR $output_dir
