export output_dir="./res/drdgv3"
export datasets_path="./datasets"
python train_residualgan.py -cfg ./configs/residualgan.yaml -opts LOSS.DEPTH 2.0 LOSS.DEPTH_CYCLE 1.0 LOSS.ADV 5 LOSS.CYCLE 10 TRAIN.DEPTH_LOSS berhu OUTPUT_DIR $output_dir MODELS.GENERATOR "UNet" MODELS.K_GRAD False DATASETS.SOURCE_PATH $datasets_path/PotsdamIRRG DATASETS.TARGET_PATH $datasets_path/Vaihingen_dsm &&
python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV False TRAIN.BATCH_SIZE 32 DATASETS.EVL_BATCH 32 DATASETS.SOURCE_DATASET_PATH $output_dir/data OUTPUT_DIR $output_dir