export model_path="./released/model.pt"
export output_dir="./res"
export evl_data_path="./datasets/Vaihingen"
python evaluate.py --model_path $model_path -cfg ./configs/segmentation.yaml -opts DATASETS.EVL_BATCH 32 OUTPUT_DIR $output_dir DATASETS.EVL_DATASET_PATH $evl_data_path