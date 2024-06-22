#!/bin/bash

#python3 tools/export_model.py -c /bigvision/PaddleOCR/configs/det/det_mv3_db.yml -o Global.pretrained_model=./det_mv3_db_v2.0_train/best_accuracy Global.save_inference_dir=./inference/det_db_mv

python3 tools/export_model.py -c /media/yobi/hugeDrive/Paddleocr/ocr-api/tools/det_r50_vd_2_cards/config.yml -o Global.checkpoints=/media/yobi/hugeDrive/Paddleocr/ocr-api/tools/det_r50_vd_2_cards/latest Global.save_inference_dir=./inference/det_db_r50_latest_1
