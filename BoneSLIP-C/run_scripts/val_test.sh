#!/bin/bash

# Usage: see example script below.
# bash run_scripts/zeroshot_eval.sh 0 \
#     ${path_to_dataset} ${dataset_name} \
#     ViT-B-16 RoBERTa-wwm-ext-base-chinese \
#     ${ckpt_path}

# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

path=/hdd/wxy/dataset/CLIP/Chinese_CLIP/kaggle/chest_xray
#context_length=76
#path=/hdd/wxy/dataset/CLIP/Chinese_CLIP/all/temp_part_file
dataset=chestxray-CLIP
#datapath=${path}/datasets/${dataset}/test
datapath=${path}/test
savedir=/hdd/wxy/models/chinese_CLIP/chestxray
vision_model=ViT-B-16 # ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese

resume=/hdd/wxy/models/chinese_CLIP/pretrained/clip_cn_vit-b-16.pt

label_file=${path}/label_cn.txt
index=${7:-}

mkdir -p ${savedir}

python -u cn_clip/eval/zeroshot_evaluation.py \
    --datapath="${datapath}" \
    --label-file=${label_file} \
    --save-dir=${savedir} \
    --dataset=${dataset} \
    --index=${index} \
    --img-batch-size=64 \
    --resume=${resume} \
    --vision-model=${vision_model} \
    --text-model=${text_model}
