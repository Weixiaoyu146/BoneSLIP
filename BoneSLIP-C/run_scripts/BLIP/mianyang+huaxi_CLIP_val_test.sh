#!/bin/bash

# Usage: see example script below.
# bash run_scripts/zeroshot_eval.sh 0 \
#     ${path_to_dataset} ${dataset_name} \
#     ViT-B-16 RoBERTa-wwm-ext-base-chinese \
#     ${ckpt_path}

# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

# must change resume, dataset(output file), path(datapath), save path, label_file
path=/hdd/wxy/dataset/CLIP/Chinese_CLIP/mianyang+huaxi/eval
#dataset=mianyang+huaxi
dataset=mianyang+huaxi_origin
datapath=${path}/test
savedir=/hdd/wxy/models/chinese_CLIP/val_result/mianyang+huaxi_origin
vision_model=ViT-B-16 # ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
#resume=/hdd/wxy/models/chinese_CLIP/train4.0_finetune_vit-b-16_roberta-base_bs8_3gpu/checkpoints/epoch_latest.pt
resume=/hdd/wxy/models/chinese_CLIP/pretrained/clip_cn_vit-b-16.pt
label_file=${path}/label_cn.txt
#label_file=${path}/label_origin/label_cn.txt
index=${7:-}

mkdir -p ${savedir}

#python -u cn_clip/eval/zeroshot_evaluation.py \
python -u cn_clip/eval/zeroshot_eva_test.py \
    --datapath="${datapath}" \
    --label-file=${label_file} \
    --save-dir=${savedir} \
    --dataset=${dataset} \
    --index=${index} \
    --img-batch-size=64 \
    --resume=${resume} \
    --vision-model=${vision_model} \
    --text-model=${text_model}
