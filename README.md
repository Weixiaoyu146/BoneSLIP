# BoneSLIP
A Real-world Bone Scan Dataset and fine-tuning visual-language foundation model for bone metastases diagnosis
# Get Started
## Installation Requirements
To start with this project, make sure that your environment meets the requirements below:

- python >= 3.6.4

- pytorch >= 1.8.0 (with torchvision >= 0.9.0)

- CUDA Version >= 10.2

Run the following command to install required packages.

`pip install -r requirements.txt`
# Tutorial
## Model finetune
`cd BoneSLIP-C/bash run_scripts/BLIP/mianyang+huaxi_BLIP_val_test.sh ${DATAPATH}`
## BoneSLIP-E finetune
`python train.py` Dataset.py中 '.tsv'为二进制图片文件，'.jsonl'为图片描述文件
