import random

from tqdm import tqdm

import os
from PIL import Image
import numpy as np
import torch
import clip
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from Dataset import MyDataset
from clip.model import convert_weights
from BBS import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model,preprocess = clip.load('ViT-B/32',device=device,jit=False)
checkpoint=torch.load("/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/model/18_model_pkl")
model.load_state_dict(checkpoint.state_dict())


test_dataset=MyDataset(is_train=False,preprocess=preprocess)
BATCH_SIZE=24
test_labels = torch.tensor([item[2] for item in test_dataset])
print("开始均衡采样")
test_sampler = BalancedBatchSampler(test_labels, BATCH_SIZE, 1)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler,pin_memory=True)


print("数据集加载完毕")
total_precision=0
number=0
for images, texts, label in test_dataloader:
    texts=texts.to(device)
    images=images.to(device)
    i=0
    n_correct=0
    number+=1
    for image in images:
        image=image.unsqueeze(0).to(device)

        with torch.no_grad():
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(texts)

            logits_per_image, logits_per_text = model(image, texts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        if np.argmax(probs) == i:
            n_correct += 1
        i+=1
    precision=n_correct/len(images)
    print("batch:{}  precision:{}".format(number-1,precision))
    total_precision+=precision
mean_precision=total_precision/number
print("mean_precision is {}".format(mean_precision))






