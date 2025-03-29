import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
import random
from BBS import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model,preprocess = clip.load('ViT-B/32',device=device,jit=False)
checkpoint=torch.load(r"/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/model4/19_4096.000_model_pkl")
model.load_state_dict(checkpoint.state_dict())


test_dataset=MyDataset(is_train=False,preprocess=preprocess)
BATCH_SIZE=24
test_labels = torch.tensor([item[2] for item in test_dataset])
print("开始均衡采样")
test_sampler = BalancedBatchSampler(test_labels, BATCH_SIZE, 1)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler,pin_memory=True)


print("数据集加载完毕")
precision=0
n_correct=0
num=0
true_labels=[]
pred_labels=[]
for images, texts, label in test_dataloader:
    num+=1
    print(num)
    texts=texts.to(device)
    images=images.to(device)
    j=random.randint(0,BATCH_SIZE-1)
    # for image in images:
    image=images[j]
    true_label=label[j]
    image=image.unsqueeze(0).to(device)

    with torch.no_grad():
        # image_features = model.encode_image(image)
        # text_features = model.encode_text(texts)

        logits_per_image, logits_per_text = model(image, texts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    if np.argmax(probs) == j:
        n_correct += 1

    a=np.argmax(probs)
    pred_label=label[a]
    true_labels.append(true_label)
    pred_labels.append(pred_label)

y_true = np.array(true_labels)
y_pred = np.array(pred_labels)
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 计算精确率
macro_precision = precision_score(y_true, y_pred, average='macro')
micro_precision=precision_score(y_true, y_pred, average='micro')
print("macro_precision:", macro_precision)
print("micro_precision:", micro_precision)

# 计算召回率
macro_recall = recall_score(y_true, y_pred, average='macro')
micro_recall = recall_score(y_true, y_pred, average='micro')
print("macro_recall:", macro_recall)
print("micro_recall:", micro_recall)


# 计算F1分数
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')
print("macro_f1-score:", macro_f1)
print("micro_f1-score:", micro_f1)

precision=n_correct/num
print("/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/model4/19_4096.000_model_pkl precision is {}   num:{}".format(precision,num))

labels=[]
for i in range(1,40):
    labels.append('class {}'.format(i))


# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
