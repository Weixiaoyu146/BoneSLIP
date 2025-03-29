import os
from PIL import Image
import numpy as np
import torch
import clip
import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from Dataset import MyDataset
from clip.model import convert_weights
from BBS import *

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model,preprocess = clip.load('ViT-B/32',device=device,jit=False)
model_text=TextCLIP(model)
model_image=ImageCLIP(model)
model_text=nn.DataParallel(model_text)
model_image=nn.DataParallel(model_image)


optimizer = optim.Adam(model.parameters(), lr=1e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# scheduler = lr_scheduler.StepLR(
#         optimizer, step_size=10, gamma=0.1)
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data=p.data.float()
        p.grad.data=p.grad.data.float()

# if device == "cpu":
#     model.float()
# else:
#     clip.model.convert_weights_to_fp16(model)



train_dataset=MyDataset(is_train=True,preprocess=preprocess)
BATCH_SIZE=24
train_labels = torch.tensor([item[2] for item in train_dataset])
print("开始均衡采样/clip2")
train_sampler = BalancedBatchSampler(train_labels, BATCH_SIZE, 1)
train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler,pin_memory=True)
# train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,drop_last=True,shuffle=True, pin_memory=True)
EPOCH=30
i=0
print("数据集加载完毕")

for epoch in range(EPOCH):
    tr_loss=0
    for images,texts,label in train_dataloader:
        i=i+1
        optimizer.zero_grad()

        images=images.to(device)
        texts=texts.to(device)
        image_embedding = model_image(images)
        text_embedding=model_text(texts)

        logit_scale=model.logit_scale.exp()
        logits_per_image,logits_per_text=create_logits(image_embedding,text_embedding,logit_scale)
        ground_truth=torch.arange(BATCH_SIZE).to(device)
        total_loss=(loss_img(logits_per_image,ground_truth)+loss_txt(logits_per_text,ground_truth))/2
        tr_loss=tr_loss+total_loss
        total_loss.backward()

        if device == "cpu":
            optimizer.step()
        else :
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        print(f"[{epoch}]-[{i}]: {total_loss.item()}")
    tr_loss = "{:.3f}".format(tr_loss)
    torch.save(model,'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/model4/'+str(epoch)+'_'+tr_loss+'_model_pkl')
    print("epoch{} model have saved  train loss is {}".format(epoch,tr_loss))
















