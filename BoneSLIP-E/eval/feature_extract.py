# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation. (with single-GPU)
'''
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
from Dataset import *
from clip.model import convert_weights
from torch.utils.data.sampler import SequentialSampler


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


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data=p.data.float()
        p.grad.data=p.grad.data.float()



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    # checkpoint = torch.load(r"/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/model4/19_4096.000_model_pkl")
    # model.load_state_dict(checkpoint.state_dict())
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)


    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    text_sampler = SequentialSampler(EvalTxtDataset())
    text_dataloader = DataLoader(
        EvalTxtDataset(),
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        sampler=text_sampler,
        drop_last=False
    )
    image_sampler = SequentialSampler(EvalImgDataset(preprocess=preprocess))
    image_dataloader = DataLoader(
        EvalImgDataset(preprocess=preprocess),
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        sampler=image_sampler,
        drop_last=False
    )

    print("start to extract features!")
    extract_text_feats_path=r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/eval_CLIP/extract_text_feats.jsonl'
    write_cnt = 0
    with open(extract_text_feats_path, "w") as fout:
        model.eval()
        with torch.no_grad():
            for text_ids, texts in text_dataloader:
                texts = texts.to(device)
                text_features = model_text(texts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                for text_id, text_feature in zip(text_ids.tolist(), text_features.tolist()):
                    fout.write("{}\n".format(json.dumps({"text_id": text_id, "feature": text_feature})))
                    write_cnt += 1
    print('{} text features are stored in {}'.format(write_cnt, extract_text_feats_path))

    extract_image_feats_path=r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/eval_CLIP/extract_image_feats.jsonl'
    write_cnt = 0
    with open(extract_image_feats_path, "w") as fout:
        model.eval()
        with torch.no_grad():
            for image_ids, images in image_dataloader:
                images = images.to(device)
                image_features = model_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                for image_id, image_feature in zip(image_ids.tolist(), image_features.tolist()):
                    fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                    write_cnt += 1
    print('{} image features are stored in {}'.format(write_cnt, extract_image_feats_path))


