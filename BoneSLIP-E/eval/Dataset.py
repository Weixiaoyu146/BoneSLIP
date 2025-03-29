import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

import clip
import cv2 as cv
import json
import json
import csv
from PIL import Image
import base64
import io

def convert_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    # image.save(output_filename)
    return image

def load_csv(path):
    csv_reader = csv.reader(open(path))
    image_list=[]
    for row in csv_reader:
        data = str(row[0]).split('	')
        # print(data[0])  # id
        # print(data[1])  # base64
        image = convert_image(data[1])
        image_list.append(image)
    return image_list


class EvalTxtDataset(Dataset):
    def __init__(self):
        self.text_ids=[]
        self.texts=[]
        jsonl_filename= r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/test/test_texts.jsonl'
        with open(jsonl_filename, "r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                text_id = obj['text_id']
                text = obj['text']
                # self.texts.append((text_id, text))
                self.text_ids.append(text_id)
                self.texts.append(text)

        self.texts = clip.tokenize(self.texts)



    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_id=self.text_ids[idx]
        text=self.texts[idx]
        return text_id, text

class EvalImgDataset(Dataset):
    def __init__(self,preprocess):
        self.img_process = preprocess
        self.read_csv = r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/test/test_imgs.tsv'
        self.images = load_csv(self.read_csv)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_id=idx+1
        # 加载图像
        image = img.convert('RGB')
        # 对图像进行转换
        image = self.img_process(image)

        return img_id, image