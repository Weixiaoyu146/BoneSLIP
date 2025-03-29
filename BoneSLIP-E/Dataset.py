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

class MyDataset(Dataset):
    def __init__(self,is_train,preprocess):
        # 1.根目录(根据自己的情况更改)
        # self.img_root = img_root
        # self.meta_root = meta_root
        # 2.训练图片和测试图片地址(根据自己的情况更改)
        # self.train_set_file = os.path.join(meta_root,'train.txt')
        # self.test_set_file = os.path.join(meta_root,'test.txt')
        # 3.训练 or 测试(根据自己的情况更改)
        self.is_train = is_train
        # 4.处理图像
        self.img_process = preprocess
        # 5.获得数据(根据自己的情况更改)
        self.samples = []
        self.sam_labels = []

        # 5.1 训练还是测试数据集
        self.read_file = ""
        self.read_csv=""
        self.image_list=[]
        self.labels=[]
        if is_train:
            self.read_file = '/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/train/train_texts.jsonl'
            self.read_csv = '/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/train/train_imgs.tsv'
            # self.read_file = '/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/valid/valid_texts.jsonl'
            # self.read_csv='/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/valid/valid_imgs.tsv'
        else:
            self.read_file =  '/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/test/test_texts.jsonl'
            self.read_csv='/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/test/test_imgs.tsv'
        self.image_list=load_csv(self.read_csv)

        # with open(self.read_file,'r') as f:
        #     for line in f:
        #         img_path = os.path.join(self.img_root,line.strip() + '.jpg')
        #         label = line.strip().split('/')[0]
        #         label = label.replace("_"," ")
        #         label = "a photo of " + label
        #         self.samples.append(img_path)
        #         self.sam_labels.append(label)
        # 转换为token
        # self.tokens = open_clip.tokenize(self.sam_labels)

        # text_id = 1
        # line_index = 0
        # NotRead_line = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]
        # with open(self.read_file, 'r+', encoding='utf-8') as f:
        #     for line in f:
        #         line_index += 1
        #         if line_index in NotRead_line:
        #             continue
        #         data = json.loads(line)
        #         text = data["text"]
        #         id_list = data["image_ids"]
        #         for id in id_list:
        #             self.samples.append(self.image_list[id - 1])
        #             self.sam_labels.append(text)
        #             self.labels.append(text_id)
        #         text_id += 1
        # self.tokens = clip.tokenize(self.sam_labels)


        line_index = 0
        # NotRead_line = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]
        with open(self.read_file, 'r+', encoding='utf-8') as f:
            for line in f:
                line_index += 1
                # if line_index in NotRead_line:
                #     continue
                data = json.loads(line)
                text = data["text"]
                id_list = data["image_ids"]
                for id in id_list:
                    self.samples.append(self.image_list[id - 1])
                    self.sam_labels.append(text)
                    self.labels.append(line_index)

        self.tokens = clip.tokenize(self.sam_labels)





    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        token = self.tokens[idx]
        label=self.labels[idx]
        # 加载图像
        image = img.convert('RGB')
        # 对图像进行转换
        image = self.img_process(image)

        return image,token,label