import os
from PIL import Image
from torch.utils.data import Dataset
import clip


class MyDataset(Dataset):
    def __init__(self, root_dir,preprocess):
        self.img_process = preprocess
        self.root_dir = root_dir
        self.image_list = []
        self.labels = []
        self.texts=[]
        self.label_to_text = {0: "No right chest abnormalities seen",
                              1: "Abnormalities of the right chest can be seen"}

        # 获取Normal文件夹中的图像
        normal_dir = os.path.join(root_dir, 'Normal')
        normal_images = os.listdir(normal_dir)
        self.image_list.extend([os.path.join(normal_dir, img) for img in normal_images])
        self.labels.extend([0] * len(normal_images))  # 使用0表示正常图像

        # 获取Abnormal文件夹中的图像
        abnormal_dir = os.path.join(root_dir, 'Abnormal')
        abnormal_images = os.listdir(abnormal_dir)
        self.image_list.extend([os.path.join(abnormal_dir, img) for img in abnormal_images])
        self.labels.extend([1] * len(abnormal_images))  # 使用1表示异常图像
        for label in self.labels:
            text = self.label_to_text[label]
            self.texts.append(text)

        self.tokens = clip.tokenize(self.texts)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path).convert('RGB')
        image = self.img_process(image)
        # label = self.labels[index]
        # text = self.label_to_text[label]
        token=self.tokens[index]
        label=self.labels[index]

        # 在这里你可以执行任何预处理操作，例如转换为张量等

        return image, token, label