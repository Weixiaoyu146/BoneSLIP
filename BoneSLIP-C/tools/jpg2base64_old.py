from PIL import Image
from io import BytesIO
import base64
import sys
import cv2
import os
import glob

def image_generate(labels, body_part, file_name, begin, strategy, output_path):
    # this_path = r'/hdd3/hdd/hdd2/wxy/dataset/CLIP/Chinese_CLIP/' + body_part + labels
    # if os.path.exists(this_path):
    #     raise Exception('%s已经存在，不可创建' %this_path)
    # os.makedirs(this_path)
    file = open(output_path + '/' + strategy + '_imgs.tsv', 'a')
    img = Image.open(file_name) # 访问图片路径
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    # print(base64_str)
    tab = '\t'
    file.write(begin + tab + base64_str + '\n')

    # print(file_name + " processed successful!")
    file.close()

def txt_generate(labels, body_parts, image_path, begin, strategy, output_path):

    # def getImageVar(image):
    #     imageVar = cv2.Laplacian(image, cv2.CV_64F).var()
    #     return imageVar

    if body_parts[0:-3] == "chestL":
        body_part = "左肋骨组"
    elif body_parts[0:-3] == "chestR":
        body_part = "右肋骨组"
    elif body_parts[0:-3] == "ankleL":
        body_part = "左踝关节"
    elif body_parts[0:-3] == "ankleR":
        body_part = "右踝关节"
    elif body_parts[0:-3] == "elbowL":
        body_part = "左肘关节"
    elif body_parts[0:-3] == "elbowR":
        body_part = "右肘关节"
    elif body_parts[0:-3] == "head":
        body_part = "头"
    elif body_parts[0:-3] == "kneeL":
        body_part = "左膝关节"
    elif body_parts[0:-3] == "kneeR":
        body_part = "右膝关节"
    elif body_parts[0:-3] == "pelvis":
        body_part = "骨盆"
    elif body_parts[0:-3] == "shoL":
        body_part = "左肩关节"
    elif body_parts[0:-3] == "shoR":
        body_part = "右肩关节"
    elif body_parts[0:-3] == "ver":
        body_part = "脊椎"
    else:
        print("illegal body_parts!")
        sys.exit(1)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess_table = ['头', '左肩关节', '右肩关节', '脊椎', '左肋骨组', '右肋骨组', '左肘关节', '右肘关节', '骨盆', '左膝关节', '右膝关节', '左踝关节', '右踝关节']
    tab = '\t'
    # text = "image_path:\n" + tab + image_path + "\n" + "txt:\n"

    # CT图像清晰度判断
    # imgVar = getImageVar(image)
    # if imgVar >= 95:
    #     # text = tab + '全身骨骼显像清晰，对比度好，'
    #     text = body_part + '骨骼显像清晰，对比度好，'
    # else:
    #     # text = tab + '全身骨骼显示欠清晰，对比度差，'
    #     text = body_part + '骨骼显示欠清晰，对比度差，'

    if labels == "Abnormal":
        # text += '可见' + body_part + '呈示踪迹摄取增高影。'
        text = '可见' + body_part + '呈示踪迹摄取增高影。'
    elif labels == "Normal":
        # text += '可见' + body_part + '骨骼示踪剂摄取适中，分布均匀、对称，未见' + body_part + '骨骼示踪剂异常浓聚、稀疏或缺如影。'
        text = '可见' + body_part + '骨骼示踪剂摄取适中，分布均匀、对称，未见' + body_part + '骨骼示踪剂异常浓聚、稀疏或缺如影。'
    else:
        print("illegal labels!")
        sys.exit(1)
    # text += "双肾略显影，位置正常。"

    this_path = output_path + '/' + strategy + '_texts.jsonl'
    # if os.path.exists(this_path):
    #     raise Exception('%s已经存在，不可创建' %this_path)
    # os.makedirs(this_path)
    file = open(this_path, 'a')
    img = Image.open(file_name) # 访问图片路径
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    # print(base64_str)
    tab = '\t'
    file.write("{\"text_id\": " + begin + ", \"text\": \"" + text + "\", \"image_ids\": [" + begin +']}\n')

    # print(file_name + " processed successful!")
    file.close()
if __name__ == "__main__":
    # labels = "Abnormal"   #标签（异常/正常）
    # # labels = "Normal"  # 标签（异常/正常）
    begin = 0           #编号开始位置，默认0(1)
    # body_part = 'chestLANT'  # 身体部位，分前后身（例：chestLANT）
    #
    strategy = 'valid' #输出策略：all/train/test/val
    #
    # data_path = r"/hdd3/hdd/hdd2/wxy/dataset/CLIP/" + body_part + r'/' + labels
    # image_paths = glob.glob(data_path + "/*")
    # data_path = r"/hdd/wxy/dataset/CLIP/Chinese_CLIP/mianyang+huaxi/up1/"
    data_path = r"/hdd/wxy/dataset/CLIP/Chinese_CLIP/mianyang+huaxi/" + strategy + '/'

    output_path = r'/hdd/wxy/dataset/CLIP/Chinese_CLIP/' + strategy
    if os.path.exists(output_path):
        raise Exception('%s已经存在，不可创建' %output_path)
    os.makedirs(output_path)
    for body_path in glob.glob(data_path + "*"):
        body_part = body_path.split('/')[-1]
        for label_path in glob.glob(body_path + "/*"):
            labels = label_path.split('/')[-1]
            for file_name in glob.glob(label_path + "/*"):
                begin += 1
                # print(file_name)
                image_generate(labels, body_part, file_name, str(begin), strategy, output_path)
                txt_generate(labels, body_part, file_name, str(begin), strategy, output_path)
            print(label_path + " processed successful!")

    # for file_name in image_paths:
    #     begin += 1
    #     image_generate(labels, body_part, file_name, str(begin), strategy)
    #     txt_generate(labels, body_part, file_name, str(begin), strategy)