import sys
import cv2
import os
import glob

def get_random():
    train_num = 160362
    test_num = 32072
    """生成随机数组，随机划分 （0，190001）txt标签行数， 7600测试集标签行数"""
    random_num = random.sample(range(0, train_num), test_num)

    return random_num
if __name__ == "__main__":
    # labels = "Abnormal"   #标签（异常/正常）
    # # labels = "Normal"  # 标签（异常/正常）
    begin = 0           #编号开始位置，默认0(1)
    # body_part = 'chestLANT'  # 身体部位，分前后身（例：chestLANT）
    #
    strategy = 'all' #输出策略：all/train/test/val
    #
    # data_path = r"/hdd3/hdd/hdd2/wxy/dataset/CLIP/" + body_part + r'/' + labels
    # image_paths = glob.glob(data_path + "/*")
    data_path = r"/hdd/wxy/dataset/CLIP/Chinese_CLIP/mianyang+huaxi/up1/"

    output_path = r'/hdd/wxy/dataset/CLIP/Chinese_CLIP/' + strategy
    if os.path.exists(output_path):
        raise Exception('%s已经存在，不可创建' %output_path)
    os.makedirs(output_path)
    flag = get_random()
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