import os

import pydicom
from PIL import Image
import numpy as np

def dcm_to_jpg(dcm_file_path, jpg_file_path):
    # 加载DICOM文件
    ds = pydicom.dcmread(dcm_file_path)
    # 尝试从DICOM元数据中获取窗宽和窗位值
    try:
        window_center, winqdow_width = ds.WindowCenter, ds.WindowWidth
    except AttributeError:
        # 如果DICOM没有窗宽和窗位信息，使用默认值或自定义
        window_center, window_width = np.median(ds.pixel_array), np.max(ds.pixel_array) - np.min(ds.pixel_array)
    # 转换为numpy数组
    pixel_array_numpy = ds.pixel_array.astype(float)
    # 将像素值调整到窗宽和窗位
    pixel_array_numpy = pixel_array_numpy - (window_center - window_width / 2)
    pixel_array_numpy = np.clip(pixel_array_numpy, 0, window_width)
    pixel_array_numpy = pixel_array_numpy / window_width * 255.0

    # 转换为整型
    pixel_array_numpy = np.uint8(pixel_array_numpy)
    # 将numpy数组转换为Pillow图片格式
    image = Image.fromarray(pixel_array_numpy)

    # 保存为JPG文件
    image.save(jpg_file_path)

if __name__ == '__main__':
    dcm_file_path = r'D:\dataset\kaggle\vindr-cxr\physionet.org\files\vindr-cxr\1.0.0\BoneSLIP\dicom\Normal'
    jpg_file_path = r'D:\dataset\kaggle\vindr-cxr\physionet.org\files\vindr-cxr\1.0.0\BoneSLIP\jpg\Normal\\'
    for filename in os.listdir(dcm_file_path):
        file_path = os.path.join(dcm_file_path, filename)
        flag = file_path.split('\\')[-1]
        flag = flag.split('.')[0]
        dcm_to_jpg(file_path, jpg_file_path + flag + '.jpg')