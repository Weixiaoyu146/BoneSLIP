import pandas as pd
import os
import shutil
import numpy as np

folder_path = r'D:\dataset\kaggle\vindr-cxr\physionet.org\files\vindr-cxr\1.0.0\BoneSLIP\all'
files = os.listdir(folder_path)

df = pd.read_excel(r'D:\dataset\kaggle\vindr-cxr\physionet.org\files\vindr-cxr\1.0.0\annotations\image_labels_test.xlsx', engine='openpyxl', sheet_name='image_labels_test', header = 0)
tumor = df[df['Lung tumor'] == 1]
nodule = df[df['Nodule/Mass'] == 1]
# print(nodule)
# print(tumor['image_id'])
tumor_list = tumor['image_id'].values.tolist()
nodule_list = nodule['image_id'].values.tolist()
final_list = tumor_list + nodule_list
final_list = list(np.unique(final_list))
# print(len(final_list))
#
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    file = file_path.split('\\')[-1]
    image = file.split('.')[0]
    if image in final_list:
        shutil.copy(file_path, r'D:\dataset\kaggle\vindr-cxr\physionet.org\files\vindr-cxr\1.0.0\BoneSLIP\nodule\Abnormal\\' + file)
    else:
        shutil.copy(file_path, r'D:\dataset\kaggle\vindr-cxr\physionet.org\files\vindr-cxr\1.0.0\BoneSLIP\nodule\Normal\\' + file)
print("Process success!")