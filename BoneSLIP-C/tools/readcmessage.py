import pydicom
from PIL import Image
import glob
# 读取dcm文件

data_path = r"D:\dataset\upload\dcm\\"
flag = 0
dcm_num = 0
m35, m36, m46, m56, m66, m76, m85 = 0, 0, 0, 0, 0, 0, 0
f35, f36, f46, f56, f66, f76, f85 = 0, 0, 0, 0, 0, 0, 0
for body_path in glob.glob(data_path + "*"):
    body_part = body_path + r"\Dicoms\\"
    for dcm_path in glob.glob(body_part + "*"):
        dcm = pydicom.dcmread(dcm_path)
        dcm_num += 1
        # 获取患者姓名、性别、年龄信息
        name = dcm.PatientName
        sex = dcm.PatientSex
        age = dcm.PatientAge
        # print("Name: ", name)
        # print("Sex: ", sex)
        # print("Age: ", age)
        if(age[-1] != "Y"):
            # print("Age: ", int(age[:2]))
            flag = int(age[:2])
        else:
            flag = int(age[:-1])
            # if(age[:3] <= 35)
            # print("Name: ", name)
            # print("Sex: ", sex)
            # print("Age: ", int(age[:-1]))
        if(flag <= 35):
            if(sex == 'M'):
                m35 += 1
            elif(sex == 'F'):
                f35 += 1
        elif(flag >= 36 and flag <= 45):
            if(sex == 'M'):
                m36 += 1
            elif(sex == 'F'):
                f36 += 1
        elif(flag >= 46 and flag <= 55):
            if(sex == 'M'):
                m46 += 1
            elif(sex == 'F'):
                f46 += 1
        elif(flag >= 56 and flag <= 65):
            if(sex == 'M'):
                m56 += 1
            elif(sex == 'F'):
                f56 += 1
        elif(flag >= 66 and flag <= 75):
            if(sex == 'M'):
                m66 += 1
            elif(sex == 'F'):
                f66 += 1
        elif(flag >= 76 and flag <= 85):
            if(sex == 'M'):
                m76 += 1
            elif(sex == 'F'):
                f76 += 1
        elif(flag > 85):
            if(sex == 'M'):
                m85 += 1
            elif(sex == 'F'):
                f85 += 1
print("Age <= 35: Male ", m35, "; Female ", f35)
print("36 <= Age <= 45: Male ", m36, "; Female ", f36)
print("46 <= Age <= 55: Male ", m46, "; Female ", f46)
print("56 <= Age <= 65: Male ", m56, "; Female ", f56)
print("66 <= Age <= 75: Male ", m66, "; Female ", f66)
print("76 <= Age <= 85: Male ", m76, "; Female ", f76)
print("Age > 85: Male ", m85, "; Female ", f85)
print("Total number = ", dcm_num)

    # 将影像数据保存成图片
    # img = Image.fromarray(dcm.pixel_array)
    # img.save('7L000.png')