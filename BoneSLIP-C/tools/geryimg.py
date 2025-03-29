import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

# 导入包

gray_img = cv2.imread(r"D:\transfer\test-image\4036.jpg", 0)  # 0表示读入的为灰度图，1表示为RGB三通道图像
# 读入图片（单通道灰度图）
# cv2.imshow("123",gray_img)
# cv2.waitKey()
photoshape_img = np.zeros((gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
#
r1, s1 = 40, 100
r2, s2 = 200, 250
k1 = s1 / r2  # 第一段函数斜率
k2 = (s2 - s1) / (r2 - r1)  # 第二段函数斜率
k3 = (255 - s2) / (255 - r2)  # 第三段函数斜率
# 构造跟原图等大小的零值图，用于保存分段线性变换结果；设置分段线性函数参数；

for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        if r1 <= gray_img[i, j] <= r2:
            photoshape_img[i, j] = k2 * (gray_img[i, j] - r1) + s1
        elif gray_img[i, j] < r1:
            photoshape_img[i, j] = k1 * gray_img[i, j]
        elif gray_img[i, j] > r2:
            photoshape_img[i, j] = k3 * (gray_img[i, j] - r2) + s2
# cv2.imshow('12',photoshape_img)
# cv2.waitKey()
# 遍历图像每个像素值，按照公式进行分段线性变换；

gamma_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 1), dtype=np.float32)
# 构造跟原图等大小的零值图，用于保存伽马变换结果；
for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        gamma_img[i, j] = math.pow(gray_img[i, j], 0.5)
cv2.normalize(gamma_img, gamma_img, 0, 255, cv2.NORM_MINMAX)
gamma_img = cv2.convertScaleAbs(gamma_img)
# cv2.imshow('12',gamma_img)
# cv2.waitKey()
# 遍历图像每个像素值，按照公式进行伽马变换；

# plt.subplot(321);
# plt.imshow(gray_img, 'gray');
# plt.title("srcImg")
# plt.subplot(322);
# plt.hist(gray_img.ravel(), 256, [0, 256]),
# plt.title("Histogram");
# plt.xlim([0, 256])
#
# plt.subplot(323), plt.imshow(photoshape_img, 'gray'), plt.title("picewise")
# plt.subplot(324), plt.hist(photoshape_img.ravel(), 256, [0, 256]),
# plt.title("Histogram"), plt.xlim([0, 256])
#
# plt.subplot(325), plt.imshow(gamma_img, 'gray'), plt.title("gamma")
# plt.subplot(326), plt.hist(gamma_img.ravel(), 256, [0, 256]),
# plt.title("Histogram"), plt.xlim([0, 256])
plt.imshow(gamma_img, 'gray')
plt.tight_layout()  # 会自动调整子图参数,使之填充整个图像区域
plt.show()