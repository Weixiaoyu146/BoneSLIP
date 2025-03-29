import os
import re
import sys
import glob

#为了示例方便，将待修改与当前工作目录取同一个
def ReName():
	data_path = r"D:\dataset\mianyang+huaxi\binary_test-clip"

	# for body_path in glob.glob(data_path + "\*"):
	# 	# body_part = body_path.split(os.sep)[-1][-3:]
	# 	# print(body_part)
	# 	if body_part == "PST":
	for label_path in glob.glob(data_path + "\*"):
		for file_name in glob.glob(label_path + "\*"):
			print(label_path + os.sep + "0" + file_name.split(os.sep)[4] + file_name.split(os.sep)[-1])
			os.rename(file_name, label_path + os.sep + "0" + file_name.split(os.sep)[4] + file_name.split(os.sep)[-1])
			# print(label_path + " processed successful!")
	#
	# for i in range(0,len(fileList)):
	# 	os.rename(fileList[i],str(("%05d"%i))+'.'+'jpg')#文件重新命名
	# 	# 如果想实现000000~999999，只需将这里的5改为6，诸如此类。
	# print("\n")
	# os.chdir(currentPath)		#改回程序运行前的工作目录
	# sys.stdin.flush()		#更新
	# print("修改后："+str(os.listdir(r"F://JPEGImages")))		#修改后的文件
ReName()