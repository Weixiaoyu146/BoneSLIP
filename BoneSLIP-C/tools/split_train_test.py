import os
import random
import glob

class SplitFiles():
    """按行分割文件"""

    def __init__(self, file_name, train_num, test_num, labels, random_num):
        """初始化要分割的源文件名和分割后的文件行数"""
        self.file_name = file_name
        self.train_num = train_num
        self.test_num = test_num
        self.labels = labels
        self.random_num = random_num

    def split_file(self):
        if self.file_name and os.path.exists(self.file_name):
            try:
                with open(self.file_name) as f:  # 使用with读文件
                    temp_count = 1
                    for line in f:
                        if temp_count in random_num:
                            self.write_file('test', line)
                        else:
                            self.write_file('train', line)
                        temp_count += 1

            except IOError as err:
                print(err)
        else:
            print("%s is not a validate file" % self.file_name)

    def get_part_file_name(self, part_name):
        """"获取分割后的文件名称：在源文件相同目录下建立临时文件夹temp_part_file，然后将分割后的文件放到该路径下"""
        temp_path = os.path.dirname(self.file_name)  # 获取文件的路径（不含文件名）
        file_folder = temp_path + "/temp_part_file"
        if not os.path.exists(file_folder):  # 如果临时目录不存在则创建
            os.makedirs(file_folder)
        part_file_name = file_folder + "/" + labels + "_" + str(part_name) + "_list.txt"
        return part_file_name

    def write_file(self, part_num, line):
        """将按行分割后的内容写入相应的分割文件中"""
        part_file_name = self.get_part_file_name(part_num)
        try:
            with open(part_file_name, "a") as part_file:
                part_file.writelines(line)
        except IOError as err:
            print(err)

def get_random():
    """生成随机数组，随机划分 （0，190001）txt标签行数， 7600测试集标签行数"""
    random_num = random.sample(range(0, train_num), test_num)

    return random_num

if __name__ == "__main__":
    strategy = 'all' #输出策略：all/train/test/val
    data_path = "/hdd/wxy/dataset/CLIP/Chinese_CLIP/all/"  # 你要划分的数据集
    train_num = 160362
    test_num = 32072
    file_path = glob.glob(data_path + '*')
    random_num = get_random()
    for path in file_path:
        labels = path.split('_')[2][0:4]
        file = SplitFiles(path, train_num, test_num, labels, random_num)
        file.split_file()