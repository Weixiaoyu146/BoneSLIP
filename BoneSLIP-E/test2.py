import jsonlines
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

csv_reader = csv.reader(open("/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/train/train_imgs.tsv"))
j=0
image_list=[]
for row in csv_reader:
    if j<5:
        j=j+1
        print(row)
        data = str(row[0]).split('	')
        print(data[0])  # id
        print(data[1])  # base64
        image = convert_image(data[1])
        image_list.append(image)
	else :




with open(r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/train/train_texts.jsonl','r+', encoding='utf-8') as f:
    # for item in jsonlines.Reader(f): # 每一行读取后都是一个json，可以按照key去取对应的值
    # 	print(item)
    # i=0
    # for line in f:
    # 	if i<1:
    # 		i=i+1
    # 		print(line)
    # 		print(type(line))
    samples=[]
    sample_labels=[]
    i=0
    for line in f:
        if i<1:
            i=i+1
            data=json.loads(line)
            text = data["text"]
            id_list = data["image_ids"]
            for id in id_list[:3]:
                samples.append(image_list[id-1])
        # 	print(data)
        # 	print(data["text"])
        # 	print(data["image_ids"])
