import csv
from PIL import Image
import base64
import io
import json

# def convert_and_save_image(base64_str, output_filename):
def convert_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    # image.save(output_filename)
    return image

csv_reader = csv.reader(open("/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/train/train_imgs.tsv"))
i=0
image_list=[]
for row in csv_reader:
    if i<100:
        i=i+1
        print(row)
        data=str(row[0]).split('	')
        print(data[0]) #id
        print(data[1]) #base64
        image=convert_image(data[1])
        image_list.append(image)
