# -*- coding: utf-8 -*-
from tqdm import tqdm
import argparse
import json

input_path=r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/test/test_texts.jsonl'

if __name__ == "__main__":

    t2i_record = dict()

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            text_id = obj['text_id']
            image_ids = obj['image_ids']
            for image_id in image_ids:
                if image_id not in t2i_record:
                    t2i_record[image_id] = []
                t2i_record[image_id].append(text_id)

    with open(input_path.replace(".jsonl", "") + ".tr.jsonl", "w", encoding="utf-8") as fout:
        for image_id, text_ids in t2i_record.items():
            out_obj = {"image_id": image_id, "text_ids": text_ids}
            fout.write("{}\n".format(json.dumps(out_obj)))

    print("Done!")